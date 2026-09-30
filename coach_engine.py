"""Motor de LSM Coach sin interfaz: procesa un fotograma y devuelve el estado.

Lo usa el servidor web (web_server.py). Fusiona cámara (dedos) y muñequera
(orientación) con evaluation.evaluate. Si hay clasificador (model.joblib), además exige
que la forma de la mano se reconozca como la letra objetivo. También decide:
  - cuándo vibrar (solo si el error persiste, sin repetir de más)
  - cuándo contar la seña como lograda (registro de progreso del aprendiz)
"""
import math
import time
from collections import deque
from dataclasses import asdict

import cv2
import mediapipe as mp

from dynamic import MAX_S, MIN_PROB, SHOW_S, STILL, GestureWindow, classify
from evaluation import (CONFIG, ERROR_SUFFIX, FINGER_LABEL, SAME_SHAPE, Issue, evaluate,
                        geometry_issues, shape_issue)
from signs import DYNAMIC, NIVEL_1
from tolerance_calculator import (FINGER_JOINTS, landmarks_to_angles, landmarks_to_features,
                                  mirror_features)

SMOOTH_FRAMES = 5
PERSIST_S = 0.6   # el error debe durar esto para avisar (evita avisos por parpadeos)
REPEAT_S = 6.0    # si sigues con el MISMO error, se repite hasta pasado este tiempo
MIN_GAP_S = 1.5   # pausa mínima entre dos avisos cualquiera
HOLD_OK_S = 1.0   # la seña debe mantenerse correcta este tiempo para contarla como lograda
SETTLE_S = 1.2    # al aparecer la mano o cambiar de letra: tiempo para formar la seña sin evaluar
FIX_STABLE_S = 0.5  # un error debe persistir esto antes de marcarse (evita parpadeos en rojo)
OK_STABLE_S = 0.15  # lo correcto se muestra casi de inmediato (para que el anillo arranque)
RED = (80, 80, 255)


class AlertPolicy:
    """Decide cuándo avisar (vibrar) de un error."""

    def __init__(self):
        self.cur_key, self.cur_since = None, 0.0
        self.last_key, self.last_t = None, 0.0

    def reset(self):
        self.cur_key = self.last_key = None

    def update(self, key, now):
        """key: el error actual (None si todo está bien). Devuelve True si hay que avisar."""
        if key != self.cur_key:
            self.cur_key, self.cur_since = key, now
        if key is None:
            self.last_key = None  # ya está bien: el próximo error avisa de inmediato
            return False
        if (now - self.cur_since >= PERSIST_S
                and now - self.last_t >= MIN_GAP_S
                and (key != self.last_key or now - self.last_t >= REPEAT_S)):
            self.last_key, self.last_t = key, now
            return True
        return False


class Coach:
    def __init__(self, tolerances, target=NIVEL_1[0], require_imu=True, model=None, dyn_model=None):
        self.tolerances = tolerances
        self.require_imu = require_imu
        self.model = model        # clasificador de landmarks (opcional)
        self.dyn_model = dyn_model  # clasificador de trazos para J, K, Ñ, Q, X, Z (opcional)
        self.gesture = GestureWindow()
        self._dyn_result = None   # último trazo clasificado: {"t", "best", "prob", "ok"}
        self.proba_hist = deque(maxlen=SMOOTH_FRAMES)
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
        self.history = {f: deque(maxlen=SMOOTH_FRAMES) for f in FINGER_JOINTS}
        self.feat_hist = deque(maxlen=SMOOTH_FRAMES)  # landmarks, para las reglas de forma
        self.alerts = AlertPolicy()
        self.progress = {}        # {seña: veces lograda en esta sesión}
        self.last_sample = None   # la última lectura cruda, para guardarla como muestra
        self._settle_until = 0.0  # hasta cuándo se le da tiempo de formar la seña
        self._was_hand = False
        self._cand, self._cand_since, self._stable = None, 0.0, None  # veredicto con histéresis
        self.set_target(target)

    def set_target(self, sign):
        self.target = sign
        self.alerts.reset()
        self.proba_hist.clear()
        self._settle_until = time.time() + SETTLE_S
        self._cand = self._stable = None
        self.gesture.reset()
        self._dyn_result = None
        self._new_attempt(time.time())

    def _stable_verdict(self, raw, now):
        """'ok'/'fix' solo cuando el veredicto crudo se sostiene un rato: un dedo que roza el
        límite un instante no debe poner la pantalla en rojo (ni vibrar)."""
        if raw != self._cand:
            self._cand, self._cand_since = raw, now
        need = OK_STABLE_S if raw == "ok" else FIX_STABLE_S
        if self._stable is None or now - self._cand_since >= need:
            self._stable = raw
        return self._stable

    def _new_attempt(self, now):
        self._attempt_start, self._ok_since = now, None
        self._counted, self._failed_seen = False, set()

    def process(self, frame, imu):
        """frame: BGR ya en espejo (se dibuja la mano encima). imu: {"roll","pitch","yaw"}
        o None si no hay lectura. Devuelve (estado, vibrar, logro o None)."""
        now = time.time()
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tol = self.tolerances.get(self.target)
        state = {"target": self.target, "calibrated": tol is not None, "hand": False,
                 "handedness": None, "fingers": [], "orientation": [], "issues": [],
                 "failed_parameters": [], "verdict": "nohand"}

        if not res.multi_hand_landmarks:
            for h in self.history.values():
                h.clear()
            self.feat_hist.clear()
            self.proba_hist.clear()
            self.last_sample = None
            self._was_hand = False
            self._cand = self._stable = None
            alert, achievement = False, None
            if self.target in DYNAMIC and self.dyn_model is not None:
                if self.gesture.active:  # la mano salió de cuadro: ahí terminó el trazo
                    alert, achievement = self._classify_segment(self.gesture.finish(now), now)
                self._dyn_state(state, now)  # el resultado se sigue mostrando aunque baje la mano
            self.gesture.reset()
            self.alerts.update(None, now)
            if self._counted:  # bajó la mano después de lograrla: empieza otro intento
                self._new_attempt(now)
            self._ok_since = None
            return state, alert, achievement

        lm = res.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        raw = landmarks_to_angles(lm.landmark)
        for f, a in raw.items():
            self.history[f].append(a)
        angles = {f: sum(v) / len(v) for f, v in self.history.items()}
        state["hand"] = True
        if not self._was_hand:  # la mano acaba de aparecer: tiempo para formar la seña
            self._settle_until = now + SETTLE_S
            self._cand = self._stable = None
        self._was_hand = True
        if res.multi_handedness:
            state["handedness"] = res.multi_handedness[0].classification[0].label
        features = landmarks_to_features(lm.landmark)
        if state["handedness"] == "Left":  # mano izquierda: se refleja para usar el mismo modelo
            features = mirror_features(features)
        self.last_sample = {"angles": raw, "landmarks": features,
                            "imu": dict(imu) if imu else None}

        if self.target in DYNAMIC and self.dyn_model is not None:
            return self._process_dynamic(state, now, features, raw, lm, state["handedness"])

        if tol is None:
            state["verdict"] = "uncalibrated"
            state["fingers"] = [{"id": f, "label": FINGER_LABEL[f], "value": round(a, 1)}
                                for f, a in angles.items()]
            return state, False, None

        # FUSIÓN: dedos (cámara) + orientación de la muñeca (IMU)
        result = evaluate(angles, imu, tol, require_imu=self.require_imu)
        self.feat_hist.append(features)
        smooth = [sum(v) / len(self.feat_hist) for v in zip(*self.feat_hist)]
        geometry = geometry_issues(smooth, self.target)  # dedos juntos (B), hueco (C)
        result.issues += geometry
        probs = self._shape_probs(self.last_sample["landmarks"])
        if probs:
            letters = {k: v for k, v in probs.items() if not k.endswith(ERROR_SUFFIX)}
            best = max(letters, key=letters.get)
            # G = L horizontal; una O con hueco claro es una C cerrada (si no hay hueco, sigue O)
            if best in SAME_SHAPE.get(self.target, ()) and not geometry:
                best = self.target
            state["shape"] = {"best": best,
                              "error": round(probs.get(self.target + ERROR_SUFFIX, 0.0), 2)}
            issue = shape_issue(probs, self.target)
            finger_wrong = any(i.parameter == CONFIG for i in result.issues)
            if issue and not finger_wrong:  # si ya falla un dedo, esa instrucción es más útil
                result.issues.append(issue)
        h, w = frame.shape[:2]
        for f, a in angles.items():
            ok = result.fingers[f][0]
            state["fingers"].append({"id": f, "label": FINGER_LABEL[f], "value": round(a, 1),
                                     "min": tol[f]["min"], "max": tol[f]["max"], "ok": ok})
            if not ok:  # marca la punta del dedo que falla
                tip = lm.landmark[FINGER_JOINTS[f][2]]
                cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 16, RED, 3)
        for axis, rng in (tol.get("orientacion") or {}).items():
            v = imu.get(axis) if imu else None
            state["orientation"].append({
                "axis": axis, "value": None if v is None else round(v, 1),
                "min": rng["min"], "max": rng["max"],
                "ok": v is not None and rng["min"] <= v <= rng["max"]})
        verdict = self._stable_verdict("ok" if result.ok else "fix", now)
        if now < self._settle_until:  # todavía está formando la seña: no se juzga
            verdict = "settling"
            state["settle"] = round((self._settle_until - now) / SETTLE_S, 2)
        state["issues"] = [asdict(i) for i in result.issues] if verdict == "fix" else []
        state["failed_parameters"] = result.failed_parameters if verdict == "fix" else []
        state["verdict"] = verdict

        alert = self.alerts.update(result.issues[0].action if verdict == "fix" and result.issues else None, now)
        achievement = self._track_progress(result, now, ok=verdict == "ok")
        # para el anillo de progreso de la interfaz: qué parte de HOLD_OK_S lleva sostenida
        state["hold"] = round(min(1.0, (now - self._ok_since) / HOLD_OK_S), 2) if self._ok_since else 0.0
        state["done"] = self._counted
        return state, alert, achievement

    def _process_dynamic(self, state, now, features, raw, lm, handedness):
        """Letras con movimiento: acumula el trazo de la muñeca y lo clasifica cuando la mano
        se detiene. Veredictos: ready (esperando el trazo), moving (grabando), ok / fix."""
        state["calibrated"] = True
        w, m = lm.landmark[0], lm.landmark[9]
        size = math.hypot(m.x - w.x, m.y - w.y)
        x = 1.0 - w.x if handedness == "Left" else w.x  # mano izquierda: reflejada, como los landmarks
        angles = [raw[f] for f in ("pulgar", "indice", "medio", "anular", "menique")]
        segment = self.gesture.feed(now, features, angles, (x, w.y, size))
        alert, achievement = self._classify_segment(segment, now)
        self._dyn_state(state, now)
        return state, alert, achievement

    def _classify_segment(self, segment, now):
        """Clasifica un trazo terminado. Devuelve (vibrar, logro)."""
        if segment is None:
            return False, None
        best, prob = classify(self.dyn_model, segment)
        ok = best == self.target and prob >= MIN_PROB
        self._dyn_result = {"t": now, "best": best, "prob": prob, "ok": ok}
        if ok:
            self.progress[self.target] = self.progress.get(self.target, 0) + 1
            return False, {"sign": self.target, "seconds": segment["duration"], "failed_parameters": []}
        return True, None  # una vibración: el trazo no fue el esperado

    def _dyn_state(self, state, now):
        """Veredicto de una letra con movimiento: el último resultado (SHOW_S), o si está
        grabando el trazo, o esperando a que empiece."""
        r = self._dyn_result
        if r and now - r["t"] < SHOW_S:
            state["verdict"] = "ok" if r["ok"] else "fix"
            state["done"] = r["ok"]
            state["hold"] = 1.0 if r["ok"] else 0.0
            state["motion"] = {"best": r["best"], "prob": round(r["prob"], 2)}
            if not r["ok"]:
                action = (f"Falta el movimiento: haz el trazo completo de la {self.target}" if r["best"] == STILL
                          else f"El trazo se parece a la {r['best']}" if r["best"] != self.target
                          else f"Repite el trazo de la {self.target} más claro")
                state["issues"] = [asdict(Issue(CONFIG, "movimiento", action))]
                state["failed_parameters"] = [CONFIG]
        elif self.gesture.active:
            state["verdict"] = "moving"
            state["hold"] = round(min(1.0, self.gesture.elapsed / MAX_S), 2)
        elif state["hand"]:
            state["verdict"] = "ready"

    def _shape_probs(self, features):
        """Probabilidades del clasificador promediadas en los últimos fotogramas."""
        if self.model is None:
            return None
        self.proba_hist.append(self.model.predict_proba([features])[0])
        n = len(self.proba_hist)
        return {str(c): sum(p[i] for p in self.proba_hist) / n
                for i, c in enumerate(self.model.classes_)}

    def _track_progress(self, result, now, ok):
        """Cuenta la seña como lograda una vez por intento, si se sostiene HOLD_OK_S.
        `ok` es el veredicto estable (no el crudo del fotograma)."""
        if not ok:
            self._failed_seen.update(result.failed_parameters)
            self._ok_since = None
            return None
        if self._ok_since is None:
            self._ok_since = now
        if self._counted or now - self._ok_since < HOLD_OK_S:
            return None
        self._counted = True
        self.progress[self.target] = self.progress.get(self.target, 0) + 1
        return {"sign": self.target, "seconds": round(now - self._attempt_start, 1),
                "failed_parameters": sorted(self._failed_seen)}
