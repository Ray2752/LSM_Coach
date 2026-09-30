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
import numpy as np

from dynamic import COOLDOWN_S, MAX_S, MIN_PROB, SHOW_S, STILL, GestureWindow, classify
from evaluation import (CONFIG, ERROR_SUFFIX, FINGER_LABEL, SAME_SHAPE, Issue, evaluate,
                        geometry_issues, shape_issue)
from signs import DYNAMIC, NIVEL_1, WORDS
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
    def __init__(self, tolerances, target=NIVEL_1[0], require_imu=True, model=None, dyn_model=None,
                 word_model=None, hands_complexity=1, classify_every=1):
        self.tolerances = tolerances
        self.require_imu = require_imu
        # En la UNO Q el bosque tarda ~50 ms: clasificar cada N fotogramas y reutilizar la última
        # probabilidad (ya se promedian los últimos SMOOTH_FRAMES, así que no cambia el veredicto)
        self.classify_every = max(1, int(classify_every))
        self._classify_i = 0
        self.model = model        # clasificador de landmarks (opcional)
        self.dyn_model = dyn_model  # clasificador de trazos para J, K, Ñ, Q, X, Z (opcional)
        self.word_model = word_model  # señas de palabras del Nivel 3 (opcional)
        self.gesture = GestureWindow()
        self._dyn_result = None   # último trazo clasificado: {"t", "best", "prob", "ok"}
        self.face_det = None      # detector de rostro (se crea al elegir una palabra)
        self._face, self._frame_i = None, 0
        self.face_t = 0.0         # cuándo se vio el rostro por última vez
        self._hand_pos, self.hand_t = None, 0.0  # centro de la palma (0-1) y cuándo se vio: cámara motorizada
        self.track_face = False   # detectar el rostro siempre (cámara motorizada que sigue a la persona)
        self._record_word = None  # {"person"}: la próxima seña se guarda como muestra
        self.notify = None        # callable(texto, kind) que pone el servidor para avisar
        self.proba_hist = deque(maxlen=SMOOTH_FRAMES)
        # model_complexity 0 = modelo ligero de manos (~2x más rápido, algo menos preciso): UNO Q
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                                              model_complexity=hands_complexity)
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        tol = self.tolerances.get(self.target)
        if self.track_face or (self.target in WORDS and (self._bundle() is not None or self._record_word)):
            self._update_face(rgb, now)  # ubicación respecto al rostro (Nivel 3) y seguimiento de cámara
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
            if self._bundle() is not None or self._record_word:
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
        palm = [lm.landmark[i] for i in (0, 5, 9, 13, 17)]  # muñeca y nudillos: centro de la palma
        self._hand_pos = (sum(p.x for p in palm) / 5, sum(p.y for p in palm) / 5)
        self.hand_t = now
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

        if self._bundle() is not None or self._record_word:  # letra con movimiento o palabra
            return self._process_dynamic(state, now, features, raw, lm, state["handedness"], imu)

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

    def _bundle(self):
        """Modelo de secuencias que aplica a la seña objetivo, o None si no hay."""
        if self.target in DYNAMIC:
            return self.dyn_model
        if self.target in WORDS and self.word_model and self.target in self.word_model["signs"]:
            return self.word_model
        return None

    def _update_face(self, rgb, now):
        """Detecta el rostro cada 3 fotogramas (es estable y así no frena la visión)."""
        self._frame_i += 1
        if self._frame_i % 3 != 1:
            return
        if self.face_det is None:
            # 0 = corto alcance (hasta ~2 m), 1 = largo alcance (hasta ~5 m): con la cámara
            # motorizada la persona puede estar más lejos.
            self.face_det = mp.solutions.face_detection.FaceDetection(model_selection=1 if self.track_face else 0,
                                                                      min_detection_confidence=0.5)
        res = self.face_det.process(rgb)
        if res.detections:
            b = res.detections[0].location_data.relative_bounding_box
            self._face = (b.xmin + b.width / 2, b.ymin + b.height / 2, b.width, b.height)
            self.face_t = now

    @property
    def busy(self):
        """True mientras se graba o evalúa un trazo: mover la cámara lo alteraría."""
        return self.gesture.active or self._record_word is not None

    def arm_word_record(self, person):
        """La próxima seña completa de la palabra objetivo se guarda como muestra."""
        self._record_word = {"person": person}
        self.gesture.reset()

    def _save_word_sample(self, segment):
        """Guarda el trazo en samples_words/<PALABRA>/ (formato de extract_words.py)."""
        import csv
        import glob
        import os
        from train_words import DATA_DIR
        word, person = self.target, self._record_word["person"]
        os.makedirs(os.path.join(DATA_DIR, word), exist_ok=True)
        n = len(glob.glob(os.path.join(DATA_DIR, word, "*.npz")))
        name = f"{person}-live-{n:03d}"
        np.savez_compressed(os.path.join(DATA_DIR, word, f"{name}.npz"), feats=segment["feats"],
                            angles=segment["angles"], wrist=segment["wrist"], face=segment["face"],
                            imu=segment["imu"], ok=segment["ok"])
        index = os.path.join(DATA_DIR, "index.csv")
        new = not os.path.exists(index)
        with open(index, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "word", "person", "source", "frames", "detected", "face_frames"])
            if new:
                w.writeheader()
            w.writerow({"file": name, "word": word, "person": person, "source": "live",
                        "frames": len(segment["ok"]), "detected": int(segment["ok"].sum()),
                        "face_frames": int(np.isfinite(segment["face"][:, 2]).sum())})
        return n + 1

    def _process_dynamic(self, state, now, features, raw, lm, handedness, imu=None):
        """Letras con movimiento y palabras: acumula el trazo de la muñeca y lo clasifica (o lo
        guarda como muestra) cuando la mano se detiene. Veredictos: ready, moving, ok / fix."""
        state["calibrated"] = self._bundle() is not None
        w, m = lm.landmark[0], lm.landmark[9]
        size = math.hypot(m.x - w.x, m.y - w.y)
        x = 1.0 - w.x if handedness == "Left" else w.x  # mano izquierda: reflejada, como los landmarks
        angles = [raw[f] for f in ("pulgar", "indice", "medio", "anular", "menique")]
        face = self._face if self.target in WORDS else None
        # inclinación del guante por fotograma (palabras): se guarda con la muestra y, si el
        # modelo se entrenó con guante, entra en la clasificación
        tilt = (imu["roll"], imu["pitch"]) if imu and imu.get("roll") is not None else None
        segment = self.gesture.feed(now, features, angles, (x, w.y, size), face, imu=tilt)
        alert, achievement = self._classify_segment(segment, now)
        self._dyn_state(state, now)
        return state, alert, achievement

    def _classify_segment(self, segment, now):
        """Clasifica un trazo terminado (o lo guarda, si se pidió grabar). Devuelve (vibrar, logro)."""
        if segment is None:
            return False, None
        if self._record_word:
            n = self._save_word_sample(segment)
            self._record_word = None
            if self.notify:
                glove = int(np.isfinite(segment["imu"]).all(axis=1).sum()) >= 4
                self.notify(f"Seña de {self.target} guardada ({n} en total, {segment['duration']} s, "
                            f"{'con' if glove else 'sin'} guante)")
            self._dyn_result = None
            return False, None
        bundle = self._bundle()
        if bundle is None:
            return False, None
        if self._dyn_result and now - self._dyn_result["t"] < COOLDOWN_S:
            return False, None  # el movimiento de "recoger" la mano no es un intento nuevo
        best, prob = classify(bundle, segment)
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
        if self._record_word:  # grabando una muestra: solo se indica en qué va
            state["recording"] = True
            state["verdict"] = "moving" if self.gesture.active else "ready" if state["hand"] else "nohand"
            state["hold"] = round(min(1.0, self.gesture.elapsed / MAX_S), 2) if self.gesture.active else 0.0
            return
        if r and now - r["t"] < SHOW_S:
            state["verdict"] = "ok" if r["ok"] else "fix"
            state["done"] = r["ok"]
            state["hold"] = 1.0 if r["ok"] else 0.0
            state["motion"] = {"best": r["best"], "prob": round(r["prob"], 2)}
            if not r["ok"]:
                word = self.target in WORDS
                name = lambda s: s if s in WORDS else f"la {s}"
                action = ((f"Falta el movimiento: haz la seña completa de {self.target}" if word
                           else f"Falta el movimiento: haz el trazo completo de la {self.target}") if r["best"] == STILL
                          else f"{'La seña' if word else 'El trazo'} se parece a {name(r['best'])}" if r["best"] != self.target
                          else f"Repite {'la seña de ' + self.target if word else 'el trazo de la ' + self.target} más claro")
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
        self._classify_i += 1
        if self._classify_i % self.classify_every == 0 or not self.proba_hist:
            self.proba_hist.append(self.model.predict_proba([features])[0])
        else:
            self.proba_hist.append(self.proba_hist[-1])  # fotograma intermedio: repite la última
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
