"""Motor de LSM Coach sin interfaz: procesa un fotograma y devuelve el estado.

Lo usa el servidor web (web_server.py). Fusiona cámara (dedos) y muñequera
(orientación) con evaluation.evaluate. Si hay clasificador (model.joblib), además exige
que la forma de la mano se reconozca como la letra objetivo. También decide:
  - cuándo vibrar (solo si el error persiste, sin repetir de más)
  - cuándo contar la seña como lograda (registro de progreso del aprendiz)
"""
import time
from collections import deque
from dataclasses import asdict

import cv2
import mediapipe as mp

from evaluation import CONFIG, ERROR_SUFFIX, FINGER_LABEL, evaluate, shape_issue
from signs import NIVEL_1
from tolerance_calculator import (FINGER_JOINTS, landmarks_to_angles, landmarks_to_features,
                                  mirror_features)

SMOOTH_FRAMES = 5
PERSIST_S = 0.6   # el error debe durar esto para avisar (evita avisos por parpadeos)
REPEAT_S = 6.0    # si sigues con el MISMO error, se repite hasta pasado este tiempo
MIN_GAP_S = 1.5   # pausa mínima entre dos avisos cualquiera
HOLD_OK_S = 1.0   # la seña debe mantenerse correcta este tiempo para contarla como lograda
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
    def __init__(self, tolerances, target=NIVEL_1[0], require_imu=True, model=None):
        self.tolerances = tolerances
        self.require_imu = require_imu
        self.model = model        # clasificador de landmarks (opcional)
        self.proba_hist = deque(maxlen=SMOOTH_FRAMES)
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
        self.history = {f: deque(maxlen=SMOOTH_FRAMES) for f in FINGER_JOINTS}
        self.alerts = AlertPolicy()
        self.progress = {}        # {seña: veces lograda en esta sesión}
        self.last_sample = None   # la última lectura cruda, para guardarla como muestra
        self.set_target(target)

    def set_target(self, sign):
        self.target = sign
        self.alerts.reset()
        self.proba_hist.clear()
        self._new_attempt(time.time())

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
            self.proba_hist.clear()
            self.last_sample = None
            self.alerts.update(None, now)
            if self._counted:  # bajó la mano después de lograrla: empieza otro intento
                self._new_attempt(now)
            self._ok_since = None
            return state, False, None

        lm = res.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        raw = landmarks_to_angles(lm.landmark)
        for f, a in raw.items():
            self.history[f].append(a)
        angles = {f: sum(v) / len(v) for f, v in self.history.items()}
        state["hand"] = True
        if res.multi_handedness:
            state["handedness"] = res.multi_handedness[0].classification[0].label
        features = landmarks_to_features(lm.landmark)
        if state["handedness"] == "Left":  # mano izquierda: se refleja para usar el mismo modelo
            features = mirror_features(features)
        self.last_sample = {"angles": raw, "landmarks": features,
                            "imu": dict(imu) if imu else None}

        if tol is None:
            state["verdict"] = "uncalibrated"
            state["fingers"] = [{"id": f, "label": FINGER_LABEL[f], "value": round(a, 1)}
                                for f, a in angles.items()]
            return state, False, None

        # FUSIÓN: dedos (cámara) + orientación de la muñeca (IMU)
        result = evaluate(angles, imu, tol, require_imu=self.require_imu)
        probs = self._shape_probs(self.last_sample["landmarks"])
        if probs:
            letters = {k: v for k, v in probs.items() if not k.endswith(ERROR_SUFFIX)}
            state["shape"] = {"best": max(letters, key=letters.get),
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
        state["issues"] = [asdict(i) for i in result.issues]
        state["failed_parameters"] = result.failed_parameters
        state["verdict"] = "ok" if result.ok else "fix"

        alert = self.alerts.update(result.issues[0].action if result.issues else None, now)
        return state, alert, self._track_progress(result, now)

    def _shape_probs(self, features):
        """Probabilidades del clasificador promediadas en los últimos fotogramas."""
        if self.model is None:
            return None
        self.proba_hist.append(self.model.predict_proba([features])[0])
        n = len(self.proba_hist)
        return {str(c): sum(p[i] for p in self.proba_hist) / n
                for i, c in enumerate(self.model.classes_)}

    def _track_progress(self, result, now):
        """Cuenta la seña como lograda una vez por intento, si se sostiene HOLD_OK_S."""
        self._failed_seen.update(result.failed_parameters)
        if not result.ok:
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
