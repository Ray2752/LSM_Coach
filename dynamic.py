"""Señas con movimiento (J, K, Ñ, Q, X, Z) en vivo: detecta cuándo empieza y termina el trazo
de la mano y lo clasifica con model_dynamic.joblib (train_dynamic.py).

Umbrales medidos en el dataset (velocidad de la muñeca en "tamaños de mano por segundo"):
reposo ~0.4 (p90 1.2), pico del trazo ~5.6 (p10 2.5); el trazo dura ~1.5 s (0.6-3 s).
No depende de cámara ni de MediaPipe: se prueba con secuencias sintéticas.
"""
import math
from collections import deque

import numpy as np

START_V = 1.6   # empieza el trazo al superar esta velocidad (manos/s) dos veces seguidas
STOP_V = 1.0    # termina cuando baja de esta velocidad...
STOP_S = 0.6    # ...durante este tiempo (letras). Se evalúa SOLO al terminar: una pausa más
                # corta que esto no corta el trazo ni saca avisos a mitad
MIN_S = 0.4     # trazos más cortos se ignoran (un temblor no es una seña)
MAX_S = 4.0     # tope: se corta y se clasifica lo que haya (letras)
WORD_STOP_S = 0.9  # palabras y frases: llevan pausas dentro (GRACIAS, POR FAVOR) y son más largas
WORD_MAX_S = 5.5
PRE_S = 0.2     # cuánto se conserva de antes del arranque (la forma inicial de la mano)
SHOW_S = 3.0    # cuánto tiempo se muestra el resultado
COOLDOWN_S = 1.2  # tras un resultado se ignoran trazos nuevos este tiempo: bajar la mano no
                  # debe pisar un "correcto" con "falta el movimiento"
MIN_PROB = 0.25  # probabilidad mínima para dar el trazo por bueno, además de ser la clase más
                 # probable (con 7 clases, incluida "quieta", la masa se reparte: 0.25 ya es claro)
MIN_PATH = 0.8    # recorrido mínimo de la muñeca (tamaños de mano) para que cuente como trazo:
                  # en el dataset toda letra recorre >= 1.1 (percentil 5); una mano quieta, ~0.2
MIN_EXTENT = 0.3  # y extensión mínima (diagonal del rectángulo que abarca): toda letra >= 0.41;
                  # el ruido de los landmarks recorre mucho pero casi no se extiende
SMOOTH = 0.6      # suavizado de la posición de la muñeca (0-1; 1 = sin suavizar): quita el
                  # temblor de la detección, que si no dispara la ventana con la mano quieta
STILL = "quieta"  # clase del modelo para "forma correcta pero sin movimiento"


class GestureWindow:
    """Recibe fotograma a fotograma (t, landmarks, ángulos, muñeca) y devuelve el segmento
    del trazo cuando la mano vuelve a quedar quieta."""

    def __init__(self, stop_s=STOP_S, max_s=MAX_S):
        self.stop_s, self.max_s = stop_s, max_s
        self.reset()

    def configure(self, word):
        """Tiempos de una palabra/frase (pausas internas, más largas) o de una letra."""
        self.stop_s, self.max_s = (WORD_STOP_S, WORD_MAX_S) if word else (STOP_S, MAX_S)

    def reset(self):
        self.buf = deque()      # (t, feats, angles, wrist)
        self.active = False
        self.start_t = None
        self._still_since = None
        self._fast = 0
        self._speed = 0.0
        self._pos = None  # muñeca suavizada (x, y)

    @property
    def elapsed(self):
        return (self.buf[-1][0] - self.start_t) if self.active and self.buf else 0.0

    def feed(self, t, feats, angles, wrist, face=None, imu=None):
        """wrist: (x, y, tamaño) en coordenadas de imagen (0-1). face: (cx, cy, ancho, alto)
        del rostro en la imagen o None (para las palabras: ubicación respecto al cuerpo).
        imu: (roll, pitch) del guante en grados o None (sin guante / sin lectura).
        Devuelve el segmento terminado ({feats, angles, wrist, face, imu, ok, duration}) o None."""
        if self._pos is None:
            self._pos = (wrist[0], wrist[1])
        else:
            self._pos = (SMOOTH * wrist[0] + (1 - SMOOTH) * self._pos[0],
                         SMOOTH * wrist[1] + (1 - SMOOTH) * self._pos[1])
        wrist = (self._pos[0], self._pos[1], wrist[2])
        if self.buf:
            t0, w0 = self.buf[-1][0], self.buf[-1][3]
            dt = max(t - t0, 1e-3)
            size = max(wrist[2], 1e-3)
            v = math.hypot(wrist[0] - w0[0], wrist[1] - w0[1]) / size / dt
            self._speed = 0.5 * self._speed + 0.5 * v  # suavizado
        self.buf.append((t, feats, angles, wrist, face, imu))
        while self.buf and t - self.buf[0][0] > self.max_s + PRE_S + 0.5:
            self.buf.popleft()

        if not self.active:
            self._fast = self._fast + 1 if self._speed > START_V else 0
            if self._fast >= 2:
                self.active, self.start_t, self._still_since = True, t - PRE_S, None
            return None

        if self._speed < STOP_V:
            self._still_since = self._still_since or t
            if t - self._still_since >= self.stop_s:
                return self._finish(self._still_since)
        else:
            self._still_since = None
        if t - self.start_t > self.max_s:
            return self._finish(t)
        return None

    def finish(self, now):
        """Cierra el trazo ahora (p. ej. la mano salió de cuadro). Segmento o None."""
        return self._finish(now) if self.active else None

    def _finish(self, end_t):
        rows = [r for r in self.buf if self.start_t <= r[0] <= end_t]
        self.active, self._fast, self._still_since = False, 0, None
        if len(rows) < 6 or rows[-1][0] - rows[0][0] < MIN_S:
            return None
        nan4, nan2 = (math.nan,) * 4, (math.nan,) * 2
        return {"feats": np.array([r[1] for r in rows], dtype=np.float32),
                "angles": np.array([r[2] for r in rows], dtype=np.float32),
                "wrist": np.array([r[3] for r in rows], dtype=np.float32),
                "face": np.array([r[4] if r[4] is not None else nan4 for r in rows], dtype=np.float32),
                "imu": np.array([r[5] if r[5] is not None else nan2 for r in rows], dtype=np.float32),
                "ok": np.ones(len(rows), dtype=np.int8),
                "duration": round(rows[-1][0] - rows[0][0], 2)}


def wrist_travel(segment):
    """Cuánto recorrió la muñeca en el segmento, en tamaños de mano."""
    w = segment["wrist"]
    size = max(float(np.median(w[:, 2])), 1e-3)
    return float(np.linalg.norm(np.diff(w[:, :2], axis=0), axis=1).sum() / size)


def wrist_extent(segment):
    """Cuánto abarca el trazo (diagonal del rectángulo que lo contiene), en tamaños de mano."""
    w = segment["wrist"]
    size = max(float(np.median(w[:, 2])), 1e-3)
    return float(np.linalg.norm(w[:, :2].max(axis=0) - w[:, :2].min(axis=0)) / size)


def classify(bundle, segment):
    """bundle: lo que guardó train_dynamic.py. Devuelve (letra más probable, probabilidad);
    (STILL, 1.0) si la mano casi no se movió: la forma sola no es la seña."""
    if wrist_travel(segment) < MIN_PATH or wrist_extent(segment) < MIN_EXTENT:
        return STILL, 1.0
    if bundle.get("kind") == "words":  # señas de palabras: además, ubicación respecto al rostro
        from train_words import word_features
        # y, si el modelo se entrenó con guante ("imu" en el paquete), la inclinación de la muñeca
        x = word_features(segment["feats"], segment["angles"], segment["wrist"],
                          segment.get("face"), segment["ok"], imu=segment.get("imu"),
                          use_imu=bool(bundle.get("imu")))
    else:
        from train_dynamic import sequence_features
        x = sequence_features(segment["feats"], segment["angles"], segment["wrist"], segment["ok"])
    probs = bundle["model"].predict_proba([x])[0]
    i = int(np.argmax(probs))
    return str(bundle["model"].classes_[i]), float(probs[i])
