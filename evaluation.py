"""Reglas de evaluación de LSM Coach: decide si una seña es correcta y qué corregir.

Fusiona dos fuentes:
  - cámara  -> ángulos de flexión de los 5 dedos (parámetro "Configuración")
  - muñequera (IMU) -> roll/pitch de la muñeca (parámetro "Orientación")
La seña solo es correcta si ambas fuentes están dentro del rango calibrado.

No depende de cámara ni de hardware, así que se puede probar y medir sin ellos.
"""
import math
from dataclasses import dataclass, field

from signs import SIGNS

CONFIG = "Configuración"
ORIENT = "Orientación"

ERROR_SUFFIX = "_mal"   # clase del clasificador con los errores típicos de una seña ("B_mal")
# si la clase de error de la seña llega a esto, se considera mal hecha. Con 0.4 (antes 0.5)
# las letras del Nivel 1 quedan equilibradas en personas que el modelo no vio: acepta ~83 %
# de las correctas y rechaza ~80 % de los errores (evaluate_accuracy.py --por-persona)
ERROR_MAX_PROB = 0.4

# Letras con la misma forma de mano que solo cambian de orientación (la G es una L
# horizontal; la H, una U horizontal). El clasificador ve la mano ya enderezada
# (rotate_upright), así que no puede separarlas: eso lo decide la muñequera.
SAME_SHAPE = {"L": {"G"}, "G": {"L"}, "U": {"H"}, "H": {"U"}}

# Separación máxima (grados) entre índice y meñique: "dedos separados" es el error típico
# de la B. En las B correctas de los expertos llega a ~7°; en sus errores, la mediana es ~14°.
MAX_SPREAD = {"B": 10.0}

FINGER_LABEL = {"pulgar": "pulgar", "indice": "índice", "medio": "medio",
                "anular": "anular", "menique": "meñique"}

# Instrucción cuando el valor queda (por debajo del mínimo, por encima del máximo).
# VERIFICAR al montar la muñequera: el sentido de roll/pitch depende de cómo quede
# puesta la Nano; si está al revés, se cambia AXIS_MAP en el firmware o se invierte aquí.
ORIENT_HINTS = {
    "roll": ("gira la muñeca hacia la izquierda", "gira la muñeca hacia la derecha"),
    "pitch": ("levanta la mano", "baja la mano"),
}


@dataclass
class Issue:
    parameter: str  # CONFIG u ORIENT: qué parámetro de la seña falló
    where: str      # dedo (pulgar, indice, ...) o eje (roll, pitch)
    action: str     # instrucción para el aprendiz, p. ej. "Extiende el índice"


@dataclass
class Result:
    fingers: dict = field(default_factory=dict)  # {dedo: (ok, texto para pantalla)}
    issues: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.issues

    @property
    def failed_parameters(self):
        return list(dict.fromkeys(i.parameter for i in self.issues))


def _finger_issues(angles, tol, result):
    for finger, ang in angles.items():
        lo, hi = tol[finger]["min"], tol[finger]["max"]
        name = FINGER_LABEL[finger]
        if ang < lo:
            result.fingers[finger] = (False, f"{name}: muy cerrado")
            result.issues.append(Issue(CONFIG, finger, f"Extiende el {name}"))
        elif ang > hi:
            result.fingers[finger] = (False, f"{name}: muy extendido")
            result.issues.append(Issue(CONFIG, finger, f"Flexiona el {name}"))
        else:
            result.fingers[finger] = (True, f"{name}: OK")


def _orientation_issues(imu, orient, require_imu, result):
    if imu is None:
        if require_imu:
            result.issues.append(Issue(ORIENT, "imu", "Conecta la muñequera"))
        return
    for axis, rng in orient.items():
        v = imu.get(axis)
        if v is None:
            continue
        below, above = ORIENT_HINTS.get(axis, (f"corrige {axis}", f"corrige {axis}"))
        if v < rng["min"]:
            result.issues.append(Issue(ORIENT, axis, below.capitalize()))
        elif v > rng["max"]:
            result.issues.append(Issue(ORIENT, axis, above.capitalize()))


def evaluate(angles, imu, tol, require_imu=True):
    """angles: {dedo: grados}. imu: {"roll","pitch","yaw"} o None si no hay lectura.
    tol: rangos de la seña (una entrada de tolerances.json).

    La orientación solo se exige si la seña tiene 'orientacion' calibrada. Con
    require_imu=True y sin lectura del dispositivo, la seña NO se da por correcta."""
    result = Result()
    _finger_issues(angles, tol, result)
    orient = tol.get("orientacion")
    if orient:
        _orientation_issues(imu, orient, require_imu, result)
    return result


def shape_issue(probs, target, max_error=ERROR_MAX_PROB):
    """probs: {clase: probabilidad} del clasificador (las clases "<SEÑA>_mal" son errores
    típicos). La forma está bien si `target` (o una letra de SAME_SHAPE) es la letra más
    probable y su clase de error no llega a `max_error`. No se exige una probabilidad mínima para la letra: con muchas
    letras parecidas (A, S, T, E) la probabilidad se reparte aunque la seña esté bien.
    Devuelve un Issue, o None si está bien o si el modelo no conoce la seña."""
    if target not in probs:
        return None
    if probs.get(target + ERROR_SUFFIX, 0.0) >= max_error:
        typical = SIGNS.get(target, {}).get("error_tipico", "").rstrip(".").lower()
        action = f"Revisa la {target}: {typical}" if typical else f"Revisa la forma de la {target}"
        return Issue(CONFIG, "forma", action)
    letters = {k: v for k, v in probs.items() if not k.endswith(ERROR_SUFFIX)}
    best = max(letters, key=letters.get)
    if best != target and best not in SAME_SHAPE.get(target, ()):
        return Issue(CONFIG, "forma", f"Ajusta la forma: se parece más a una {best}")
    return None


def finger_spread(features):
    """Ángulo (grados) entre el índice y el meñique, cada uno de la base a la punta, con
    los 63 números de landmarks_to_features. ~0° = dedos paralelos (juntos)."""
    def direction(base, tip):
        return [features[3 * tip + k] - features[3 * base + k] for k in range(3)]
    a, b = direction(5, 8), direction(17, 20)
    norm = math.hypot(*a) * math.hypot(*b) or 1e-9
    cos = sum(x * y for x, y in zip(a, b)) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def spread_issue(spread, target):
    """spread: finger_spread de la mano. Issue si la seña pide dedos juntos y están separados."""
    limit = MAX_SPREAD.get(target)
    if limit is not None and spread > limit:
        return Issue(CONFIG, "separacion", "Junta los dedos")
    return None
