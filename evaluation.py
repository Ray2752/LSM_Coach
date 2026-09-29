"""Reglas de evaluación de LSM Coach: decide si una seña es correcta y qué corregir.

Fusiona dos fuentes:
  - cámara  -> ángulos de flexión de los 5 dedos (parámetro "Configuración")
  - muñequera (IMU) -> roll/pitch de la muñeca (parámetro "Orientación")
La seña solo es correcta si ambas fuentes están dentro del rango calibrado.

No depende de cámara ni de hardware, así que se puede probar y medir sin ellos.
"""
from dataclasses import dataclass, field

CONFIG = "Configuración"
ORIENT = "Orientación"

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
