"""Evaluador en vivo de LSM Coach.

Usa la MISMA función de ángulos que tolerance_calculator.py y lee tolerances.json.
Ponlo en la misma carpeta que tolerance_calculator.py e imu_source.py.

Uso:
    python live_evaluator.py                          # AUTOMÁTICO: detecta la letra solo (necesita model.joblib)
    python live_evaluator.py --sign A                 # tú eliges la letra; Nano real por BLE
    python live_evaluator.py --sign A --voz           # ESPACIO = te dice en voz alta qué tienes mal
    python live_evaluator.py --sign A --imu none      # solo cámara
    python live_evaluator.py --sign A --imu mock      # IMU simulado (solo pruebas, NO en la demo)

Teclas: Q salir. Con IMU simulado: a/d roll, w/s pitch, j/l yaw, r reset.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

from tolerance_calculator import (landmarks_to_angles, landmarks_to_features,
                                  FINGER_JOINTS, OUTPUT_FILE)
from imu_source import MockIMU, BLEIMU

SMOOTH_FRAMES = 5
PERSIST_S = 0.6   # el error debe durar esto para avisar (evita avisos por parpadeos)
REPEAT_S = 6.0    # si sigues con el MISMO error, se repite hasta pasado este tiempo
MIN_GAP_S = 1.5   # pausa mínima entre dos avisos cualquiera
VIBRATE_MS = 250
SPOKEN_NAMES = {"pulgar": "pulgar", "indice": "índice", "medio": "medio",
                "anular": "anular", "menique": "meñique"}
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 200, 255)


# En Windows se usa la voz del sistema (System.Speech) con PowerShell, un proceso
# NUEVO por cada frase: no guarda estado, así que suena siempre y no trae el bug
# de pyttsx3 (que solo hablaba la primera vez). Funciona sin internet.
_PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$vs = $s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo };"
    "$pick = $null;"
    "if ($env:LSM_HINT) { $pick = $vs | Where-Object { $_.Name -like ('*' + $env:LSM_HINT + '*') } | Select-Object -First 1 };"
    "if (-not $pick) { $pick = $vs | Where-Object { $_.Culture.Name -eq 'es-MX' } | Select-Object -First 1 };"
    "if (-not $pick) { $pick = $vs | Where-Object { $_.Culture.Name -like 'es-*' } | Select-Object -First 1 };"
    "if ($pick) { $s.SelectVoice($pick.Name) };"
    "$s.Speak($env:LSM_TEXT)"
)


class Speaker:
    """Voz sin internet. Mientras una frase se está diciendo, las nuevas
    peticiones se ignoran (mantener ESPACIO no acumula nada ni traba el video).

    Para forzar una voz de Windows, pon parte de su nombre en VOICE_HINT
    (ej. "Sabina"). Lista las voces instaladas con:
        powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | % { $_.VoiceInfo.Name + ' ' + $_.VoiceInfo.Culture.Name }"
    Linux (la Q): usa espeak-ng (sudo apt install espeak-ng). macOS: usa say.
    """
    VOICE_HINT = ""

    def __init__(self):
        self.proc = None

    @property
    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def say(self, text, still_valid=None):
        if self.busy:
            return
        text = text.replace("\n", " ")
        env = dict(os.environ, LSM_TEXT=text, LSM_HINT=self.VOICE_HINT)
        if sys.platform == "win32":
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_SCRIPT]
            flags = 0x08000000  # CREATE_NO_WINDOW
        elif sys.platform == "darwin":
            cmd, flags = ["say", text], 0
        elif shutil.which("espeak-ng"):
            cmd, flags = ["espeak-ng", "-v", "es", text], 0
        else:
            print("Voz: instala espeak-ng (sudo apt install espeak-ng)")
            return
        try:
            self.proc = subprocess.Popen(cmd, env=env, creationflags=flags)
        except Exception as e:
            print("Voz:", e)

    def close(self):
        if self.busy:
            self.proc.kill()


def load_tolerances(sign):
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No existe {OUTPUT_FILE}. Corre primero: "
                 f"python tolerance_calculator.py analyze --sign {sign}")
    if sign not in data:
        sys.exit(f"'{sign}' no está en {OUTPUT_FILE}. Disponibles: {list(data)}")
    return data[sign]


def load_auto(model_path):
    """Modo automático: carga el clasificador y todos los rangos calibrados."""
    try:
        import joblib
        model = joblib.load(model_path)
    except FileNotFoundError:
        sys.exit(f"No existe {model_path}. Entrena primero: python train_classifier.py")
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            all_tol = json.load(f)
    except FileNotFoundError:
        all_tol = {}
    return model, all_tol


def evaluate_fingers(angles, tol):
    """Devuelve {dedo: (ok, mensaje)}."""
    result = {}
    for finger, ang in angles.items():
        lo, hi = tol[finger]["min"], tol[finger]["max"]
        if ang < lo:
            result[finger] = (False, f"{finger}: muy cerrado, extiéndelo")
        elif ang > hi:
            result[finger] = (False, f"{finger}: muy extendido, flexiónalo")
        else:
            result[finger] = (True, f"{finger}: OK")
    return result


def evaluate_orientation(imu_vals, tol):
    """Opcional: solo si tolerances.json trae 'orientacion' para la seña.
    Formato: {"orientacion": {"roll": {"min": -20, "max": 20}, "pitch": {...}}}"""
    orient = tol.get("orientacion") if tol else None
    if not orient:
        return []
    problems = []
    for axis, rng in orient.items():
        v = imu_vals.get(axis)
        if v is None:
            continue
        if v < rng["min"]:
            problems.append(f"{axis}: {v:.0f}° (mín {rng['min']}°)")
        elif v > rng["max"]:
            problems.append(f"{axis}: {v:.0f}° (máx {rng['max']}°)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", help="letra a evaluar; si no la pones, la detecta sola")
    ap.add_argument("--modelo", default="model.joblib")
    ap.add_argument("--umbral", type=float, default=0.6,
                    help="confianza mínima para dar por buena la letra detectada")
    ap.add_argument("--imu", choices=["mock", "ble", "none"], default="ble",
                    help="ble = Nano real (por defecto); mock solo para pruebas sin hardware")
    ap.add_argument("--voz", action="store_true", help="ESPACIO: dice en voz alta qué tienes mal")
    args = ap.parse_args()

    auto = args.sign is None
    if auto:
        model, all_tol = load_auto(args.modelo)
        detect_hist = deque(maxlen=7)  # voto de los últimos fotogramas: evita parpadeos
        tol = None
    else:
        tol = load_tolerances(args.sign)
    imu = {"mock": MockIMU, "ble": BLEIMU}.get(args.imu, lambda: None)()
    if imu:
        imu.start()
    speaker = Speaker() if args.voz else None
    last_alert_t, last_alert_key = 0.0, None
    cur_key, cur_since = None, 0.0
    state = {"key": None}  # error actual (lo lee el hilo de voz)

    history = {f: deque(maxlen=SMOOTH_FRAMES) for f in FINGER_JOINTS}
    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la cámara.")

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # igual que al grabar
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            all_ok, y, problems = False, 30, []
            cur_sign, det_conf = args.sign, 0.0
            if auto:
                tol, cur_sign = None, None
                if not res.multi_hand_landmarks:
                    detect_hist.clear()
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                raw = landmarks_to_angles(lm.landmark)
                for f, a in raw.items():
                    history[f].append(a)
                angles = {f: sum(v) / len(v) for f, v in history.items()}

                if auto:
                    proba = model.predict_proba([landmarks_to_features(lm.landmark)])[0]
                    i = int(proba.argmax())
                    detect_hist.append((str(model.classes_[i]), float(proba[i])))
                    labels = [l for l, _ in detect_hist]
                    top = max(set(labels), key=labels.count)
                    det_conf = sum(c for l, c in detect_hist if l == top) / labels.count(top)
                    if det_conf >= args.umbral:
                        cur_sign = top
                        tol = all_tol.get(top)
                    txt = (f"Detecté: {top} ({det_conf * 100:.0f}%)" if cur_sign
                           else "Detectando...")
                    cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, GREEN if cur_sign else YELLOW, 2)
                    y += 32

                if tol:
                    fingers = evaluate_fingers(angles, tol)
                    all_ok = all(ok_ for ok_, _ in fingers.values())

                    for finger, (ok_, msg) in fingers.items():
                        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, GREEN if ok_ else RED, 2)
                        y += 25
                        if not ok_:  # marca la punta del dedo que falla
                            verb = "Extiende" if angles[finger] < tol[finger]["min"] else "Flexiona"
                            problems.append(f"{verb} el {SPOKEN_NAMES[finger]}")
                            tip = lm.landmark[FINGER_JOINTS[finger][2]]
                            cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 14, RED, 3)
            else:
                cv2.putText(frame, "No se detecta la mano", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            if imu:
                v = imu.latest
                linked = getattr(imu, "connected", True)  # MockIMU no tiene .connected
                if linked:
                    cv2.putText(frame,
                                f"IMU roll={v['roll']:.0f} pitch={v['pitch']:.0f} yaw={v['yaw']:.0f}",
                                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
                else:  # sin conexión no se evalúa la muñeca con ceros falsos
                    cv2.putText(frame, "IMU: buscando LSM-Wrist...", (10, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
                for p in (evaluate_orientation(v, tol) if linked else []):
                    all_ok = False
                    problems.append("Corrige la muñeca")
                    cv2.putText(frame, "Muñeca -> " + p, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                    y += 25

            # Vibración automática: solo si el error persiste, sin repetir de más
            now = time.time()
            key = problems[0] if problems else None
            state["key"] = key
            if key != cur_key:
                cur_key, cur_since = key, now
            if (key and now - cur_since >= PERSIST_S
                    and now - last_alert_t >= MIN_GAP_S
                    and (key != last_alert_key or now - last_alert_t >= REPEAT_S)):
                last_alert_t, last_alert_key = now, key
                if imu:
                    imu.vibrate(VIBRATE_MS)
            if key is None:
                last_alert_key = None  # ya está bien: el próximo error avisa de inmediato

            if cur_sign is None:
                label, color = "Haz una seña...", YELLOW
            elif tol is None:
                label, color = f"'{cur_sign}' (sin rangos calibrados)", YELLOW
            else:
                label = f"Seña '{cur_sign}': " + ("CORRECTA" if all_ok else "corrige")
                color = GREEN if all_ok else RED
            cv2.putText(frame, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if speaker:
                cv2.putText(frame, "ESPACIO: que tengo mal?", (w - 260, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
            cv2.imshow("LSM Coach - evaluador", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and speaker:
                if not res.multi_hand_landmarks:
                    text = "No veo tu mano"
                elif problems:
                    text = ". ".join(list(dict.fromkeys(problems))[:3])
                else:
                    text = "Todo bien"
                speaker.say(text, lambda: True)
            if isinstance(imu, MockIMU) and key != 255:
                imu.handle_key(chr(key))

    cap.release()
    cv2.destroyAllWindows()
    if imu:
        imu.stop()
    if speaker:
        speaker.close()


if __name__ == "__main__":
    main()