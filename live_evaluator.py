"""Evaluador en vivo de LSM Coach.

Fusiona cámara (dedos) y muñequera (orientación) para decidir si la seña es correcta
y decirle al aprendiz QUÉ parámetro corregir. Lee tolerances.json.

Uso:
    python live_evaluator.py                  # seña objetivo A, Nano real por BLE
    python live_evaluator.py --sign L         # otra seña objetivo
    python live_evaluator.py --voz            # ESPACIO = te dice en voz alta qué tienes mal
    python live_evaluator.py --imu none       # solo cámara (pruebas)
    python live_evaluator.py --auto           # detecta la letra sola (necesita model.joblib)

Teclas: 1-5 cambian la seña objetivo (A B C L Y), Q sale.
Con --imu mock (solo pruebas, NO en la demo): a/d roll, w/s pitch, j/l yaw, r reset.
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

from evaluation import ORIENT, evaluate
from imu_source import MockIMU, BLEIMU
from signs import NIVEL_1, describe
from tolerance_calculator import (landmarks_to_angles, landmarks_to_features,
                                  FINGER_JOINTS, OUTPUT_FILE)
from ui import TextLayer

SMOOTH_FRAMES = 5
PERSIST_S = 0.6   # el error debe durar esto para avisar (evita avisos por parpadeos)
REPEAT_S = 6.0    # si sigues con el MISMO error, se repite hasta pasado este tiempo
MIN_GAP_S = 1.5   # pausa mínima entre dos avisos cualquiera
VIBRATE_MS = 250
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (80, 80, 255), (0, 200, 255), (255, 255, 255)


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


def load_all_tolerances():
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Aviso: no existe {OUTPUT_FILE}. Corre: python tolerance_calculator.py analyze --sign A")
        return {}


def load_model(model_path):
    """Modo automático: carga el clasificador."""
    try:
        import joblib
        return joblib.load(model_path)
    except FileNotFoundError:
        sys.exit(f"No existe {model_path}. Entrena primero: python train_classifier.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", default=NIVEL_1[0], help="seña objetivo (default A)")
    ap.add_argument("--auto", action="store_true",
                    help="detecta la letra sola con model.joblib en vez de una seña objetivo")
    ap.add_argument("--modelo", default="model.joblib")
    ap.add_argument("--umbral", type=float, default=0.6,
                    help="confianza mínima para dar por buena la letra detectada")
    ap.add_argument("--imu", choices=["mock", "ble", "none"], default="ble",
                    help="ble = Nano real (por defecto); mock solo para pruebas sin hardware")
    ap.add_argument("--voz", action="store_true", help="ESPACIO: dice en voz alta qué tienes mal")
    args = ap.parse_args()

    all_tol = load_all_tolerances()
    auto = args.auto
    target = args.sign
    if auto:
        model = load_model(args.modelo)
        detect_hist = deque(maxlen=7)  # voto de los últimos fotogramas: evita parpadeos
    imu = {"mock": MockIMU, "ble": BLEIMU}.get(args.imu, lambda: None)()
    if imu:
        imu.start()
    speaker = Speaker() if args.voz else None
    last_alert_t, last_alert_key = 0.0, None
    cur_key, cur_since = None, 0.0

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
            layer = TextLayer()

            # Lectura de la muñequera (None si no hay conexión: nunca ceros falsos)
            linked = imu is not None and getattr(imu, "connected", True)
            imu_vals = dict(imu.latest) if linked else None

            cur_sign = None if auto else target
            tol, evaluation, y = None, None, 8
            if auto:
                if not res.multi_hand_landmarks:
                    detect_hist.clear()
            else:
                layer.add(f"Seña objetivo: {target}", (10, y), WHITE, 26)
                y += 32
                layer.add(describe(target), (10, y), YELLOW, 17)
                y += 24

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
                    txt = (f"Detecté: {top} ({det_conf * 100:.0f}%)" if cur_sign
                           else "Detectando...")
                    layer.add(txt, (10, y), GREEN if cur_sign else YELLOW, 24)
                    y += 30

                tol = all_tol.get(cur_sign) if cur_sign else None
                if tol:
                    # FUSIÓN: dedos (cámara) + orientación de la muñeca (IMU)
                    evaluation = evaluate(angles, imu_vals, tol, require_imu=imu is not None)
                    for finger, (ok_, msg) in evaluation.fingers.items():
                        layer.add(msg, (10, y), GREEN if ok_ else RED, 17)
                        y += 22
                    for issue in evaluation.issues:
                        if issue.parameter == ORIENT:
                            layer.add(f"Orientación: {issue.action}", (10, y), RED, 17)
                            y += 22
                        elif issue.where in FINGER_JOINTS:  # marca la punta del dedo que falla
                            tip = lm.landmark[FINGER_JOINTS[issue.where][2]]
                            cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 14, RED, 3)
                    if evaluation.issues:
                        layer.add("Corrige: " + ", ".join(evaluation.failed_parameters),
                                  (10, y + 4), RED, 20)
            else:
                layer.add("No se detecta la mano", (10, y), YELLOW, 20)

            if imu:
                if linked:
                    v = imu_vals
                    layer.add(f"Muñequera: roll={v['roll']:.0f}  pitch={v['pitch']:.0f}  "
                              f"yaw={v['yaw']:.0f}", (10, h - 62), YELLOW, 16)
                else:  # sin conexión no se evalúa la muñeca con ceros falsos
                    layer.add("Muñequera: buscando LSM-Wrist...", (10, h - 62), RED, 16)

            # Vibración automática: solo si el error persiste, sin repetir de más
            problems = [i.action for i in evaluation.issues] if evaluation else []
            now = time.time()
            key = problems[0] if problems else None
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
            elif evaluation is None:
                label, color = f"Seña '{cur_sign}': muestra tu mano", YELLOW
            else:
                label = f"Seña '{cur_sign}': " + ("CORRECTA" if evaluation.ok else "corrige")
                color = GREEN if evaluation.ok else RED
            layer.add(label, (10, h - 34), color, 26)
            hints = ("1-5: cambiar seña" + ("   ESPACIO: ¿qué tengo mal?" if speaker else "")
                     if not auto else "")
            if hints:
                layer.add(hints, (w - 270, h - 26), YELLOW, 14)
            cv2.imshow("LSM Coach - evaluador", layer.draw(frame))

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if not auto and ord("1") <= k <= ord(str(len(NIVEL_1))):
                target = NIVEL_1[k - ord("1")]
                cur_key, last_alert_key = None, None
            if k == ord(" ") and speaker:
                if not res.multi_hand_landmarks:
                    text = "No veo tu mano"
                elif problems:
                    text = ". ".join(list(dict.fromkeys(problems))[:3])
                else:
                    text = "Todo bien"
                speaker.say(text)
            if isinstance(imu, MockIMU) and k != 255:
                imu.handle_key(chr(k))

    cap.release()
    cv2.destroyAllWindows()
    if imu:
        imu.stop()
    if speaker:
        speaker.close()


if __name__ == "__main__":
    main()
