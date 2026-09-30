"""Mide cuánto tarda cada parte de la visión en esta máquina (para la UNO Q).

    python uno_q/bench.py            # con el navegador cerrado, para medir solo la visión
    python uno_q/bench.py --cam 2    # otra cámara

Imprime ms por fotograma de: captura + decodificación, MediaPipe Hands (modelo completo y
ligero), clasificador y codificación JPEG, y los fps que darían.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="2" if sys.platform.startswith("linux") else "0")
    ap.add_argument("--res", default="640x480")
    args = ap.parse_args()
    w, h = (int(v) for v in args.res.split("x"))
    import mediapipe as mp

    print(f"CPU: {os.cpu_count()} núcleos | OpenCV {cv2.__version__} | MediaPipe {mp.__version__}")
    dev = int(args.cam) if args.cam.isdigit() else args.cam
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(dev)
    if sys.platform.startswith("linux"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if not ok:
        sys.exit("No se pudo leer la cámara")
    for _ in range(10):
        cap.read()
    ms_cap = timed(lambda: cap.read(), 30)
    print(f"captura + decodificación ({frame.shape[1]}x{frame.shape[0]}): {ms_cap:.1f} ms")
    frames = [cap.read()[1] for _ in range(10)]
    cap.release()

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    for complexity in (1, 0):
        hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                                         model_complexity=complexity)
        hands.process(rgb[0])  # calentamiento
        i = [0]

        def step():
            hands.process(rgb[i[0] % len(rgb)])
            i[0] += 1
        ms = timed(step, 20)
        print(f"MediaPipe Hands modelo {'completo' if complexity else 'ligero  '} (complexity={complexity}): "
              f"{ms:.0f} ms  ->  {1000 / ms:.1f} fps" + ("  (sin mano en la imagen: solo detección)" if not hands.process(rgb[0]).multi_hand_landmarks else ""))
        hands.close()

    try:
        import joblib
        model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.joblib"))
        (model[-1] if hasattr(model, "steps") else model).n_jobs = 1
        x = np.random.default_rng(0).normal(size=(1, 63))
        print(f"clasificador de letras: {timed(lambda: model.predict_proba(x), 20):.1f} ms")
    except Exception as e:
        print("clasificador:", e)
    print(f"codificar JPEG q55: {timed(lambda: cv2.imencode('.jpg', frames[0], [cv2.IMWRITE_JPEG_QUALITY, 55]), 20):.1f} ms")
    print("\nSi 'modelo ligero' es mucho más rápido que 'completo', arranca con: run.sh --modelo-ligero")


if __name__ == "__main__":
    main()
