"""Convierte videos de señas de palabras (Nivel 3) en muestras para train_words.py.

Cada clip debe contener UNA ejecución de la seña, de frente, con el rostro visible si es
posible (la ubicación respecto al rostro es uno de los parámetros que se evalúan). Sirven
videos descargados (recortados a cada ejecución) o grabados por el equipo.

Uso:
    python extract_words.py --word HOLA --person experta clip1.mp4 clip2.mp4
    python extract_words.py --dir videos_palabras      # videos_palabras/HOLA/*.mp4, .../GRACIAS/*.mp4
    python extract_words.py --dir videos_palabras --auto   # cada video puede traer varias ejecuciones:
                                                           # se separan por el movimiento de la muñeca
"""
import argparse
import csv
import glob
import math
import os
import re
import sys
import unicodedata

import cv2
import mediapipe as mp
import numpy as np

from dynamic import GestureWindow
from signs import WORDS
from tolerance_calculator import landmarks_to_angles, landmarks_to_features, mirror_features

DATA_DIR = "samples_words"
FINGERS = ("pulgar", "indice", "medio", "anular", "menique")


def face_of(fd, rgb):
    res = fd.process(rgb)
    if not res.detections:
        return None
    b = res.detections[0].location_data.relative_bounding_box
    return (b.xmin + b.width / 2, b.ymin + b.height / 2, b.width, b.height)


def read_clip(hands, fd, path, mirror=True):
    """Fotograma a fotograma: (t, feats, angles, wrist, face, ok). mirror: los videos crudos
    no van en espejo como la cámara de la app; se reflejan para que coincidan."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, face, i = [], None, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if i % 3 == 0:
            face = face_of(fd, rgb) or face
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            label = res.multi_handedness[0].classification[0].label if res.multi_handedness else "Right"
            f = landmarks_to_features(lm)
            wx = lm[0].x
            if label == "Left":  # mano izquierda: se refleja para usar el mismo modelo
                f = mirror_features(f)
                wx = 1.0 - wx
            a = landmarks_to_angles(lm)
            size = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
            rows.append((i / fps, f, [a[k] for k in FINGERS], (wx, lm[0].y, size), face, 1))
        i += 1
    cap.release()
    return rows, fps


def segments_of(rows, auto):
    """Lista de segmentos (listas de filas). Sin --auto, todo el clip es una ejecución."""
    if not auto:
        return [rows] if rows else []
    win, out, t_last = GestureWindow(), [], None
    for r in rows:
        seg = win.feed(r[0], r[1], r[2], r[3], r[4])
        if seg is not None:
            t0, t1 = r[0] - seg["duration"], r[0]
            out.append([x for x in rows if t0 - 0.05 <= x[0] <= t1 + 0.05])
    if win.active:  # el video terminó a media seña
        seg = win.finish(rows[-1][0])
        if seg is not None:
            out.append([x for x in rows if rows[-1][0] - seg["duration"] - 0.05 <= x[0]])
    return out


def save_segment(rows, word, person, source, index_writer, n):
    nan4 = (math.nan,) * 4
    name = f"{person}-{source}-{n:03d}"
    os.makedirs(os.path.join(DATA_DIR, word), exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, word, f"{name}.npz"),
                        feats=np.array([r[1] for r in rows], np.float32),
                        angles=np.array([r[2] for r in rows], np.float32),
                        wrist=np.array([r[3] for r in rows], np.float32),
                        face=np.array([r[4] if r[4] is not None else nan4 for r in rows], np.float32),
                        ok=np.ones(len(rows), np.int8))
    index_writer.writerow({"file": name, "word": word, "person": person, "source": source,
                           "frames": len(rows), "detected": len(rows),
                           "face_frames": sum(r[4] is not None for r in rows)})


def open_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "index.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["file", "word", "person", "source", "frames", "detected", "face_frames"])
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", help="clips (uno por ejecución) de --word")
    ap.add_argument("--word", help="palabra de los clips: " + ", ".join(WORDS))
    ap.add_argument("--person", default="video", help="quién hace la seña (para medir por persona)")
    ap.add_argument("--dir", help="carpeta con una subcarpeta por palabra (HOLA/, GRACIAS/, ...)")
    ap.add_argument("--auto", action="store_true", help="separa varias ejecuciones por movimiento")
    ap.add_argument("--sin-espejo", action="store_true", help="el video ya está en espejo (grabado con la app)")
    args = ap.parse_args()

    jobs = []  # (word, person, path)
    norm = lambda s: unicodedata.normalize("NFC", s).upper().replace("_", " ")
    if args.dir:
        for word_dir in sorted(glob.glob(os.path.join(args.dir, "*"))):
            word = norm(os.path.basename(word_dir))
            if word not in WORDS:
                print(f"Se ignora {word_dir}: no es una palabra del reto ({', '.join(WORDS)})")
                continue
            for p in sorted(glob.glob(os.path.join(word_dir, "*.*"))):
                if re.search(r"\.(mp4|mov|m4v|avi|mkv|webm)$", p, re.I):
                    jobs.append((word, args.person, p))
    if args.videos:
        if not args.word or norm(args.word) not in WORDS:
            sys.exit("Indica --word con una palabra del reto: " + ", ".join(WORDS))
        jobs += [(norm(args.word), args.person, p) for p in args.videos]
    if not jobs:
        sys.exit("No hay videos. Ejemplo: python extract_words.py --word HOLA --person ana hola1.mp4")

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    f, writer = open_index()
    counts = {}
    try:
        for word, person, path in jobs:
            rows, fps = read_clip(hands, fd, path, mirror=not args.sin_espejo)
            hands.reset()
            segs = segments_of(rows, args.auto)
            for seg in segs:
                if len(seg) < 6:
                    continue
                counts[word] = counts.get(word, 0) + 1
                n = len(glob.glob(os.path.join(DATA_DIR, word, "*.npz")))
                save_segment(seg, word, person, os.path.splitext(os.path.basename(path))[0][:20], writer, n)
            faces = sum(r[4] is not None for r in rows)
            times = ", ".join(f"{s[0][0]:.1f}-{s[-1][0]:.1f}s" for s in segs if len(s) >= 6)
            print(f"{os.path.basename(path):32s} {word:10s} mano en {len(rows)} fotogramas, rostro en {faces}, "
                  f"{len(segs)} ejecución(es)" + (f" [{times}]" if args.auto and times else "")
                  + ("" if faces else "  <- sin rostro: ubicación aproximada"))
            f.flush()
    finally:
        f.close()
    print(f"\nGuardado en {DATA_DIR}/: {counts}. Entrena con: python train_words.py")


if __name__ == "__main__":
    main()
