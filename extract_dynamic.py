"""Extrae la trayectoria de la mano de cada video del dataset de señas con movimiento
(MSL dynamic signs, Zenodo 10.5281/zenodo.14689869: J, K, Ñ, Q, X, Z; 900x900, 30 fps, ~2 s).

Por cada video guarda, fotograma a fotograma:
  - feats  (T, 63): landmarks relativos a la muñeca y escalados (como las estáticas; la
                    mano izquierda se refleja para que la geometría sea de mano derecha)
  - angles (T, 5):  flexión de cada dedo
  - wrist  (T, 3):  posición de la muñeca en la imagen (x, y en 0-1) y tamaño de la mano
                    (distancia muñeca -> base del medio), para el movimiento en el espacio
  - ok     (T,):    1 si en ese fotograma se detectó la mano
en samples_dynamic/<train|test>/<nombre>.npz, más samples_dynamic/index.csv.

Uso:
    python extract_dynamic.py --root ~/Downloads/datos_ent/MSL-dynamic-signs
    python extract_dynamic.py --root ... --solo 20     # prueba rápida con 20 videos
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
import unicodedata

import cv2
import mediapipe as mp
import numpy as np

from tolerance_calculator import landmarks_to_angles, landmarks_to_features, mirror_features

OUT_DIR = "samples_dynamic"
# "S1-J-frontal-1.mp4" y también "S2-K-Frontal.1.mp4" (45 videos vienen así); los "Perfil" se saltan
NAME_RE = re.compile(r"^(S\d+)-(.+?)-frontal[-.](\d+)\.mp4$", re.IGNORECASE)


def process_video(hands, path):
    cap = cv2.VideoCapture(path)
    feats, angles, wrist, ok, handed = [], [], [], [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            label = res.multi_handedness[0].classification[0].label if res.multi_handedness else "Right"
            f = landmarks_to_features(lm)
            if label == "Left":
                f = mirror_features(f)
            a = landmarks_to_angles(lm)
            feats.append(f)
            angles.append([a[k] for k in ("pulgar", "indice", "medio", "anular", "menique")])
            scale = float(np.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y))
            wrist.append([lm[0].x, lm[0].y, scale])
            ok.append(1)
            handed.append(label)
        else:  # sin mano: se repite el último valor (o ceros) y se marca ok=0
            feats.append(feats[-1] if feats else [0.0] * 63)
            angles.append(angles[-1] if angles else [0.0] * 5)
            wrist.append(wrist[-1] if wrist else [0.0, 0.0, 0.0])
            ok.append(0)
    cap.release()
    return (np.array(feats, dtype=np.float32), np.array(angles, dtype=np.float32),
            np.array(wrist, dtype=np.float32), np.array(ok, dtype=np.int8),
            max(set(handed), key=handed.count) if handed else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="carpeta con train/ y test/")
    ap.add_argument("--solo", type=int, help="procesa solo N videos (prueba)")
    args = ap.parse_args()

    videos = []
    for split in ("train", "test"):
        for path in sorted(glob.glob(os.path.join(os.path.expanduser(args.root), split, "*.mp4"))):
            m = NAME_RE.match(os.path.basename(path))
            if m:  # la Ñ del nombre de archivo viene descompuesta (N + tilde) en macOS: se unifica
                sign = unicodedata.normalize("NFC", m.group(2))
                videos.append((split, path, m.group(1), sign, int(m.group(3))))
    if args.solo:
        videos = videos[:args.solo]
    if not videos:
        sys.exit("No encontré videos S#-LETRA-frontal-#.mp4 en train/ o test/")

    os.makedirs(OUT_DIR, exist_ok=True)
    index_path = os.path.join(OUT_DIR, "index.csv")
    done = set()
    if os.path.exists(index_path):
        with open(index_path, newline="", encoding="utf-8") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    new_index = not os.path.exists(index_path)

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    t0 = time.time()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "person", "sign", "take", "frames",
                                          "detected", "handed"])
        if new_index:
            w.writeheader()
        for i, (split, path, person, sign, take) in enumerate(videos, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in done:
                continue
            feats, angles, wrist, ok, handed = process_video(hands, path)
            os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)
            np.savez_compressed(os.path.join(OUT_DIR, split, f"{name}.npz"),
                                feats=feats, angles=angles, wrist=wrist, ok=ok)
            w.writerow({"file": name, "split": split, "person": person, "sign": sign, "take": take,
                        "frames": len(ok), "detected": int(ok.sum()), "handed": handed})
            f.flush()
            hands.reset()  # que un video no arrastre el rastreo del anterior
            if i % 20 == 0 or i == len(videos):
                el = time.time() - t0
                print(f"{i}/{len(videos)}  {el/60:.1f} min  (faltan ~{el/i*(len(videos)-i)/60:.1f} min)",
                      flush=True)
    print(f"Listo: {len(videos)} videos en {OUT_DIR}/ ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
