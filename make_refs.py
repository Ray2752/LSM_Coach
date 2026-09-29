"""Genera las imágenes de referencia de cada letra para la interfaz (web/ref/).

Letras estáticas: una foto del dataset. Letras con movimiento: una animación GIF
recortada alrededor de la mano. Fuentes (CC BY 4.0, se citan en la interfaz y el README):
  - Morfín, R. Mexican Sign Language Alphabet (static signs only). Zenodo, 10.5281/zenodo.10067508
  - Navarrete-López, J. A., López-Nava, I. H. Mexican Sign Language Alphabet (dynamic signs only).
    Zenodo, 10.5281/zenodo.14689869

Uso:
    python make_refs.py --root ~/Downloads/datos_ent
"""
import argparse
import os

import cv2
import mediapipe as mp
from PIL import Image

from signs import DYNAMIC, STATIC, ref_name

OUT_DIR = os.path.join("web", "ref")
SIZE = 240
PERSON = "S1"        # participante de las referencias
GIF_STEP = 3         # uno de cada 3 fotogramas (60 fps -> 20 fps)
PAD = 0.35           # margen alrededor de la mano en las animaciones


def static_ref(root, sign):
    folder = os.path.join(root, "MSL-ABC", "lsm-abc-A", "train", sign)
    files = sorted(f for f in os.listdir(folder) if f.startswith(f"{PERSON}-"))
    image = Image.open(os.path.join(folder, files[len(files) // 2])).convert("RGB")
    image.resize((SIZE, SIZE)).save(os.path.join(OUT_DIR, ref_name(sign, "jpg")), quality=85)


def hand_box(hands, frames):
    """Caja cuadrada (en píxeles) que contiene la mano en todos los fotogramas."""
    h, w = frames[0].shape[:2]
    xs, ys = [], []
    for f in frames:
        res = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for p in res.multi_hand_landmarks[0].landmark:
                xs.append(p.x * w)
                ys.append(p.y * h)
    if not xs:
        return 0, 0, min(h, w)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * PAD)
    side = int(min(max(side, 120), h, w))
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, side


def dynamic_ref(root, sign, hands):
    for split in ("train", "test"):
        path = os.path.join(root, "MSL-dynamic-signs", split, f"{PERSON}-{sign}-frontal-1.mp4")
        if os.path.exists(path):
            break
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % GIF_STEP == 0:
            frames.append(frame)
        i += 1
    x0, y0, side = hand_box(hands, frames)
    images = [Image.fromarray(cv2.cvtColor(f[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB))
              .resize((SIZE, SIZE)) for f in frames]
    images[0].save(os.path.join(OUT_DIR, ref_name(sign, "gif")), save_all=True,
                   append_images=images[1:], duration=int(1000 * GIF_STEP / 60), loop=0,
                   optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="carpeta con MSL-ABC y MSL-dynamic-signs")
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    os.makedirs(OUT_DIR, exist_ok=True)
    for sign in STATIC:
        static_ref(root, sign)
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for sign in DYNAMIC:
            dynamic_ref(root, sign, hands)
    print(f"Referencias en {OUT_DIR}/: {len(STATIC)} fotos y {len(DYNAMIC)} animaciones")


if __name__ == "__main__":
    main()
