"""Importa el dataset público MSL de letras estáticas al formato de LSM Coach.

Fuente: Morfín, R. "Mexican Sign Language Alphabet (static signs only)". Zenodo.
DOI 10.5281/zenodo.10067508. Licencia CC BY 4.0. 21 letras, 20 participantes,
3 grupos de variación de pose (carpetas lsm-abc-A, -B, -C; NO son las letras A, B, C).

Por cada imagen corre MediaPipe Hands y guarda los 5 ángulos + 63 landmarks en
samples_public/<SEÑA>.csv, con el mismo formato que samples/ más una columna "grupo"
(A, B o C: el grupo C tiene más rotación de la mano). La persona queda como "msl-S<n>". Los participantes que el dataset reserva para prueba (carpeta test) no
se mezclan con los de entrenamiento, así se puede medir con gente que no se vio.

La cámara de la app está en espejo y MediaPipe llama "Right" a la mano derecha. Si
una imagen del dataset sale como "Left", se refleja para que la geometría coincida.

Uso:
    python import_msl.py --root ~/Downloads/datos_ent/MSL-ABC
    python import_msl.py --root ... --signs A B C L Y --por-persona 30 --nube
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import mediapipe as mp

from signs import NIVEL_1
from tolerance_calculator import (FINGER_JOINTS, IMU_COLUMNS, csv_header,
                                  landmarks_to_angles, landmarks_to_features)

PUBLIC_DIR = "samples_public"
SOURCE = "zenodo-msl-abc"
GROUPS = ("lsm-abc-A", "lsm-abc-B", "lsm-abc-C")
PAD = 0.25  # margen alrededor del recorte: el detector falla si la mano llena la imagen
FILE_RE = re.compile(r"^(S\d+)-[A-Z]+-\d+-(\d+)\.jpg$", re.IGNORECASE)


def pick_files(folder, per_person):
    """{persona: [rutas]} con `per_person` fotogramas repartidos a lo largo de su video."""
    by_person = defaultdict(list)
    for name in os.listdir(folder):
        m = FILE_RE.match(name)
        if m:
            by_person[m.group(1)].append((int(m.group(2)), name))
    picked = {}
    for person, frames in by_person.items():
        frames.sort()
        step = max(1, len(frames) / per_person)
        idx = sorted({int(i * step) for i in range(min(per_person, len(frames)))})
        picked[person] = [os.path.join(folder, frames[i][1]) for i in idx]
    return picked


def detect(hands, image):
    """(landmarks, etiqueta de mano) o (None, None). Añade margen antes de detectar."""
    h, w = image.shape[:2]
    p = int(PAD * max(h, w))
    padded = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_REPLICATE)
    res = hands.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None, None
    return res.multi_hand_landmarks[0].landmark, res.multi_handedness[0].classification[0].label


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="carpeta MSL-ABC ya descomprimida")
    ap.add_argument("--signs", nargs="+", default=NIVEL_1)
    ap.add_argument("--por-persona", type=int, default=30,
                    help="fotogramas por persona, letra y grupo (default 30)")
    ap.add_argument("--nube", action="store_true",
                    help="además guarda en lsm_coach.db para sincronizar con Supabase")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(os.path.join(root, GROUPS[0])):
        sys.exit(f"No encuentro {GROUPS[0]} dentro de {root}")
    store = None
    if args.nube:
        from store import Store
        store = Store()

    os.makedirs(PUBLIC_DIR, exist_ok=True)
    header = csv_header() + ["grupo"]
    stats = Counter()
    labels = Counter()
    t0 = time.time()
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=0.5) as hands:
        for sign in args.signs:
            path = os.path.join(PUBLIC_DIR, f"{sign}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for group in GROUPS:
                    for split in ("train", "test"):
                        folder = os.path.join(root, group, split, sign)
                        if not os.path.isdir(folder):
                            continue
                        for person, files in sorted(pick_files(folder, args.por_persona).items()):
                            for file in files:
                                image = cv2.imread(file)
                                if image is None:
                                    stats["ilegible"] += 1
                                    continue
                                lm, label = detect(hands, image)
                                if lm is not None and label == "Left":
                                    lm, label2 = detect(hands, cv2.flip(image, 1))
                                    labels["reflejada"] += 1
                                    if label2 != "Right":
                                        lm = None  # ambigua: mejor no usarla
                                elif lm is not None:
                                    labels["tal cual"] += 1
                                if lm is None:
                                    stats[f"{sign} sin mano"] += 1
                                    continue
                                angles = landmarks_to_angles(lm)
                                feats = landmarks_to_features(lm)
                                who = f"msl-{person}"
                                writer.writerow([datetime.now().isoformat(), who, sign]
                                                + [angles[k] for k in FINGER_JOINTS] + feats
                                                + [""] * len(IMU_COLUMNS) + [group[-1]])
                                if store:
                                    store.add_sample(who, sign, angles, feats, None,
                                                     source=SOURCE, camera=f"{group}/{split}")
                                stats[f"{sign} ok"] += 1
            ok, miss = stats[f"{sign} ok"], stats[f"{sign} sin mano"]
            print(f"{sign}: {ok} muestras  ({miss} sin mano detectada, "
                  f"{100 * ok / max(ok + miss, 1):.0f}% detectadas)  -> {path}")

    print(f"\nOrientación de las imágenes: {dict(labels)}")
    print(f"Tiempo: {time.time() - t0:.0f} s")
    if store:
        print(f"En lsm_coach.db para subir a la nube: {store.pending()} filas pendientes "
              "(se suben con: python store.py sync)")


if __name__ == "__main__":
    main()
