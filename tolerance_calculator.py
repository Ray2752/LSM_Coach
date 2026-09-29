"""
Calculadora de tolerancias para LSM Coach
==========================================

Idea: en vez de que alguien "invente" el rango correcto de cada seña,
lo derivamos grabando varias repeticiones de la misma seña (hechas por
distintas personas del equipo, imitando un video de referencia oficial)
y calculando el rango real (promedio +/- margen) de los ángulos de los
dedos que resultan.

NOVEDAD: cada muestra ahora guarda, además de los 5 ángulos, los 21
landmarks de la mano normalizados (63 números). Con eso mismo archivo sirve
para (1) calcular tolerancias y (2) entrenar después un clasificador que
reconozca la seña solo. Si ya tenías CSVs viejos (solo ángulos), se migran
automáticamente al formato nuevo la primera vez que grabes sobre ellos
(se deja una copia .bak); las muestras viejas siguen sirviendo para
`analyze`, pero no para entrenar el clasificador porque no tienen landmarks.

Uso:
    1. Instalar dependencias:
       pip install mediapipe opencv-python numpy

    2. Grabar muestras de una seña (repite varias veces, con distintas
       personas si se puede):
       python tolerance_calculator.py record --sign hola --person jorge

       - Se abre la cámara. Mantén la seña fija y presiona ESPACIO para
         capturar una muestra. Presiona Q para salir.
       - Repite el comando con --person distinto para cada integrante
         que grabe muestras de la misma seña.
       - Todos deben usar la MISMA mano al grabar (la que van a usar en la
         demo), para que las muestras sean comparables.

    3. Cuando ya tengan varias muestras (idealmente 15-30 por seña,
       entre varias personas), calcula el rango de tolerancia:
       python tolerance_calculator.py analyze --sign hola

       - Esto imprime el rango sugerido por dedo y lo guarda en
         tolerances.json, listo para que el motor de reglas lo use.

Los ángulos que se calculan son de flexión de cada dedo (MCP-PIP-TIP),
que es lo que la cámara puede medir de forma confiable. La orientación
de la muñeca (IMU) se calibra aparte, con el mismo principio: grabar
varias repeticiones "correctas" y sacar el rango.
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
from datetime import datetime

DATA_DIR = "samples"
OUTPUT_FILE = "tolerances.json"

# Índices de landmarks de MediaPipe Hands para cada dedo:
# (base del dedo / MCP, articulación media / PIP, punta / TIP)
FINGER_JOINTS = {
    "pulgar":  (1, 2, 4),
    "indice":  (5, 6, 8),
    "medio":   (9, 10, 12),
    "anular":  (13, 14, 16),
    "menique": (17, 18, 20),
}

# Columnas de los 21 landmarks (x, y, z de cada punto, en ese orden)
LANDMARK_COLUMNS = [f"lm{i}_{axis}" for i in range(21) for axis in "xyz"]


def angle_between(a, b, c):
    """Ángulo en grados en el punto b, formado por los segmentos a-b y b-c."""
    import numpy as np
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def landmarks_to_angles(landmarks):
    """landmarks: lista de 21 puntos (x, y, z) de MediaPipe Hands.
    Devuelve un dict {dedo: angulo_grados}."""
    angles = {}
    for finger, (i1, i2, i3) in FINGER_JOINTS.items():
        p1 = (landmarks[i1].x, landmarks[i1].y, landmarks[i1].z)
        p2 = (landmarks[i2].x, landmarks[i2].y, landmarks[i2].z)
        p3 = (landmarks[i3].x, landmarks[i3].y, landmarks[i3].z)
        angles[finger] = angle_between(p1, p2, p3)
    return angles


def landmarks_to_features(landmarks):
    """Devuelve 63 números: los 21 landmarks relativos a la muñeca (punto 0) y
    divididos entre la distancia muñeca -> base del dedo medio (punto 9).
    Así no dependen de dónde está la mano en la imagen ni de qué tan cerca
    está de la cámara, pero sí conservan la forma y la orientación de la mano."""
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    wx, wy, wz = pts[0]
    scale = math.dist(pts[0], pts[9]) or 1e-6
    feats = []
    for x, y, z in pts:
        feats += [(x - wx) / scale, (y - wy) / scale, (z - wz) / scale]
    return [round(v, 5) for v in feats]


def csv_header():
    return ["timestamp", "person", "sign"] + list(FINGER_JOINTS.keys()) + LANDMARK_COLUMNS


def migrate_csv_if_needed(csv_path, header):
    """Si el CSV existe con el formato viejo (solo ángulos), lo pasa al formato
    nuevo dejando vacías las columnas de landmarks. Guarda copia .bak."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == header:
        return
    old_header = rows[0]
    shutil.copy(csv_path, csv_path + ".bak")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows[1:]:
            d = dict(zip(old_header, r))
            writer.writerow([d.get(c, "") for c in header])
    print(f"Se migró {csv_path} al formato nuevo (copia en {csv_path}.bak). "
          f"Las {len(rows) - 1} muestras viejas no tienen landmarks.")


def cmd_record(args):
    import cv2
    import mediapipe as mp

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{args.sign}.csv")
    header = csv_header()
    migrate_csv_if_needed(csv_path, header)
    is_new = not os.path.exists(csv_path)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        sys.exit(1)

    print(f"Grabando muestras de '{args.sign}' para '{args.person}'.")
    print("Mantén la seña fija y presiona ESPACIO para capturar. Q para salir.")

    count = 0
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands, \
         open(csv_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            angles = None
            features = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                angles = landmarks_to_angles(hand_landmarks.landmark)
                features = landmarks_to_features(hand_landmarks.landmark)
                text_y = 30
                for finger, ang in angles.items():
                    cv2.putText(frame, f"{finger}: {ang:.1f}", (10, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    text_y += 25

            cv2.putText(frame, f"Muestras capturadas: {count}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Grabar seña - ESPACIO para capturar, Q para salir", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and angles is not None:
                row = [datetime.now().isoformat(), args.person, args.sign] + \
                      [angles[fn] for fn in FINGER_JOINTS.keys()] + features
                writer.writerow(row)
                f.flush()
                count += 1
                print(f"  Muestra {count} capturada: {angles}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. {count} muestras guardadas en {csv_path}")


def cmd_analyze(args):
    csv_path = os.path.join(DATA_DIR, f"{args.sign}.csv")
    if not os.path.exists(csv_path):
        print(f"No hay muestras para '{args.sign}'. Graba primero con: record --sign {args.sign}")
        sys.exit(1)

    per_finger = {f: [] for f in FINGER_JOINTS.keys()}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            n += 1
            for finger in FINGER_JOINTS.keys():
                per_finger[finger].append(float(row[finger]))

    if n < 5:
        print(f"Aviso: solo hay {n} muestras. Idealmente graben al menos 15-30 "
              "(varias personas repitiendo la seña) antes de fijar el rango.")

    margin = args.margin
    result = {}
    print(f"\nResultados para '{args.sign}' ({n} muestras):\n")
    for finger, values in per_finger.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        low = mean - margin * stdev if stdev > 0 else mean - 10
        high = mean + margin * stdev if stdev > 0 else mean + 10
        low, high = max(low, 0.0), min(high, 180.0)  # un ángulo real está en [0°, 180°]
        result[finger] = {"min": round(low, 1), "max": round(high, 1),
                           "promedio": round(mean, 1)}
        print(f"  {finger:10s}  promedio={mean:6.1f}°  rango sugerido=[{low:.1f}°, {high:.1f}°]")

    # Guardar/actualizar tolerances.json
    all_tolerances = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            all_tolerances = json.load(f)
    all_tolerances[args.sign] = result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tolerances, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en {OUTPUT_FILE}. El motor de reglas puede leer este archivo directo.")


def main():
    parser = argparse.ArgumentParser(description="Calculadora de tolerancias LSM Coach")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Grabar muestras de una seña con la cámara")
    p_record.add_argument("--sign", required=True, help="Nombre de la seña, ej: hola")
    p_record.add_argument("--person", required=True, help="Quién está grabando la muestra")
    p_record.set_defaults(func=cmd_record)

    p_analyze = sub.add_parser("analyze", help="Calcular rango de tolerancia de una seña")
    p_analyze.add_argument("--sign", required=True, help="Nombre de la seña a analizar")
    p_analyze.add_argument("--margin", type=float, default=2.0,
                            help="Cuántas desviaciones estándar de margen (default 2.0)")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()