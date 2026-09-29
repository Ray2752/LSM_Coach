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
ERRORS_DIR = "samples_errors"  # ejecuciones INCORRECTAS a propósito (para medir aciertos)
OUTPUT_FILE = "tolerances.json"

# Lecturas de la muñequera al capturar (vacías si no había IMU conectada)
IMU_COLUMNS = ["imu_roll", "imu_pitch", "imu_yaw"]
MIN_IMU_SAMPLES = 5
ORIENT_AXES = ("roll", "pitch")  # el yaw no se usa: sin magnetómetro solo es relativo

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


def rotate_upright(X):
    """Gira cada mano en el plano de la imagen para que muñeca -> base del dedo medio
    apunte hacia arriba. X: filas de 63 números de landmarks_to_features.

    Así el clasificador reconoce la FORMA aunque la mano esté inclinada (con la mano
    girada 60°, sin esto la C bajaba de 98 % a 23 % de aciertos). La inclinación de
    la muñeca la evalúa la muñequera, no el clasificador."""
    import numpy as np
    P = np.asarray(X, dtype=float).reshape(-1, 21, 3).copy()
    ang = np.arctan2(P[:, 9, 0], -P[:, 9, 1])  # ángulo respecto a "arriba" (-y en la imagen)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x, y = P[:, :, 0].copy(), P[:, :, 1].copy()
    P[:, :, 0] = c * x + s * y
    P[:, :, 1] = -s * x + c * y
    return P.reshape(len(P), -1)


def csv_header():
    return (["timestamp", "person", "sign"] + list(FINGER_JOINTS.keys())
            + LANDMARK_COLUMNS + IMU_COLUMNS)


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


def append_sample(sign, person, angles, features, imu_vals, errores=False):
    """Agrega una muestra al CSV de la seña, con el mismo formato que `record`.
    Devuelve la ruta del CSV."""
    data_dir = ERRORS_DIR if errores else DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{sign}.csv")
    header = csv_header()
    migrate_csv_if_needed(csv_path, header)
    is_new = not os.path.exists(csv_path)
    imu_row = ([imu_vals["roll"], imu_vals["pitch"], imu_vals["yaw"]]
               if imu_vals else [""] * len(IMU_COLUMNS))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow([datetime.now().isoformat(), person, sign]
                        + [angles[fn] for fn in FINGER_JOINTS.keys()] + features + imu_row)
    return csv_path


def cmd_record(args):
    import cv2
    import mediapipe as mp
    from ui import TextLayer

    green, yellow, red = (0, 255, 0), (0, 200, 255), (0, 0, 255)
    data_dir = ERRORS_DIR if args.errores else DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{args.sign}.csv")
    header = csv_header()
    migrate_csv_if_needed(csv_path, header)
    is_new = not os.path.exists(csv_path)

    imu = None
    if args.imu == "ble":
        from imu_source import BLEIMU
        imu = BLEIMU()
        imu.start()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        sys.exit(1)

    kind = "ERRORES intencionales" if args.errores else "ejecuciones correctas"
    print(f"Grabando {kind} de '{args.sign}' para '{args.person}' en {csv_path}.")
    print("Mantén la seña fija y presiona ESPACIO para capturar. Q para salir.")
    if imu:
        print("Esperando la muñequera 'LSM-Wrist' (usa --imu none para grabar solo con cámara)...")

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
            layer = TextLayer()

            angles = None
            features = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                angles = landmarks_to_angles(hand_landmarks.landmark)
                features = landmarks_to_features(hand_landmarks.landmark)
                for i, (finger, ang) in enumerate(angles.items()):
                    layer.add(f"{finger}: {ang:.1f}", (10, 8 + 24 * i), green, 18)

            imu_vals = dict(imu.latest) if imu and imu.connected else None
            h = frame.shape[0]
            if imu:
                if imu_vals:
                    layer.add(f"IMU roll={imu_vals['roll']:.0f} pitch={imu_vals['pitch']:.0f} "
                              f"yaw={imu_vals['yaw']:.0f}", (10, h - 62), yellow, 18)
                else:
                    layer.add("IMU: buscando LSM-Wrist...", (10, h - 62), red, 18)
            layer.add(f"{kind}: {args.sign} — muestras capturadas: {count}",
                      (10, h - 32), yellow, 18)
            frame = layer.draw(frame)
            cv2.imshow("Grabar sena - ESPACIO para capturar, Q para salir", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and angles is not None:
                if imu and imu_vals is None:
                    print("  (sin captura: la muñequera aún no está conectada)")
                    continue
                imu_row = ([imu_vals["roll"], imu_vals["pitch"], imu_vals["yaw"]]
                           if imu_vals else [""] * len(IMU_COLUMNS))
                row = [datetime.now().isoformat(), args.person, args.sign] + \
                      [angles[fn] for fn in FINGER_JOINTS.keys()] + features + imu_row
                writer.writerow(row)
                f.flush()
                count += 1
                print(f"  Muestra {count} capturada: {angles}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if imu:
        imu.stop()
    print(f"\nListo. {count} muestras guardadas en {csv_path}")


def _range(values, margin, floor, lo_limit=None, hi_limit=None):
    """Rango = promedio +/- max(margen * desviación, holgura mínima)."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    half = max(margin * stdev, floor)
    low, high = mean - half, mean + half
    if lo_limit is not None:
        low = max(low, lo_limit)
    if hi_limit is not None:
        high = min(high, hi_limit)
    return {"min": round(low, 1), "max": round(high, 1), "promedio": round(mean, 1)}


def cmd_analyze(args):
    csv_path = os.path.join(DATA_DIR, f"{args.sign}.csv")
    if not os.path.exists(csv_path):
        print(f"No hay muestras para '{args.sign}'. Graba primero con: record --sign {args.sign}")
        sys.exit(1)

    per_finger = {f: [] for f in FINGER_JOINTS.keys()}
    per_axis = {a: [] for a in ORIENT_AXES}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            n += 1
            for finger in FINGER_JOINTS.keys():
                per_finger[finger].append(float(row[finger]))
            for axis in ORIENT_AXES:
                v = row.get(f"imu_{axis}")
                if v not in (None, ""):
                    per_axis[axis].append(float(v))

    if n < 5:
        print(f"Aviso: solo hay {n} muestras. Idealmente graben al menos 15-30 "
              "(varias personas repitiendo la seña) antes de fijar el rango.")

    all_tolerances = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            all_tolerances = json.load(f)
    widths = all_tolerances.get(args.sign, {}).get("ancho")
    centered = widths is not None and not args.solo_propias

    result = {}
    if centered:
        # Centro = la mediana de SUS muestras (su mano, su cámara); ancho = el que midió
        # evaluate_public.py con 20 personas del dataset público.
        result["ancho"] = widths
        method = "centrado en sus muestras, con el ancho del dataset público"
    else:
        method = f"promedio ± {args.margin} desviaciones de sus muestras"
    result["fuente"] = f"{n} muestras propias; {method}"
    print(f"\nResultados para '{args.sign}' ({n} muestras, {method}):\n")
    for finger, values in per_finger.items():
        if centered:
            med = statistics.median(values)
            down, up = widths[finger]
            result[finger] = {"min": round(max(med - down, 0.0), 1),
                              "max": round(min(med + up, 180.0), 1),
                              "promedio": round(statistics.mean(values), 1)}
        else:  # un ángulo real está en [0°, 180°]
            result[finger] = _range(values, args.margin, args.holgura_min, 0.0, 180.0)
        r = result[finger]
        print(f"  {finger:10s}  promedio={r['promedio']:6.1f}°  "
              f"rango sugerido=[{r['min']:.1f}°, {r['max']:.1f}°]")

    if all(len(v) >= MIN_IMU_SAMPLES for v in per_axis.values()):
        result["orientacion"] = {axis: _range(vals, args.margin, args.holgura_orient)
                                 for axis, vals in per_axis.items()}
        print("\n  Orientación de la muñeca (IMU):")
        for axis, r in result["orientacion"].items():
            print(f"  {axis:10s}  promedio={r['promedio']:6.1f}°  "
                  f"rango sugerido=[{r['min']:.1f}°, {r['max']:.1f}°]")
    else:
        print(f"\nAviso: menos de {MIN_IMU_SAMPLES} muestras con IMU; esta seña quedará "
              "SIN orientación (solo configuración de dedos). Regraba con la muñequera puesta.")

    # Guardar/actualizar tolerances.json
    all_tolerances[args.sign] = result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tolerances, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en {OUTPUT_FILE}. El motor de reglas puede leer este archivo directo.")


def main():
    parser = argparse.ArgumentParser(description="Calculadora de tolerancias LSM Coach")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Grabar muestras de una seña con la cámara")
    p_record.add_argument("--sign", required=True, help="Nombre de la seña, ej: A")
    p_record.add_argument("--person", required=True, help="Quién está grabando la muestra")
    p_record.add_argument("--imu", choices=["ble", "none"], default="ble",
                          help="ble = guarda también la orientación de la muñequera (default)")
    p_record.add_argument("--errores", action="store_true",
                          help=f"guarda ejecuciones INCORRECTAS a propósito en {ERRORS_DIR}/ "
                               "(sirven para medir aciertos con evaluate_accuracy.py)")
    p_record.set_defaults(func=cmd_record)

    p_analyze = sub.add_parser("analyze", help="Calcular rango de tolerancia de una seña")
    p_analyze.add_argument("--sign", required=True, help="Nombre de la seña a analizar")
    p_analyze.add_argument("--margin", type=float, default=2.0,
                            help="Cuántas desviaciones estándar de margen (default 2.0)")
    p_analyze.add_argument("--holgura-min", type=float, default=6.0,
                            help="holgura mínima en grados para los dedos (default 6)")
    p_analyze.add_argument("--holgura-orient", type=float, default=10.0,
                            help="holgura mínima en grados para roll/pitch (default 10)")
    p_analyze.add_argument("--solo-propias", action="store_true",
                            help="ignora el ancho del dataset público y usa promedio ± margen")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
