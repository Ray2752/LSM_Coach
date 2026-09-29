"""Mide qué tan bien distingue el sistema ejecuciones correctas de incorrectas.

La rúbrica (A2) pide >= 90 % de aciertos en la prueba en vivo. Aquí se estima con las
muestras grabadas, sin cámara ni muñequera:

  - correctas  : samples/<SEÑA>.csv                      -> debe aceptarlas
  - errores    : samples_errors/<SEÑA>.csv (record --errores) -> debe rechazarlas
  - otras letras: las muestras correctas de las demás señas -> debe rechazarlas
                  (hacer una B cuando se pide una A es un error)

Uso:
    python evaluate_accuracy.py                # todas las señas de tolerances.json
    python evaluate_accuracy.py --signs A B C L Y

Ojo: los rangos se calcularon con esas mismas muestras correctas, así que el % de
aceptación es optimista. La cifra confiable es la de una persona que NO grabó.
"""
import argparse
import csv
import glob
import json
import os
import sys

from evaluation import evaluate
from tolerance_calculator import (DATA_DIR, ERRORS_DIR, FINGER_JOINTS, OUTPUT_FILE)


def load_rows(path):
    """[(persona, {dedo: ángulo}, {roll,pitch,yaw} o None)] de un CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            angles = {finger: float(r[finger]) for finger in FINGER_JOINTS}
            imu = None
            if all(r.get(f"imu_{a}") not in (None, "") for a in ("roll", "pitch", "yaw")):
                imu = {a: float(r[f"imu_{a}"]) for a in ("roll", "pitch", "yaw")}
            rows.append((r.get("person", ""), angles, imu))
    return rows


def sign_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def accepted(rows, tol, require_imu):
    return sum(evaluate(a, imu, tol, require_imu).ok for _, a, imu in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signs", nargs="*", help="señas a evaluar (default: todas las calibradas)")
    ap.add_argument("--sin-imu", action="store_true",
                    help="ignora la orientación aunque la seña la tenga calibrada")
    args = ap.parse_args()

    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            tolerances = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No existe {OUTPUT_FILE}. Corre primero: python tolerance_calculator.py analyze --sign A")

    correct = {sign_of(p): load_rows(p) for p in sorted(glob.glob(f"{DATA_DIR}/*.csv"))}
    errors = {sign_of(p): load_rows(p) for p in sorted(glob.glob(f"{ERRORS_DIR}/*.csv"))}
    signs = args.signs or [s for s in tolerances if s in correct]

    print(f"{'seña':5s} {'correctas':>10s} {'errores':>10s} {'otras letras':>13s}   exactitud")
    tot_pos = tot_neg = ok_pos = ok_neg = 0
    for sign in signs:
        tol = tolerances[sign]
        pos = correct.get(sign, [])
        err = errors.get(sign, [])
        others = [row for s, rows in correct.items() if s != sign for row in rows]
        require = not args.sin_imu

        a_pos = accepted(pos, tol, require)
        r_err = len(err) - accepted(err, tol, require)
        r_oth = len(others) - accepted(others, tol, require)
        n_neg = len(err) + len(others)
        ok_pos += a_pos
        tot_pos += len(pos)
        ok_neg += r_err + r_oth
        tot_neg += n_neg

        def pct(a, b):
            return f"{100 * a / b:.0f}% ({a}/{b})" if b else "—"
        exact = (a_pos + r_err + r_oth) / (len(pos) + n_neg) if (pos or n_neg) else 0
        print(f"{sign:5s} {pct(a_pos, len(pos)):>10s} {pct(r_err, len(err)):>10s} "
              f"{pct(r_oth, len(others)):>13s}   {100 * exact:.0f}%")

    if tot_pos and tot_neg:
        tpr, tnr = ok_pos / tot_pos, ok_neg / tot_neg
        print(f"\nGeneral: aceptó {100 * tpr:.0f}% de las correctas y rechazó "
              f"{100 * tnr:.0f}% de las incorrectas.")
        print(f"Exactitud balanceada: {100 * (tpr + tnr) / 2:.0f}%  (meta de la rúbrica: >= 90%)")
    if not errors:
        print("\nNota: no hay muestras en samples_errors/. Grábalas con "
              "'record --errores' para medir errores intencionales (dedo mal flexionado, etc.).")


if __name__ == "__main__":
    main()
