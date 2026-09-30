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

from evaluation import evaluate, finger_spread, shape_issue, spread_issue
from tolerance_calculator import (DATA_DIR, ERRORS_DIR, FINGER_JOINTS, LANDMARK_COLUMNS,
                                  OUTPUT_FILE)

MODEL_FILE = "model.joblib"


def load_rows(path):
    """[(persona, {dedo: ángulo}, {roll,pitch,yaw} o None, landmarks o None)] de un CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            angles = {finger: float(r[finger]) for finger in FINGER_JOINTS}
            imu = None
            if all(r.get(f"imu_{a}") not in (None, "") for a in ("roll", "pitch", "yaw")):
                imu = {a: float(r[f"imu_{a}"]) for a in ("roll", "pitch", "yaw")}
            feats = ([float(r[c]) for c in LANDMARK_COLUMNS]
                     if all(r.get(c) for c in LANDMARK_COLUMNS) else None)
            rows.append((r.get("person", ""), angles, imu, feats))
    return rows


def sign_of(path):
    return os.path.splitext(os.path.basename(path))[0]


class Judge:
    """Decide como la app: reglas (dedos + muñeca) y, si hay modelo, el clasificador."""

    def __init__(self, model, require_imu):
        self.model, self.require_imu = model, require_imu
        self._proba = {}

    def probs(self, feats):
        key = tuple(feats)
        if key not in self._proba:
            p = self.model.predict_proba([feats])[0]
            self._proba[key] = dict(zip(map(str, self.model.classes_), p))
        return self._proba[key]

    def passes(self, row, sign, tol):
        _, angles, imu, feats = row
        if not evaluate(angles, imu, tol, self.require_imu).ok:
            return False
        if feats is not None and spread_issue(finger_spread(feats), sign):
            return False
        if self.model is None or feats is None or sign not in map(str, self.model.classes_):
            return True
        return shape_issue(self.probs(feats), sign) is None


def cross_person(signs, tolerances, require_imu):
    """Deja fuera a cada persona: entrena el clasificador con las demás (más el dataset
    público) y la evalúa con sus correctas y sus errores. Es la cifra honesta para alguien
    que el sistema nunca vio (como un integrante del jurado)."""
    from import_msl import PUBLIC_DIR
    from train_classifier import ERROR_SUFFIX, load_samples, make_model
    Xp, yp, _, _ = load_samples((PUBLIC_DIR,))
    correct = {s: load_rows(f"{DATA_DIR}/{s}.csv") for s in signs if os.path.exists(f"{DATA_DIR}/{s}.csv")}
    errors = {s: load_rows(f"{ERRORS_DIR}/{s}.csv") for s in signs if os.path.exists(f"{ERRORS_DIR}/{s}.csv")}
    all_c = {s: load_rows(p) for p in glob.glob(f"{DATA_DIR}/*.csv") for s in [sign_of(p)]}
    all_e = {s: load_rows(p) for p in glob.glob(f"{ERRORS_DIR}/*.csv") for s in [sign_of(p)]}
    people = sorted({r[0] for rows in list(correct.values()) + list(errors.values()) for r in rows})
    per_sign = {s: [0, 0, 0, 0] for s in signs}
    print(f"Dejando fuera a cada persona ({', '.join(people)}); tarda ~15 s por persona...\n")
    print(f"{'persona':10s} {'acepta correctas':>17s} {'rechaza errores':>16s}")
    for person in people:
        X, y = list(Xp), list(yp)
        for s, rows in all_c.items():
            X += [r[3] for r in rows if r[0] != person and r[3]]
            y += [s for r in rows if r[0] != person and r[3]]
        for s, rows in all_e.items():
            X += [r[3] for r in rows if r[0] != person and r[3]]
            y += [s + ERROR_SUFFIX for r in rows if r[0] != person and r[3]]
        judge = Judge(make_model().fit(X, y), require_imu)
        a = n = rj = ne = 0
        for s in signs:
            for r in correct.get(s, []):
                if r[0] == person:
                    ok = judge.passes(r, s, tolerances[s])
                    a, n = a + ok, n + 1
                    per_sign[s][0] += ok
                    per_sign[s][1] += 1
            for r in errors.get(s, []):
                if r[0] == person:
                    bad = not judge.passes(r, s, tolerances[s])
                    rj, ne = rj + bad, ne + 1
                    per_sign[s][2] += bad
                    per_sign[s][3] += 1
        pct = lambda x, t: f"{100 * x / t:.0f}% ({x}/{t})" if t else "—"
        print(f"{person:10s} {pct(a, n):>17s} {pct(rj, ne):>16s}")
    print(f"\n{'seña':5s} {'acepta correctas':>17s} {'rechaza errores':>16s}")
    tot = [0, 0, 0, 0]
    for s, v in per_sign.items():
        if v[1] or v[3]:
            print(f"{s:5s} {v[0] / max(v[1], 1) * 100:>16.0f}% {v[2] / max(v[3], 1) * 100:>15.0f}%")
            tot = [t + x for t, x in zip(tot, v)]
    if tot[1] and tot[3]:
        print(f"\nCon personas que el modelo no vio: acepta {100 * tot[0] / tot[1]:.0f}% de las correctas, "
              f"rechaza {100 * tot[2] / tot[3]:.0f}% de los errores -> aciertos correctas vs. errores "
              f"{100 * (tot[0] + tot[2]) / (tot[1] + tot[3]):.0f}% (meta >= 90%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signs", nargs="*", help="señas a evaluar (default: todas las calibradas)")
    ap.add_argument("--sin-imu", action="store_true",
                    help="ignora la orientación aunque la seña la tenga calibrada")
    ap.add_argument("--sin-clasificador", action="store_true",
                    help=f"solo reglas por dedo, sin {MODEL_FILE}")
    ap.add_argument("--por-persona", action="store_true",
                    help="deja fuera a cada persona y la evalúa con un modelo que no la vio "
                         "(la cifra honesta; tarda unos minutos)")
    args = ap.parse_args()

    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            tolerances = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No existe {OUTPUT_FILE}. Corre primero: python tolerance_calculator.py analyze --sign A")

    if args.por_persona:
        signs = args.signs or [sign_of(p) for p in sorted(glob.glob(f"{DATA_DIR}/*.csv"))
                               if sign_of(p) in tolerances]
        return cross_person(signs, tolerances, require_imu=not args.sin_imu)

    model = None
    if not args.sin_clasificador and os.path.exists(MODEL_FILE):
        import joblib
        model = joblib.load(MODEL_FILE)
    judge = Judge(model, require_imu=not args.sin_imu)
    print("Evaluando como la app: reglas por dedo" + (" + clasificador" if model else "")
          + ("" if args.sin_imu else " + orientación (si está calibrada)") + "\n")

    correct = {sign_of(p): load_rows(p) for p in sorted(glob.glob(f"{DATA_DIR}/*.csv"))}
    errors = {sign_of(p): load_rows(p) for p in sorted(glob.glob(f"{ERRORS_DIR}/*.csv"))}
    signs = args.signs or [s for s in tolerances if s in correct]

    def pct(a, b):
        return f"{100 * a / b:.0f}% ({a}/{b})" if b else "—"

    print(f"{'seña':5s} {'acepta correctas':>17s} {'rechaza errores':>16s} {'rechaza otras letras':>21s}")
    tot = {"pos": [0, 0], "err": [0, 0], "oth": [0, 0]}
    per_person = {}
    for sign in signs:
        tol = tolerances[sign]
        pos = correct.get(sign, [])
        err = errors.get(sign, [])
        others = [row for s, rows in correct.items() if s != sign for row in rows]
        a_pos = sum(judge.passes(r, sign, tol) for r in pos)
        r_err = sum(not judge.passes(r, sign, tol) for r in err)
        r_oth = sum(not judge.passes(r, sign, tol) for r in others)
        for key, x, n in (("pos", a_pos, len(pos)), ("err", r_err, len(err)), ("oth", r_oth, len(others))):
            tot[key][0] += x
            tot[key][1] += n
        for r in pos:
            per_person.setdefault(sign, {}).setdefault(r[0], [0, 0])
            per_person[sign][r[0]][0] += judge.passes(r, sign, tol)
            per_person[sign][r[0]][1] += 1
        print(f"{sign:5s} {pct(a_pos, len(pos)):>17s} {pct(r_err, len(err)):>16s} "
              f"{pct(r_oth, len(others)):>21s}")

    (ap_, np_), (re_, ne_), (ro_, no_) = tot["pos"], tot["err"], tot["oth"]
    if np_:
        print(f"\nAcepta correctas: {100 * ap_ / np_:.0f}%   "
              + (f"Rechaza errores intencionales: {100 * re_ / ne_:.0f}%   " if ne_ else "")
              + (f"Rechaza otras letras: {100 * ro_ / no_:.0f}%" if no_ else ""))
        if ne_:
            print(f"Aciertos correctas vs. errores (lo que prueba el jurado, meta >= 90%): "
                  f"{100 * (ap_ + re_) / (np_ + ne_):.0f}%")
    many = {s: d for s, d in per_person.items() if len(d) > 1}
    if many:
        print("\nCorrectas aceptadas por persona:")
        for sign, d in many.items():
            print(f"  {sign}: " + "  ".join(f"{p} {a}/{n}" for p, (a, n) in sorted(d.items())))
    if not errors:
        print("\nNota: no hay muestras en samples_errors/. Grábalas con 'Guardar error intencional' "
              "para medir errores (dedo mal flexionado, etc.).")
    print("\nOjo: si los rangos o el clasificador se calcularon con estas mismas muestras correctas,"
          " su % de aceptación sale optimista; el de errores no.")


if __name__ == "__main__":
    main()
