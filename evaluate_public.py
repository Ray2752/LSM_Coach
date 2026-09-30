"""Mide el sistema con el dataset público usando personas que NO se usaron para calibrar.

Los rangos por dedo se calculan con los participantes de entrenamiento del dataset y se
prueban con los de prueba (por defecto S18 y S20, los que el propio dataset reserva).
Mide lo que pide la rúbrica, con tres cifras:
  - acepta correctas ....... la seña bien hecha por alguien nuevo pasa
  - rechaza otras letras ... hacer otra letra no pasa
  - detecta error de dedo .. la seña con UN dedo semiflexionado / semiextendido no pasa
    (error sintético: se toma una muestra correcta y se mueve el ángulo de un dedo)

Los rangos usan percentiles (resisten mejor los valores raros que promedio +/- desviación)
y por defecto solo los grupos A y B del dataset: el grupo C trae la mano rotada, lo que
ensancha mucho los rangos. El clasificador sí aprende de los tres grupos.

Uso:
    python import_msl.py --root ~/Downloads/datos_ent/MSL-ABC     # antes, una vez
    python evaluate_public.py
    python evaluate_public.py --percentiles 3 97 --guardar        # rangos -> tolerances.json
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

from evaluation import evaluate, shape_issue
from import_msl import PUBLIC_DIR, SOURCE
from tolerance_calculator import DATA_DIR, FINGER_JOINTS, LANDMARK_COLUMNS, OUTPUT_FILE, one_sided

PERCENTILES = ((5, 95), (3, 97), (2, 98), (1, 99))
FLOOR = 6.0          # holgura mínima (grados) a cada lado de la mediana
ERROR_SHIFT = 0.6    # qué tanto se mueve el dedo en el error sintético (0-1 hacia el otro extremo)
CURVED_SHIFT = 40.0  # para dedos curvados (C): +/- grados
EXPERT_PCT = (10, 90)  # --expertos: parte de la forma de cada experto que el rango debe cubrir
EXPERT_MIN = 3         # --expertos: muestras mínimas de esa persona en la seña


def load(folder):
    """{seña: [(persona, grupo, {dedo: ángulo}, landmarks o None)]}"""
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        sign = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                angles = {k: float(r[k]) for k in FINGER_JOINTS}
                feats = ([float(r[c]) for c in LANDMARK_COLUMNS]
                         if all(r.get(c) for c in LANDMARK_COLUMNS) else None)
                data[sign].append((r["person"], r.get("grupo", ""), angles, feats))
    return data


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, max(0, round(q / 100 * (len(v) - 1))))]


def pct_range(values, lo, hi):
    med = statistics.median(values)
    low = min(pct(values, lo), med - FLOOR)
    high = max(pct(values, hi), med + FLOOR)
    return one_sided({"min": round(max(low, 0.0), 1), "max": round(min(high, 180.0), 1),
                      "promedio": round(statistics.mean(values), 1)})


def ranges_from(rows, lo, hi):
    return {k: pct_range([r[2][k] for r in rows], lo, hi) for k in FINGER_JOINTS}


CALIB_N = 10  # muestras por seña en la calibración breve simulada


def widths_from(rows, lo, hi):
    """Cuánto se aleja la población de su mediana, por dedo: (hacia abajo, hacia arriba)."""
    out = {}
    for k in FINGER_JOINTS:
        values = [r[2][k] for r in rows]
        med = statistics.median(values)
        out[k] = (max(med - pct(values, lo), FLOOR), max(pct(values, hi) - med, FLOOR))
    return out


def personal_ranges(widths, calib_rows):
    """Rango centrado en la mediana de la persona, con el ancho de la población."""
    tol = {}
    for k, (down, up) in widths.items():
        values = [r[2][k] for r in calib_rows]
        med = statistics.median(values)
        tol[k] = one_sided({"min": round(max(med - down, 0.0), 1),
                            "max": round(min(med + up, 180.0), 1),
                            "promedio": round(statistics.mean(values), 1)})
    return tol


def ok(angles, tol):
    return evaluate(angles, None, tol, require_imu=False).ok


def finger_errors(angles, tol):
    """Versiones de la muestra con UN dedo mal: flexionado -> semiextendido,
    extendido -> semiflexionado, curvado -> más cerrado y más abierto."""
    out = []
    for k in FINGER_JOINTS:
        med = tol[k]["promedio"]
        if med < 90:
            bad = [med + ERROR_SHIFT * (180 - med)]
        elif med > 140:
            bad = [med * (1 - ERROR_SHIFT)]
        else:
            bad = [med - CURVED_SHIFT, min(180.0, med + CURVED_SHIFT)]
        out += [{**angles, k: b} for b in bad]
    return out


def score(tols, test):
    """(acepta correctas, rechaza otras letras, detecta error de dedo), cada una en 0-1,
    y el detalle por seña."""
    detail = {}
    tot = [0, 0, 0, 0, 0, 0]
    for sign, tol in tols.items():
        pos = test.get(sign, [])
        neg = [r for s, rows in test.items() if s != sign for r in rows]
        errs = [e for r in pos for e in finger_errors(r[2], tol)]
        a = sum(ok(r[2], tol) for r in pos)
        rj = sum(not ok(r[2], tol) for r in neg)
        de = sum(not ok(e, tol) for e in errs)
        detail[sign] = (a / max(len(pos), 1), rj / max(len(neg), 1), de / max(len(errs), 1))
        for i, (x, n) in enumerate(((a, len(pos)), (rj, len(neg)), (de, len(errs)))):
            tot[2 * i] += x
            tot[2 * i + 1] += n
    return tuple(tot[2 * i] / max(tot[2 * i + 1], 1) for i in range(3)), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prueba", nargs="+", default=["S18", "S20"],
                    help="participantes que NO se usan para calibrar (default: S18 S20)")
    ap.add_argument("--grupos", nargs="+", default=["A", "B"],
                    help="grupos del dataset para los rangos por dedo (default: A B)")
    ap.add_argument("--percentiles", nargs=2, type=float, metavar=("BAJO", "ALTO"),
                    help="percentiles para --guardar (default: el mejor de la tabla)")
    ap.add_argument("--incluir-propias", action="store_true",
                    help="al guardar, amplía cada rango para que también acepte las muestras "
                         f"correctas propias ({DATA_DIR}/, percentiles 5-95, sin valores extremos)")
    ap.add_argument("--expertos", nargs="+", metavar="PERSONA", default=[],
                    help="al guardar, amplía cada rango para que acepte la forma típica de cada una de "
                         f"estas personas (percentiles {EXPERT_PCT[0]}-{EXPERT_PCT[1]} de sus correctas, "
                         f"si tiene >= {EXPERT_MIN} en esa seña)")
    ap.add_argument("--guardar", action="store_true",
                    help=f"guarda en {OUTPUT_FILE} rangos calculados con TODOS los participantes")
    args = ap.parse_args()

    data = load(PUBLIC_DIR)
    if not data:
        raise SystemExit(f"No hay datos en {PUBLIC_DIR}/. Corre primero import_msl.py")
    if not any(r[1] for rows in data.values() for r in rows):
        raise SystemExit("Los CSV no traen la columna 'grupo': vuelve a correr import_msl.py")
    test_people = {f"msl-{p}" for p in args.prueba}
    groups = set(args.grupos)
    in_groups = {s: [r for r in rows if r[1] in groups] for s, rows in data.items()}
    train = {s: [r for r in rows if r[0] not in test_people] for s, rows in in_groups.items()}
    test = {s: [r for r in rows if r[0] in test_people] for s, rows in in_groups.items()}
    print(f"Señas: {', '.join(sorted(data))}.  Grupos: {', '.join(sorted(groups))}.  "
          f"Calibración: {len({r[0] for rows in train.values() for r in rows})} personas, "
          f"{sum(map(len, train.values()))} muestras.  Prueba: {', '.join(sorted(test_people))}, "
          f"{sum(map(len, test.values()))} muestras.\n")

    print("Reglas por dedo (lo que usa la app), personas no vistas:")
    print(f"  {'percentiles':>11}  {'acepta correctas':>16}  {'rechaza otras letras':>20}  "
          f"{'detecta error de dedo':>21}")
    results = {}
    for lo, hi in PERCENTILES:
        (a, rj, de), detail = score({s: ranges_from(r, lo, hi) for s, r in train.items()}, test)
        results[(lo, hi)] = (a, rj, de, detail)
        print(f"  {f'{lo}-{hi}':>11}  {100 * a:>15.0f}%  {100 * rj:>19.0f}%  {100 * de:>20.0f}%")
    # El mejor: el más estricto con errores entre los que aceptan >= 90 % de las correctas
    good = [k for k, v in results.items() if v[0] >= 0.90] or list(results)
    best = max(good, key=lambda k: (results[k][2], results[k][0]))
    lo, hi = args.percentiles or best
    detail = results.get((lo, hi), (0, 0, 0, None))[3]
    if detail:
        print(f"\nDetalle con percentiles {lo}-{hi}:")
        for sign, (a, rj, de) in detail.items():
            print(f"  {sign}: acepta {100 * a:.0f}%  rechaza otras {100 * rj:.0f}%  "
                  f"detecta error de dedo {100 * de:.0f}%")

    # Calibración breve por persona: el centro del rango sale de SU mano (pocas muestras)
    # y el ancho, de la población. Se simula con las personas de prueba.
    print(f"\nCon calibración breve por persona ({CALIB_N} muestras por seña; se prueba con el resto):")
    print(f"  {'percentiles':>11}  {'acepta correctas':>16}  {'rechaza otras letras':>20}  "
          f"{'detecta error de dedo':>21}")
    for plo, phi in PERCENTILES:
        widths = {s: widths_from(r, plo, phi) for s, r in train.items()}
        tot = [0] * 6
        for person in sorted(test_people):
            mine = {s: [r for r in rows if r[0] == person] for s, rows in test.items()}
            calib = {s: rows[::max(1, len(rows) // CALIB_N)][:CALIB_N] for s, rows in mine.items()}
            rest = {s: [r for r in rows if r not in calib[s]] for s, rows in mine.items()}
            tols_p = {s: personal_ranges(widths[s], calib[s]) for s in widths if calib.get(s)}
            (a, rj, de), _ = score(tols_p, rest)
            n_pos = sum(map(len, rest.values()))
            for i, (v, n) in enumerate(((a, n_pos), (rj, n_pos * (len(rest) - 1)), (de, n_pos))):
                tot[2 * i] += v * n
                tot[2 * i + 1] += n
        a, rj, de = (tot[2 * i] / max(tot[2 * i + 1], 1) for i in range(3))
        print(f"  {f'{plo}-{phi}':>11}  {100 * a:>15.0f}%  {100 * rj:>19.0f}%  {100 * de:>20.0f}%")

    # ¿Los rangos aceptan lo grabado con NUESTRA cámara?
    tols = {s: ranges_from(r, lo, hi) for s, r in train.items()}
    own = load(DATA_DIR)
    common = [s for s in tols if own.get(s)]
    if common:
        print("\nMuestras propias de samples/ (su cámara, sus personas):  "
              + "  ".join(f"{s}: {sum(ok(r[2], tols[s]) for r in own[s])}/{len(own[s])}"
                          for s in common))

    # Clasificador de landmarks (todos los grupos) y fusión reglas + clasificador
    try:
        from train_classifier import make_model
        tr = [(s, r[3]) for s, rows in data.items() for r in rows
              if r[3] and r[0] not in test_people]
        te = [(s, r[2], r[3]) for s, rows in test.items() for r in rows if r[3]]
        clf = make_model()
        clf.fit([f for _, f in tr], [s for s, _ in tr])
        proba = clf.predict_proba([f for _, _, f in te])
        pred = [clf.classes_[p.argmax()] for p in proba]
        hit = sum(p == s for p, (s, _, _) in zip(pred, te))
        print(f"\nClasificador (reconoce la letra), personas no vistas: {100 * hit / len(te):.1f}%")
        by_sign = defaultdict(list)
        for p, (s, _, _) in zip(pred, te):
            by_sign[s].append(p)
        weak = []
        for s, ps in sorted(by_sign.items()):
            acc = sum(p == s for p in ps) / len(ps)
            if acc < 0.9:
                worst = max((p for p in set(ps) if p != s), key=ps.count)
                weak.append(f"{s} {100 * acc:.0f}% (la confunde con {worst})")
        if weak:
            print("  Letras por debajo de 90 %: " + ", ".join(weak))
        tp = fp = npos = nneg = 0
        for sign, tol in tols.items():
            for p, (real, angles, _) in zip(proba, te):
                passed = ok(angles, tol) and shape_issue(dict(zip(clf.classes_, p)), sign) is None
                npos += real == sign
                nneg += real != sign
                tp += passed and real == sign
                fp += passed and real != sign
        print(f"Reglas + clasificador: acepta {100 * tp / npos:.0f}% de correctas, "
              f"rechaza {100 * (1 - fp / nneg):.0f}% de otras letras")
    except ImportError:
        print("\n(sin scikit-learn: se omite el clasificador)")

    if args.guardar:
        full = {s: ranges_from(rows, lo, hi) for s, rows in in_groups.items()}
        full_widths = {s: widths_from(rows, lo, hi) for s, rows in in_groups.items()}
        n_people = len({r[0] for rows in in_groups.values() for r in rows})
        all_tol = {}
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                all_tol = json.load(f)
        if args.incluir_propias:
            for sign, rng in full.items():
                rows = own.get(sign, [])
                if len(rows) < 10:
                    continue
                for k in FINGER_JOINTS:
                    values = [r[2][k] for r in rows]
                    rng[k] = one_sided({"min": round(min(rng[k]["min"], pct(values, 5)), 1),
                                        "max": round(max(rng[k]["max"], pct(values, 95)), 1),
                                        "promedio": rng[k]["promedio"]})
        # Juntas, las muestras de un experto con pocas grabaciones quedan en las orillas y los
        # percentiles 5-95 las recortan: la app le decía "Extiende el índice" a una F correcta.
        # Por eso cada experto amplía el rango con SU forma típica (sin su valor más raro).
        for sign, rng in full.items():
            for person in args.expertos:
                rows = [r for r in own.get(sign, []) if r[0] == person]
                if len(rows) < EXPERT_MIN:
                    continue
                for k in FINGER_JOINTS:
                    values = [r[2][k] for r in rows]
                    rng[k] = one_sided({"min": round(min(rng[k]["min"], pct(values, EXPERT_PCT[0])), 1),
                                        "max": round(max(rng[k]["max"], pct(values, EXPERT_PCT[1])), 1),
                                        "promedio": rng[k]["promedio"]})
        for sign, rng in full.items():
            old = all_tol.get(sign, {})
            entry = {**rng, "fuente": f"{SOURCE}, grupos {''.join(sorted(groups))}, "
                                      f"{n_people} personas, percentiles {lo}-{hi}"
                                      + (" + muestras propias" if args.incluir_propias else "")
                                      + (f" + expertos: {', '.join(args.expertos)}" if args.expertos else ""),
                     # ancho de la población (abajo, arriba) para centrar el rango en cada usuario
                     "ancho": {k: [round(d, 1), round(u, 1)] for k, (d, u) in full_widths[sign].items()}}
            if old.get("orientacion"):  # la orientación (IMU) se conserva: el dataset no la tiene
                entry["orientacion"] = old["orientacion"]
            all_tol[sign] = entry
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_tol, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado en {OUTPUT_FILE}: {', '.join(full)} (percentiles {lo}-{hi}).")


if __name__ == "__main__":
    main()
