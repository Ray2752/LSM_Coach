"""Entrena el clasificador de señas de LSM Coach.

Lee los CSV de samples/ (los que grabó tolerance_calculator.py), usa los 63
números de landmarks de cada muestra como características y guarda el modelo
en model.joblib, que live_evaluator.py carga en modo automático.

Las muestras viejas (solo ángulos, sin landmarks) se ignoran: hay que volver a
grabarlas con `python tolerance_calculator.py record --sign X --person Y`.

Uso:
    pip install scikit-learn joblib
    python train_classifier.py
    python train_classifier.py --publicos          # suma el dataset público (samples_public/)
    python train_classifier.py --salida model.joblib --min-muestras 10
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

from tolerance_calculator import DATA_DIR, LANDMARK_COLUMNS, rotate_upright

MODEL_FILE = "model.joblib"


def make_model():
    """Endereza la mano (rotate_upright) y clasifica con un bosque aleatorio. Es un
    Pipeline: quien lo cargue le pasa los 63 landmarks tal cual y él los endereza."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    return make_pipeline(FunctionTransformer(rotate_upright),
                         RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))


def load_samples(dirs=(DATA_DIR,), signs=None):
    """Devuelve (X, y, personas) solo con las filas que traen landmarks."""
    X, y, persons = [], [], []
    skipped = Counter()
    paths = [p for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.csv")))]
    for path in paths:
        if signs and os.path.splitext(os.path.basename(path))[0] not in signs:
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sign = row.get("sign") or os.path.splitext(os.path.basename(path))[0]
                if not all(row.get(c) for c in LANDMARK_COLUMNS):
                    skipped[sign] += 1
                    continue
                X.append([float(row[c]) for c in LANDMARK_COLUMNS])
                y.append(sign)
                persons.append(row.get("person", ""))
    return X, y, persons, skipped


def evaluate(make_model, X, y, persons):
    """Precisión con validación cruzada. Si hay 2+ personas se deja una persona
    fuera en cada vuelta (mide si generaliza a alguien nuevo); si no, k-fold."""
    import numpy as np
    from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                         cross_val_score)
    X, y = np.array(X), np.array(y)
    if len(set(persons)) >= 2:
        cv, groups, name = LeaveOneGroupOut(), np.array(persons), "dejando una persona fuera"
    else:
        k = min(5, min(Counter(y).values()))
        if k < 2:
            return None, None
        cv, groups, name = StratifiedKFold(k, shuffle=True, random_state=0), None, f"{k}-fold"
    scores = cross_val_score(make_model(), X, y, cv=cv, groups=groups)
    return float(scores.mean()), name


def main():
    ap = argparse.ArgumentParser(description="Entrena el clasificador de señas")
    ap.add_argument("--salida", default=MODEL_FILE)
    ap.add_argument("--min-muestras", type=int, default=5,
                    help="mínimo de muestras con landmarks por seña para incluirla (default 5)")
    ap.add_argument("--publicos", action="store_true",
                    help="incluye el dataset público importado con import_msl.py (samples_public/)")
    ap.add_argument("--signs", nargs="+", help="solo estas señas (default: todas)")
    args = ap.parse_args()

    try:
        import joblib
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("Faltan dependencias. Instala: pip install scikit-learn joblib")

    from import_msl import PUBLIC_DIR
    dirs = (DATA_DIR, PUBLIC_DIR) if args.publicos else (DATA_DIR,)
    X, y, persons, skipped = load_samples(dirs, args.signs)
    if skipped:
        print("Muestras ignoradas por no tener landmarks (formato viejo): "
              + ", ".join(f"{s}={n}" for s, n in sorted(skipped.items())))

    counts = Counter(y)
    keep = {s for s, n in counts.items() if n >= args.min_muestras}
    for s in sorted(set(counts) - keep):
        print(f"Aviso: '{s}' tiene solo {counts[s]} muestras con landmarks; se excluye.")
    rows = [(x, l, p) for x, l, p in zip(X, y, persons) if l in keep]
    if len(keep) < 2:
        sys.exit("Necesito al menos 2 señas con landmarks para entrenar. Vuelve a grabar "
                 "con: python tolerance_calculator.py record --sign X --person Y")
    X, y, persons = zip(*rows)

    print("\nMuestras usadas:")
    for s, n in sorted(Counter(y).items()):
        print(f"  {s}: {n} ({', '.join(sorted({p for l, p in zip(y, persons) if l == s}))})")

    acc, how = evaluate(make_model, X, y, persons)
    if acc is None:
        print("\nNo hay muestras suficientes para validar el modelo.")
    else:
        print(f"\nPrecisión ({how}): {acc * 100:.0f}%")
        if len(set(persons)) < 2:
            print("Aviso: todas las muestras son de una sola persona; esta cifra es "
                  "optimista. Graba con más gente para que generalice.")

    model = make_model().fit(list(X), list(y))
    joblib.dump(model, args.salida)
    print(f"\nModelo guardado en {args.salida} (señas: {', '.join(model.classes_)})")


if __name__ == "__main__":
    main()
