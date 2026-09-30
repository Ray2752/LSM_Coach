"""Entrena el reconocedor de señas de palabras (Nivel 3: HOLA, GRACIAS, POR FAVOR, AYUDA, MAMÁ)
con las muestras de samples_words/ (grabadas en la app con "Guardar correcta" o extraídas de
videos con extract_words.py).

Cada muestra es un trazo: forma de la mano, recorrido de la muñeca y, lo que distingue a las
palabras, la UBICACIÓN de la mano respecto al rostro (rostro, pecho, espacio neutro). Igual que
con las letras con movimiento, se añade la clase "quieta" para que la forma sola no baste.

Uso:
    python train_words.py                 # entrena y mide (deja fuera a cada persona si hay 2+)
    python train_words.py --sin-validar
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from dynamic import MIN_EXTENT, MIN_PATH, wrist_extent, wrist_travel
from tolerance_calculator import rotate_upright
from train_dynamic import T, resample

DATA_DIR = "samples_words"
MODEL_FILE = "model_words.joblib"
STILL = "quieta"
# Aumento: cada ejecución se añade también recortada al inicio y/o al final (fracciones). En vivo
# el detector de trazos no corta exactamente donde cortó al extraer el video; con esto POR FAVOR
# de una fuente nueva pasó de 0-40 % a 65-85 % de aciertos.
TRIMS = ((0.15, 0.0), (0.0, 0.15), (0.1, 0.1))


def face_box(face):
    """(cx, cy, ancho, alto) medianos del rostro en el clip, o None si nunca se detectó."""
    if face is None or len(face) == 0:
        return None
    face = np.asarray(face, dtype=np.float32)
    if not np.isfinite(face[:, 2]).any():
        return None
    return np.nanmedian(face, axis=0)


def word_features(feats, angles, wrist, face, ok):
    """Vector fijo por muestra. feats (T,63), angles (T,5), wrist (T,3: x, y, tamaño),
    face (T,4: cx, cy, ancho, alto; NaN si no hubo rostro) o None."""
    keep = np.asarray(ok).astype(bool)
    if keep.sum() >= 4:
        feats, angles, wrist = feats[keep], angles[keep], wrist[keep]
        if face is not None and len(face) == len(keep):
            face = face[keep]
    shape = rotate_upright(feats)
    tilt = np.arctan2(feats[:, 27], -feats[:, 28])[:, None]
    tilt = np.hstack([np.sin(tilt), np.cos(tilt)])
    size = max(float(np.median(wrist[:, 2])), 1e-3)
    path = (wrist[:, :2] - wrist[0, :2]) / size
    vel = np.vstack([np.zeros((1, 2)), np.diff(path, axis=0)])
    box = face_box(face)
    if box is not None:  # ubicación respecto al rostro, en anchos de rostro
        cx, cy, fw, _ = box
        fw = max(float(fw), 1e-3)
        loc = (wrist[:, :2] - [cx, cy]) / fw
        rel = [size / fw, 1.0]
    else:  # sin rostro: respecto al centro de la imagen, en tamaños de mano (y bandera 0)
        loc = (wrist[:, :2] - 0.5) / size * 0.25
        rel = [0.0, 0.0]
    seq = resample(np.hstack([shape, angles / 180.0, path, vel, tilt, loc]))  # (T, 76)
    extent = path.max(axis=0) - path.min(axis=0)
    turn = tilt[-1] - tilt[0]
    return np.concatenate([seq.ravel(), extent, turn, loc[0], loc[-1], loc.mean(axis=0), rel,
                           [len(ok) / 30.0]])


def still_word_sample(feats, angles, wrist, face, ok, rng, shake=0.012, walk=True):
    """Forma inicial sostenida con temblor (sin seña): clase "quieta"."""
    i = int(np.argmax(ok)) if np.asarray(ok).any() else 0
    n = 30
    size = max(float(wrist[i, 2]), 1e-3)
    jitter = rng.normal(0, shake * size, (n, 2))
    if walk:
        jitter = jitter.cumsum(axis=0)
    drift = rng.uniform(-0.3, 0.3, 2) * size * np.linspace(0, 1, n)[:, None]
    w = np.tile(wrist[i], (n, 1)); w[:, :2] += jitter + drift
    f = np.tile(feats[i], (n, 1)) + rng.normal(0, 0.01, (n, feats.shape[1]))
    a = np.tile(angles[i], (n, 1)) + rng.normal(0, 2.0, (n, angles.shape[1]))
    box = face_box(face)
    fc = None if box is None else np.tile(box, (n, 1))
    return word_features(f.astype(np.float32), a.astype(np.float32), w.astype(np.float32), fc,
                         np.ones(n, dtype=np.int8))


def load(with_still=True, trims=TRIMS):
    """X, y, persona y grupo (índice de la ejecución original: sus recortes y sus "quietas"
    comparten grupo para que la validación no los separe)."""
    X, y, persons, groups = [], [], [], []
    rng = np.random.default_rng(0)
    index = os.path.join(DATA_DIR, "index.csv")
    if not os.path.exists(index):
        sys.exit(f"No hay {index}. Graba señas en la app (Calibración > Guardar correcta con una "
                 "palabra elegida) o extrae videos con extract_words.py")
    skipped = Counter()
    with open(index, newline="", encoding="utf-8") as f:
        for g, r in enumerate(csv.DictReader(f)):
            path = os.path.join(DATA_DIR, r["word"], f"{r['file']}.npz")
            if not os.path.exists(path) or int(r["detected"]) < 6:
                continue
            d = np.load(path)
            seg = {"wrist": d["wrist"]}
            if wrist_travel(seg) < MIN_PATH or wrist_extent(seg) < MIN_EXTENT:
                skipped[r["word"]] += 1  # en vivo, dynamic.classify la daría por "quieta" sin mirarla
                continue
            face = d["face"] if "face" in d else None
            n = len(d["ok"])
            windows = [(0, n)] + [(int(n * a), n - int(n * b)) for a, b in trims]
            for i0, i1 in windows:
                if i1 - i0 < 8:
                    continue
                X.append(word_features(d["feats"][i0:i1], d["angles"][i0:i1], d["wrist"][i0:i1],
                                       None if face is None else face[i0:i1], d["ok"][i0:i1]))
                y.append(r["word"])
                persons.append(r["person"])
                groups.append(g)
            if with_still:
                for shake, walk in ((0.012, True), (rng.uniform(0.025, 0.045), True),
                                    (rng.uniform(0.015, 0.045), False)):
                    X.append(still_word_sample(d["feats"], d["angles"], d["wrist"], face, d["ok"],
                                               rng, shake, walk))
                    y.append(STILL)
                    persons.append(r["person"])
                    groups.append(g)
    if skipped:
        print(f"Se omiten {sum(skipped.values())} muestras con poco recorrido de la muñeca "
              f"(la app las rechazaría antes de clasificar): {dict(skipped)}")
    return np.array(X), np.array(y), np.array(persons), np.array(groups)


def make_model():
    from sklearn.ensemble import ExtraTreesClassifier
    # balanced: hay 3 "quietas" sintéticas por cada seña real y pocas muestras por palabra
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                random_state=0, n_jobs=-1)


def report(name, y_true, y_pred):
    strokes = y_true != STILL
    acc = float(np.mean(y_pred[strokes] == y_true[strokes])) if strokes.any() else 0.0
    print(f"\n{name}: señas {acc * 100:.0f}% ({int((y_pred[strokes] == y_true[strokes]).sum())}/{int(strokes.sum())})"
          + (f"  | quietas detectadas: {np.mean(y_pred[~strokes] == STILL) * 100:.0f}%" if (~strokes).any() else ""))
    labels = sorted(set(y_true) | set(y_pred))
    conf = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    width = max(len(l) for l in labels)
    print(" " * (width + 2) + " ".join(f"{l[:6]:>6s}" for l in labels))
    for t in labels:
        print(f"  {t:>{width}s} " + " ".join(f"{conf[t][p]:6d}" for p in labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sin-validar", action="store_true")
    ap.add_argument("--salida", default=MODEL_FILE)
    args = ap.parse_args()
    import joblib

    X, y, persons, groups = load()
    words = sorted(set(y) - {STILL})
    strokes = y != STILL
    real = Counter(y[strokes][np.unique(groups[strokes], return_index=True)[1]])  # ejecuciones reales
    sources = {w: sorted(set(persons[(y == w)])) for w in words}
    print(f"ejecuciones: {dict(sorted(real.items()))} de {len(set(persons))} persona(s)/fuente(s): "
          f"{', '.join(sorted(set(persons)))}; con recortes, {int(strokes.sum())} muestras de señas")
    few = [w for w, n in real.items() if n < 8]
    if few:
        print(f"Aviso: pocas muestras de {', '.join(few)} (ideal >= 15 por palabra y 2+ personas)")

    if not args.sin_validar:
        import warnings
        from sklearn.model_selection import StratifiedGroupKFold
        pred = np.empty_like(y)
        if len(set(persons)) >= 2:
            for person in sorted(set(persons)):  # cifra honesta: personas nuevas
                m = persons == person
                pred[m] = make_model().fit(X[~m], y[~m]).predict(X[m])
            single = [w for w, s in sources.items() if len(s) < 2]
            report("Dejando fuera a cada persona/fuente", y, pred)
            if single:
                print(f"  (solo cuentan las palabras con 2+ fuentes; {', '.join(single)} tienen una sola: "
                      "al dejarla fuera el modelo nunca vio esa palabra)")
        k = min(5, len(set(groups[strokes])))
        if k >= 2:  # cada ejecución con sus recortes y quietas en el mismo pliegue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=0).split(X, y, groups):
                    pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
            report(f"Validación {k}-fold por ejecución (mismas personas: cifra optimista)", y, pred)

    model = make_model().fit(X, y)
    joblib.dump({"model": model, "T": T, "signs": words, "kind": "words"}, args.salida, compress=3)
    print(f"\nModelo guardado en {args.salida} ({os.path.getsize(args.salida) / 1e6:.1f} MB): "
          f"{', '.join(words)}")


if __name__ == "__main__":
    main()
