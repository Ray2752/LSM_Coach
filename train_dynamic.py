"""Entrena el clasificador de señas con movimiento (J, K, Ñ, Q, X, Z) con las trayectorias
que extrajo extract_dynamic.py (samples_dynamic/).

Cada video se remuestrea a T fotogramas y se describe con:
  - la forma de la mano en cada fotograma (landmarks enderezados + ángulos de los dedos)
  - el recorrido de la muñeca respecto al inicio, en unidades del tamaño de la mano
  - la velocidad de la muñeca
Con eso un bosque aleatorio distingue el trazo (J: gancho, Z: zigzag, X: gancho del índice...).

La cifra honesta es dejando fuera a cada persona (--por-persona) y la del split de prueba
del propio dataset (test/). Guarda model_dynamic.joblib.

Uso:
    python train_dynamic.py                 # entrena con train/, mide con test/
    python train_dynamic.py --por-persona   # además deja fuera a cada persona (tarda)
"""
import argparse
import csv
import os
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from tolerance_calculator import rotate_upright

DATA_DIR = "samples_dynamic"
MODEL_FILE = "model_dynamic.joblib"
T = 32  # fotogramas por secuencia tras remuestrear


def resample(a, n=T):
    """Interpola una secuencia (len, d) a n pasos igualmente espaciados."""
    if len(a) == 1:
        return np.repeat(a, n, axis=0)
    src = np.linspace(0, len(a) - 1, n)
    idx = np.arange(len(a))
    return np.stack([np.interp(src, idx, a[:, j]) for j in range(a.shape[1])], axis=1)


def sequence_features(feats, angles, wrist, ok):
    """Vector fijo por video. feats (T,63), angles (T,5), wrist (T,3: x, y, tamaño)."""
    keep = ok.astype(bool)
    if keep.sum() >= 4:  # se ignoran los fotogramas sin mano
        feats, angles, wrist = feats[keep], angles[keep], wrist[keep]
    shape = rotate_upright(feats)                       # forma sin la inclinación de la mano
    # La inclinación sí importa en el movimiento (la Q gira la muñeca hacia abajo; la K y la X
    # no): se guarda aparte como ángulo muñeca -> base del medio, fotograma a fotograma.
    tilt = np.arctan2(feats[:, 27], -feats[:, 28])[:, None]   # landmark 9 (x, y)
    tilt = np.hstack([np.sin(tilt), np.cos(tilt)])
    size = max(float(np.median(wrist[:, 2])), 1e-3)
    path = (wrist[:, :2] - wrist[0, :2]) / size          # recorrido de la muñeca desde el inicio
    vel = np.vstack([np.zeros((1, 2)), np.diff(path, axis=0)])
    tip = feats[:, 24:26] + path                         # punta del índice (landmark 8) en el espacio
    seq = np.hstack([shape, angles / 180.0, path, vel, tilt, tip])  # (len, 76)
    seq = resample(seq)
    extent = path.max(axis=0) - path.min(axis=0)         # cuánto se movió en x e y
    turn = tilt[-1] - tilt[0]                            # cuánto giró la mano de inicio a fin
    return np.concatenate([seq.ravel(), extent, turn, [len(ok) / 30.0]])


STILL = "quieta"  # forma de la letra sin movimiento: NO es la seña (ver dynamic.STILL)


def still_sample(feats, angles, wrist, ok, rng, shake=0.012, walk=True):
    """Ejemplo sintético de "quieta": la forma inicial del video, sostenida 1 s con temblor.
    shake = desviación por fotograma, en tamaños de mano. walk=True: paseo aleatorio (la mano
    se mueve un poco sin trazo); walk=False: ruido blanco (el temblor de los landmarks de
    MediaPipe con la mano quieta: recorre mucho pero no se extiende)."""
    i = int(np.argmax(ok)) if ok.any() else 0
    n = 30
    size = max(float(wrist[i, 2]), 1e-3)
    jitter = rng.normal(0, shake * size, (n, 2))
    if walk:
        jitter = jitter.cumsum(axis=0)
    drift = rng.uniform(-0.3, 0.3, 2) * size * np.linspace(0, 1, n)[:, None]  # deriva lenta
    w = np.tile(wrist[i], (n, 1)); w[:, :2] += jitter + drift
    f = np.tile(feats[i], (n, 1)) + rng.normal(0, 0.01, (n, feats.shape[1]))
    a = np.tile(angles[i], (n, 1)) + rng.normal(0, 2.0, (n, angles.shape[1]))
    return sequence_features(f.astype(np.float32), a.astype(np.float32), w.astype(np.float32),
                             np.ones(n, dtype=np.int8))


def load(split=None, with_still=True):
    """X, y, personas de samples_dynamic/ (solo el split indicado, o todos). Con `with_still`
    añade por cada video un ejemplo "quieta" (misma persona), para que el modelo aprenda que
    la forma sin trazo no es la letra."""
    X, y, persons = [], [], []
    rng = np.random.default_rng(0)
    with open(os.path.join(DATA_DIR, "index.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            d = np.load(os.path.join(DATA_DIR, r["split"], f"{r['file']}.npz"))
            if int(r["detected"]) < 4:
                continue
            wrist = d["wrist"].copy()
            if r["handed"] == "Left":  # los landmarks ya van reflejados; el recorrido también debe ir
                wrist[:, 0] = 1.0 - wrist[:, 0]
            X.append(sequence_features(d["feats"], d["angles"], wrist, d["ok"]))
            # macOS guarda "Ñ" en el nombre de archivo como N + tilde (NFD): se unifica con la
            # "Ñ" del catálogo (NFC) o el modelo aprendería una letra que la app nunca pide
            y.append(unicodedata.normalize("NFC", r["sign"]))
            persons.append(r["person"])
            if with_still:  # temblor fino, sacudida y ruido de landmarks: ninguno es la seña
                for shake, walk in ((0.012, True), (rng.uniform(0.025, 0.045), True),
                                    (rng.uniform(0.015, 0.045), False)):
                    X.append(still_sample(d["feats"], d["angles"], wrist, d["ok"], rng, shake, walk))
                    y.append(STILL)
                    persons.append(r["person"])
    return np.array(X), np.array(y), np.array(persons)


def make_model():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)


def report(name, y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    strokes = y_true != STILL
    print(f"\n{name}: {acc * 100:.0f}% ({int((y_true == y_pred).sum())}/{len(y_true)})"
          f"  | trazos reales: {np.mean(y_pred[strokes] == y_true[strokes]) * 100:.0f}%"
          + (f"  | quietas detectadas: {np.mean(y_pred[~strokes] == STILL) * 100:.0f}%" if (~strokes).any() else ""))
    labels = sorted(set(y_true) | set(y_pred))
    conf = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    print("      " + " ".join(f"{l:>4s}" for l in labels))
    for t in labels:
        print(f"  {t:>3s} " + " ".join(f"{conf[t][p]:4d}" for p in labels))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--por-persona", action="store_true", help="deja fuera a cada persona (lento)")
    ap.add_argument("--salida", default=MODEL_FILE)
    args = ap.parse_args()
    try:
        import joblib
    except ImportError:
        sys.exit("Instala: pip install scikit-learn joblib")

    Xtr, ytr, ptr = load("train")
    Xte, yte, pte = load("test")
    print(f"train: {len(ytr)} videos, {len(set(ptr))} personas, letras {dict(sorted(Counter(ytr).items()))}")
    print(f"test:  {len(yte)} videos, personas {sorted(set(pte))}")

    if len(yte):
        model = make_model().fit(Xtr, ytr)
        report("Split de prueba del dataset (personas nuevas)", yte, model.predict(Xte))

    if args.por_persona:
        preds = np.empty_like(ytr)
        for person in sorted(set(ptr)):
            m = ptr == person
            preds[m] = make_model().fit(Xtr[~m], ytr[~m]).predict(Xtr[m])
        report("Dejando fuera a cada persona (train)", ytr, preds)

    X, y = np.vstack([Xtr, Xte]) if len(yte) else Xtr, np.concatenate([ytr, yte]) if len(yte) else ytr
    model = make_model().fit(X, y)
    joblib.dump({"model": model, "T": T, "signs": sorted(set(y))}, args.salida, compress=3)
    print(f"\nModelo guardado en {args.salida} ({os.path.getsize(args.salida) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
