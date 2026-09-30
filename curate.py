"""Aparta muestras mal etiquetadas (p. ej. una "Y" grabada con otra letra seleccionada).

Nada se borra: las filas se mueven a apartadas/correctas/<SEÑA>.csv con el motivo. Para
deshacer, se copian de vuelta (o con git).

Nota: se probó apartar automáticamente lo que "contradice" a una persona de referencia
(vecinos más cercanos), pero las manos de personas distintas difieren más entre sí que
una seña correcta de un error, así que apartaba muestras buenas. Por eso solo es manual.

Uso:
    python curate.py exp3:Y exp3:L               # solo muestra qué apartaría
    python curate.py exp3:Y exp3:L --aplicar
    python curate.py --espejo exp3 --aplicar     # refleja las muestras de quien grabó con la
                                                 # mano izquierda (repetirlo lo deshace)
"""
import argparse
import csv
import os

from tolerance_calculator import DATA_DIR, ERRORS_DIR, LANDMARK_COLUMNS, csv_header, mirror_features

OUT_DIR = "apartadas"


def read(path):
    if not os.path.exists(path):
        return [], csv_header()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def move(rows, fields, keep_idx_out, sign, subdir, reasons):
    """Escribe el CSV sin las filas apartadas y las agrega a apartadas/<subdir>/<seña>.csv."""
    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)
    out_path = os.path.join(OUT_DIR, subdir, f"{sign}.csv")
    is_new = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["motivo"])
        if is_new:
            w.writeheader()
        for i in sorted(keep_idx_out):
            w.writerow({**rows[i], "motivo": reasons[i]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("marcas", nargs="*", metavar="PERSONA:SEÑA",
                    help="aparta todas las muestras correctas de esa persona en esa seña")
    ap.add_argument("--espejo", nargs="+", metavar="PERSONA", default=[],
                    help="refleja (mano izquierda -> derecha) todas las muestras de estas personas")
    ap.add_argument("--aplicar", action="store_true", help="sin esto solo muestra qué haría")
    args = ap.parse_args()

    for person in args.espejo:
        n = 0
        for d in (DATA_DIR, ERRORS_DIR):
            for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                path = os.path.join(d, name)
                rows, fields = read(path)
                hit = [r for r in rows if r["person"] == person and all(r.get(c) for c in LANDMARK_COLUMNS)]
                n += len(hit)
                if args.aplicar and hit:
                    for r in hit:
                        mirrored = mirror_features([float(r[c]) for c in LANDMARK_COLUMNS])
                        r.update({c: v for c, v in zip(LANDMARK_COLUMNS, mirrored)})
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        w.writerows(rows)
        print(f"{'Reflejadas' if args.aplicar else 'Se reflejarían'}: {n} muestras de {person}")

    total = 0
    for mark in args.marcas:
        person, sign = mark.split(":", 1)
        path = os.path.join(DATA_DIR, f"{sign}.csv")
        rows, fields = read(path)
        bad = {i for i, r in enumerate(rows) if r["person"] == person}
        print(f"{sign}: {len(bad)} correctas de {person}")
        total += len(bad)
        if args.aplicar and bad:
            move(rows, fields, bad, sign, "correctas",
                 {i: "etiqueta equivocada (revisión manual)" for i in bad})
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(r for i, r in enumerate(rows) if i not in bad)
    verb = "Apartadas" if args.aplicar else "Se apartarían"
    print(f"\n{verb}: {total} muestras" + ("" if args.aplicar else "  (usa --aplicar para hacerlo)"))


if __name__ == "__main__":
    main()
