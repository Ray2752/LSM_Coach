"""Catálogo de señas del alcance (Documento oficial del reto, sección 4).

Las descripciones son las "generales y orientativas" del documento; la valoración
final de qué es una seña correcta corresponde a los expertos en LSM del panel.
"""

NIVEL_1 = ["A", "B", "C", "L", "Y"]  # letras estáticas (obligatorio)

# "forma": cómo va cada dedo, para mostrarlo al aprendiz (según la descripción del reto)
SIGNS = {
    "A": {"descripcion": "Mano cerrada, con el pulgar a un costado.",
          "error_tipico": "Pulgar mal colocado; dedos no cerrados por completo.",
          "forma": {"pulgar": "extendido al costado", "indice": "cerrado", "medio": "cerrado",
                    "anular": "cerrado", "menique": "cerrado"}},
    "B": {"descripcion": "Mano plana, dedos extendidos y juntos.",
          "error_tipico": "Dedos separados o semiflexionados.",
          "forma": {"pulgar": "doblado sobre la palma", "indice": "extendido", "medio": "extendido",
                    "anular": "extendido", "menique": "extendido"}},
    "C": {"descripcion": "Mano curvada en forma de “C”.",
          "error_tipico": "Curvatura insuficiente o excesiva; orientación incorrecta.",
          "forma": {"pulgar": "curvado", "indice": "curvado", "medio": "curvado",
                    "anular": "curvado", "menique": "curvado"}},
    "L": {"descripcion": "Índice y pulgar extendidos formando una “L”.",
          "error_tipico": "Dedo medio u otros dedos no flexionados.",
          "forma": {"pulgar": "extendido", "indice": "extendido", "medio": "cerrado",
                    "anular": "cerrado", "menique": "cerrado"}},
    "Y": {"descripcion": "Pulgar y meñique extendidos.",
          "error_tipico": "Anular o índice parcialmente extendidos.",
          "forma": {"pulgar": "extendido", "indice": "cerrado", "medio": "cerrado",
                    "anular": "cerrado", "menique": "extendido"}},
}


def describe(sign):
    """Descripción de la seña para mostrar al aprendiz ('' si no está en el catálogo)."""
    return SIGNS.get(sign, {}).get("descripcion", "")
