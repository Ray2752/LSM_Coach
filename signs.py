"""Catálogo de señas: el abecedario completo de LSM (27 letras).

Niveles del reto (Documento oficial, sección 4): Nivel 1 = A, B, C, L, Y (estáticas,
obligatorio); Nivel 2 = J, Ñ, Q, X, Z (con movimiento). Las demás letras están fuera
del alcance que se califica, pero la app las incluye para practicar todo el abecedario.

Las descripciones de las 10 letras del reto son las "generales y orientativas" del
documento; la valoración final de qué es una seña correcta corresponde a los expertos
en LSM. Para las demás letras la app muestra la imagen de referencia del dataset.
"""

NIVEL_1 = ["A", "B", "C", "L", "Y"]   # letras estáticas del reto (obligatorio)
NIVEL_2 = ["J", "Ñ", "Q", "X", "Z"]   # letras con movimiento del reto
ABECEDARIO = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ",
              "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
DYNAMIC = ["J", "K", "Ñ", "Q", "X", "Z"]  # llevan movimiento
STATIC = [s for s in ABECEDARIO if s not in DYNAMIC]
# Nivel 3 (opcional): vocabulario funcional. Son señas propias (no deletreo): se evalúan por
# configuración, ubicación respecto al cuerpo (rostro, pecho, espacio neutro) y movimiento.
NIVEL_3 = ["HOLA", "GRACIAS", "POR FAVOR", "AYUDA", "MAMÁ"]
WORDS = NIVEL_3

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
    "J": {"descripcion": "Meñique extendido que traza un movimiento en el aire.",
          "error_tipico": "Movimiento incompleto o trazado en sentido incorrecto."},
    "Ñ": {"descripcion": "Configuración de la N con un movimiento ondulante.",
          "error_tipico": "Omitir el movimiento; configuración base incorrecta."},
    "Q": {"descripcion": "Configuración específica con movimiento de la mano o muñeca.",
          "error_tipico": "Giro ausente o exagerado; orientación incorrecta."},
    "X": {"descripcion": "Dedo índice en gancho con movimiento.",
          "error_tipico": "Índice recto en lugar de en gancho; sin movimiento."},
    "Z": {"descripcion": "Dedo índice que traza la forma de la “Z” en el aire.",
          "error_tipico": "Trazo incompleto, invertido o demasiado pequeño."},
}
for _sign in ABECEDARIO:  # el resto del abecedario: se guía con la imagen de referencia
    SIGNS.setdefault(_sign, {"descripcion": "Observa la imagen de referencia.",
                             "error_tipico": ""})

# Palabras del reto (Documento oficial, sección 4.4): contexto y parámetros que se evalúan
SIGNS.update({
    "HOLA": {"descripcion": "Saludo e inicio de cualquier conversación.",
             "error_tipico": "Configuración, ubicación o movimiento distintos a la seña.",
             "parametros": "Configuración, ubicación y movimiento."},
    "GRACIAS": {"descripcion": "Cortesía básica en la convivencia diaria.",
                "error_tipico": "No parte del rostro, o falta el movimiento.",
                "parametros": "Configuración, ubicación respecto al rostro y movimiento."},
    "POR FAVOR": {"descripcion": "Pedir algo de forma amable.",
                  "error_tipico": "Mano lejos del cuerpo, o movimiento incompleto.",
                  "parametros": "Configuración, ubicación respecto al cuerpo y movimiento."},
    "AYUDA": {"descripcion": "Pedir u ofrecer ayuda; clave en situaciones de emergencia.",
              "error_tipico": "Orientación de la mano o movimiento incorrectos.",
              "parametros": "Configuración, orientación y movimiento."},
    "MAMÁ": {"descripcion": "Vocabulario familiar esencial en el hogar.",
             "error_tipico": "Mano lejos del rostro.",
             "parametros": "Configuración y ubicación respecto al rostro."},
})


def level(sign):
    return 1 if sign in NIVEL_1 else 2 if sign in NIVEL_2 else 3 if sign in NIVEL_3 else 0


def ref_name(sign, ext):
    """Nombre de archivo ASCII de la referencia (la Ñ no va bien en todas las rutas)."""
    return f"{'NN' if sign == 'Ñ' else sign}.{ext}"


def describe(sign):
    """Descripción de la seña para mostrar al aprendiz ('' si no está en el catálogo)."""
    return SIGNS.get(sign, {}).get("descripcion", "")
