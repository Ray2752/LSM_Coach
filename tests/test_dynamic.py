"""Pruebas del detector de trazos (sin cámara ni modelo).

    python -m unittest discover -s tests -t .
"""
import math
import unittest

import numpy as np

from dynamic import (MAX_S, MIN_EXTENT, MIN_PATH, MIN_S, STILL, STOP_S, GestureWindow, classify,
                     wrist_extent, wrist_travel)

FEATS = [0.0] * 63
ANGLES = [0.0] * 5
SIZE = 0.2  # tamaño de la mano en la imagen (0-1)


def feed_path(win, xs, fps=30, t0=0.0):
    """Alimenta posiciones x de la muñeca (y fija) a `fps`. Devuelve los segmentos que salieron."""
    out = []
    for i, x in enumerate(xs):
        seg = win.feed(t0 + i / fps, FEATS, ANGLES, (x, 0.5, SIZE))
        if seg:
            out.append(seg)
    return out


class GestureWindowTests(unittest.TestCase):
    def test_quieta_no_produce_trazo(self):
        win = GestureWindow()
        self.assertEqual(feed_path(win, [0.5] * 90), [])
        self.assertFalse(win.active)

    def test_trazo_claro_se_detecta_al_parar(self):
        win = GestureWindow()
        still = [0.5] * 15
        move = [0.5 + 0.02 * i for i in range(45)]   # 0.02 * 30 fps / 0.2 = 3 manos/s durante 1.5 s
        segs = feed_path(win, still + move + [move[-1]] * 30)
        self.assertEqual(len(segs), 1)
        self.assertGreaterEqual(segs[0]["duration"], MIN_S)
        self.assertGreater(len(segs[0]["feats"]), 30)
        self.assertFalse(win.active)

    def test_un_temblor_corto_se_ignora(self):
        win = GestureWindow()
        blip = [0.5, 0.52, 0.55, 0.57, 0.58]           # rápido pero dura 0.15 s
        self.assertEqual(feed_path(win, [0.5] * 15 + blip + [0.58] * 40), [])

    def test_trazo_largo_se_corta_en_el_tope(self):
        win = GestureWindow()
        n = int((MAX_S + 1.0) * 30)
        move = [0.1 + 0.01 * math.sin(i / 3) + 0.015 * i for i in range(n)]  # ~2.2 manos/s durante 4.5 s
        segs = feed_path(win, [0.1] * 10 + move)
        self.assertEqual(len(segs), 1)
        self.assertLessEqual(segs[0]["duration"], MAX_S + 0.1)

    def test_el_segmento_incluye_el_arranque(self):
        win = GestureWindow()
        move = [0.5 + 0.02 * i for i in range(40)]
        segs = feed_path(win, [0.5] * 30 + move + [move[-1]] * int(STOP_S * 30) + [move[-1]] * 5)
        self.assertEqual(len(segs), 1)
        self.assertLess(segs[0]["wrist"][0][0], 0.52)  # empieza cerca de la posición inicial


class StillGateTests(unittest.TestCase):
    def segment(self, xs):
        return {"feats": np.zeros((len(xs), 63), np.float32), "angles": np.zeros((len(xs), 5), np.float32),
                "wrist": np.array([[x, 0.5, SIZE] for x in xs], np.float32), "ok": np.ones(len(xs), np.int8)}

    def test_mide_el_recorrido_en_tamanos_de_mano(self):
        seg = self.segment([0.5, 0.5 + SIZE, 0.5 + 2 * SIZE])  # dos pasos de un tamaño de mano
        self.assertAlmostEqual(wrist_travel(seg), 2.0, places=5)

    def test_sin_movimiento_no_se_clasifica_como_letra(self):
        seg = self.segment([0.5 + 0.001 * i for i in range(30)])  # temblor: ~0.15 manos en total
        self.assertLess(wrist_travel(seg), MIN_PATH)
        self.assertEqual(classify(None, seg), (STILL, 1.0))  # ni siquiera toca el modelo

    def test_ruido_de_landmarks_recorre_mucho_pero_no_se_extiende(self):
        seg = self.segment([0.5 + (0.03 if i % 2 else -0.03) for i in range(30)])  # zigzag de ±0.15 manos
        self.assertGreater(wrist_travel(seg), MIN_PATH)   # el recorrido solo no basta
        self.assertLess(wrist_extent(seg), MIN_EXTENT)
        self.assertEqual(classify(None, seg), (STILL, 1.0))


if __name__ == "__main__":
    unittest.main()
