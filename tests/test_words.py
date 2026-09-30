"""Pruebas de las características de señas de palabras (sin cámara ni modelo).

    python -m unittest discover -s tests -t .
"""
import math
import unittest

import numpy as np

from train_words import IMU_N, face_box, still_word_sample, word_features


def clip(n=40, with_face=True):
    rng = np.random.default_rng(0)
    feats = rng.normal(0, .5, (n, 63)).astype(np.float32)
    angles = rng.uniform(0, 180, (n, 5)).astype(np.float32)
    x = np.linspace(.4, .6, n)
    wrist = np.stack([x, np.full(n, .7), np.full(n, .15)], axis=1).astype(np.float32)
    face = np.tile([.5, .3, .2, .25], (n, 1)).astype(np.float32) if with_face else np.full((n, 4), math.nan, np.float32)
    return feats, angles, wrist, face, np.ones(n, np.int8)


class WordFeatureTests(unittest.TestCase):
    def test_vector_fijo_con_y_sin_rostro(self):
        a = word_features(*clip(40, True))
        b = word_features(*clip(25, False))
        self.assertEqual(a.shape, b.shape)          # mismo tamaño aunque cambie la duración
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        # cola del vector: ..., loc.mean (x, y), rel (tamaño/rostro, bandera de rostro), duración
        self.assertEqual(a[-2], 1.0)                 # bandera: hubo rostro
        self.assertEqual(b[-2], 0.0)                 # no hubo rostro

    def test_ubicacion_respecto_al_rostro(self):
        feats, angles, wrist, face, ok = clip()
        wrist[:, 1] = .3          # mano a la altura del rostro
        near = word_features(feats, angles, wrist, face, ok)
        wrist[:, 1] = .9          # mano abajo, al pecho
        far = word_features(feats, angles, wrist, face, ok)
        self.assertLess(abs(near[-4]), abs(far[-4]))  # loc.mean(y): más lejos del rostro

    def test_face_box_ignora_nan(self):
        face = np.full((10, 4), math.nan, np.float32)
        self.assertIsNone(face_box(face))
        face[3] = [.5, .3, .2, .25]
        self.assertAlmostEqual(float(face_box(face)[2]), .2, places=5)

    def test_muestra_quieta_no_falla_sin_rostro(self):
        feats, angles, wrist, face, ok = clip(30, False)
        v = still_word_sample(feats, angles, wrist, face, ok, np.random.default_rng(1))
        self.assertTrue(np.isfinite(v).all())


def glove(n=40, roll=(10.0, 40.0), pitch=(-5.0, -5.0)):
    """Inclinación del guante durante la seña: roll y pitch en grados, lineales."""
    return np.stack([np.linspace(*roll, n), np.linspace(*pitch, n)], axis=1).astype(np.float32)


class GloveFeatureTests(unittest.TestCase):
    def test_sin_guante_el_vector_no_cambia(self):
        a = word_features(*clip(40))
        b = word_features(*clip(40), imu=glove(40), use_imu=False)
        np.testing.assert_array_equal(a, b)

    def test_con_guante_se_agregan_rasgos_y_bandera(self):
        base = word_features(*clip(40))
        con = word_features(*clip(40), imu=glove(40), use_imu=True)
        sin = word_features(*clip(40), imu=None, use_imu=True)
        self.assertEqual(len(con), len(base) + IMU_N)
        self.assertEqual(len(sin), len(base) + IMU_N)
        self.assertEqual(con[-1], 1.0)                       # bandera: hubo guante
        self.assertTrue((sin[-IMU_N:] == 0).all())           # sin guante: ceros y bandera 0
        self.assertTrue(np.isfinite(con).all())
        self.assertAlmostEqual(con[-1 - 8 + 3], (40.0 - 10.0) / 90.0, places=5)  # cambio de roll

    def test_lecturas_perdidas_no_rompen(self):
        imu = glove(40)
        imu[5:30] = math.nan                                  # el guante se desconectó a mitad
        v = word_features(*clip(40), imu=imu, use_imu=True)
        self.assertTrue(np.isfinite(v).all())
        imu[:] = math.nan                                     # nunca hubo lectura
        v = word_features(*clip(40), imu=imu, use_imu=True)
        self.assertEqual(v[-1], 0.0)

    def test_quieta_con_guante_mantiene_la_inclinacion(self):
        feats, angles, wrist, face, ok = clip(30)
        v = still_word_sample(feats, angles, wrist, face, ok, np.random.default_rng(1),
                              imu=glove(30, roll=(20.0, 60.0)), use_imu=True)
        self.assertEqual(v[-1], 1.0)
        self.assertLess(abs(v[-1 - 8 + 3]), 0.15)             # casi sin cambio de roll: quieta


if __name__ == "__main__":
    unittest.main()
