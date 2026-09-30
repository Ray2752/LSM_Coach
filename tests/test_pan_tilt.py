"""Pruebas del seguimiento de cámara (sin servos ni router).

    python -m unittest discover -s tests -t .
"""
import unittest

from pan_tilt import DEADBAND, MAX_STEP, PAN_CENTER, PAN_RANGE, PAN_SIGN, TILT_CENTER, UPDATE_S, Tracker


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.tr = Tracker(lambda p, t: self.sent.append((p, t)))

    def test_rostro_centrado_no_mueve(self):
        self.assertFalse(self.tr.update((0.5, 0.5, .2, .25), now=10.0))
        self.assertFalse(self.tr.update((0.5 + DEADBAND * .9, 0.5, .2, .25), now=11.0))
        self.assertEqual(self.sent, [])

    def test_rostro_a_un_lado_gira_en_el_sentido_configurado(self):
        self.assertTrue(self.tr.update((0.9, 0.5, .2, .25), now=10.0))
        pan, tilt = self.sent[-1]
        self.assertEqual(tilt, TILT_CENTER)
        self.assertEqual(pan - PAN_CENTER, PAN_SIGN * MAX_STEP)  # paso limitado, con el signo del eje

    def test_no_mueve_mientras_se_evalua_un_trazo(self):
        self.assertFalse(self.tr.update((0.9, 0.5, .2, .25), now=10.0, busy=True))
        self.assertEqual(self.sent, [])

    def test_respeta_la_frecuencia_de_actualizacion(self):
        self.assertTrue(self.tr.update((0.9, 0.5, .2, .25), now=10.0))
        self.assertFalse(self.tr.update((0.9, 0.5, .2, .25), now=10.0 + UPDATE_S / 2))
        self.assertTrue(self.tr.update((0.9, 0.5, .2, .25), now=10.0 + UPDATE_S * 1.1))
        self.assertEqual(len(self.sent), 2)

    def test_sin_rostro_reciente_no_mueve(self):
        self.assertFalse(self.tr.update(None, now=10.0))
        self.assertFalse(self.tr.update((0.9, 0.5, .2, .25), now=10.0, face_t=5.0))  # rostro viejo

    def test_no_pasa_de_los_topes(self):
        t = 0.0
        for _ in range(100):
            t += UPDATE_S * 1.1
            # el lado del rostro que hace crecer el pan depende del signo del eje
            self.tr.update((1.0 if PAN_SIGN > 0 else 0.0, 0.5, .2, .25), now=t)
        self.assertEqual(self.sent[-1][0], PAN_RANGE[1])


if __name__ == "__main__":
    unittest.main()
