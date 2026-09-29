"""Pruebas de las reglas de evaluación (no necesitan cámara ni muñequera).

    python -m unittest discover -s tests -t .
"""
import unittest

from evaluation import CONFIG, ORIENT, evaluate
from tolerance_calculator import _range

TOL = {f: {"min": 160.0, "max": 180.0} for f in ("pulgar", "indice", "medio", "anular", "menique")}
TOL_ORIENT = {**TOL, "orientacion": {"roll": {"min": -10, "max": 10},
                                     "pitch": {"min": -10, "max": 10}}}
FLAT = {f: 170.0 for f in TOL}


class EvaluateTests(unittest.TestCase):
    def test_sena_correcta(self):
        r = evaluate(FLAT, None, TOL)
        self.assertTrue(r.ok)
        self.assertEqual(r.issues, [])

    def test_dedo_muy_cerrado_pide_extender(self):
        r = evaluate({**FLAT, "indice": 90.0}, None, TOL)
        self.assertFalse(r.ok)
        self.assertEqual(r.issues[0].action, "Extiende el índice")
        self.assertEqual(r.failed_parameters, [CONFIG])

    def test_dedo_muy_extendido_pide_flexionar(self):
        tol = {**TOL, "menique": {"min": 10.0, "max": 60.0}}
        r = evaluate({**FLAT, "menique": 175.0}, None, tol)
        self.assertEqual(r.issues[0].action, "Flexiona el meñique")

    def test_orientacion_fuera_de_rango(self):
        r = evaluate(FLAT, {"roll": 0.0, "pitch": 40.0, "yaw": 0.0}, TOL_ORIENT)
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_parameters, [ORIENT])
        self.assertEqual(r.issues[0].where, "pitch")

    def test_dedos_y_orientacion_fallan_a_la_vez(self):
        r = evaluate({**FLAT, "medio": 90.0}, {"roll": 50.0, "pitch": 0.0, "yaw": 0.0}, TOL_ORIENT)
        self.assertEqual(r.failed_parameters, [CONFIG, ORIENT])

    def test_sin_muneca_no_se_da_por_correcta(self):
        # la seña exige orientación: sin lectura del dispositivo no puede aprobarse
        r = evaluate(FLAT, None, TOL_ORIENT, require_imu=True)
        self.assertFalse(r.ok)
        self.assertEqual(r.issues[0].action, "Conecta la muñequera")

    def test_sin_imu_permitido_ignora_orientacion(self):
        self.assertTrue(evaluate(FLAT, None, TOL_ORIENT, require_imu=False).ok)

    def test_orientacion_correcta(self):
        r = evaluate(FLAT, {"roll": 3.0, "pitch": -4.0, "yaw": 99.0}, TOL_ORIENT)
        self.assertTrue(r.ok)  # el yaw no se evalúa


class RangeTests(unittest.TestCase):
    def test_holgura_minima_evita_rangos_diminutos(self):
        r = _range([176.0, 176.1, 176.0], margin=2.0, floor=6.0)
        self.assertGreaterEqual(r["max"] - r["min"], 12.0 - 0.2)

    def test_limites_fisicos(self):
        r = _range([5.0, 175.0], margin=2.0, floor=6.0, lo_limit=0.0, hi_limit=180.0)
        self.assertEqual((r["min"], r["max"]), (0.0, 180.0))


if __name__ == "__main__":
    unittest.main()
