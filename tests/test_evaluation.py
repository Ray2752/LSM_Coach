"""Pruebas de las reglas de evaluación (no necesitan cámara ni muñequera).

    python -m unittest discover -s tests -t .
"""
import unittest

from evaluation import CONFIG, ORIENT, evaluate
from tolerance_calculator import _range, mirror_features, one_sided, rotate_upright

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


class MirrorTests(unittest.TestCase):
    def test_refleja_solo_x_y_dos_veces_vuelve_igual(self):
        f = [float(i) for i in range(63)]
        m = mirror_features(f)
        self.assertEqual(m[0:3], [-0.0, 1.0, 2.0])
        self.assertEqual(m[3:6], [-3.0, 4.0, 5.0])
        self.assertEqual(mirror_features(m), f)


class OneSidedTests(unittest.TestCase):
    def test_dedo_cerrado_no_puede_estar_demasiado_cerrado(self):
        self.assertEqual(one_sided({"min": 40, "max": 110, "promedio": 60})["min"], 0.0)

    def test_dedo_extendido_no_puede_estar_demasiado_extendido(self):
        self.assertEqual(one_sided({"min": 150, "max": 175, "promedio": 165})["max"], 180.0)

    def test_dedo_curvado_conserva_ambos_limites(self):
        self.assertEqual(one_sided({"min": 100, "max": 160, "promedio": 125}),
                         {"min": 100, "max": 160, "promedio": 125})


class RotateUprightTests(unittest.TestCase):
    def test_endereza_una_mano_inclinada(self):
        import math
        hand = [0.0] * 63
        hand[9 * 3], hand[9 * 3 + 1] = 1.0, 0.0      # base del dedo medio a la derecha
        hand[8 * 3], hand[8 * 3 + 1] = 2.0, 0.0      # punta del índice, más a la derecha
        out = rotate_upright([hand])[0]
        self.assertAlmostEqual(out[9 * 3], 0.0)
        self.assertAlmostEqual(out[9 * 3 + 1], -1.0)  # ahora apunta hacia arriba (-y)
        self.assertAlmostEqual(out[8 * 3 + 1], -2.0)
        self.assertTrue(math.isclose(sum(v * v for v in out), sum(v * v for v in hand)))

    def test_mano_ya_derecha_no_cambia(self):
        hand = [0.0] * 63
        hand[9 * 3 + 1], hand[4 * 3] = -1.0, 0.5
        out = rotate_upright([hand])[0]
        for a, b in zip(out, hand):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
