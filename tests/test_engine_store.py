"""Pruebas de la política de avisos y del almacén local (sin cámara, sin internet).

    python -m unittest discover -s tests -t .
"""
import os
import tempfile
import unittest

from coach_engine import MIN_GAP_S, PERSIST_S, REPEAT_S, AlertPolicy
from evaluation import CONFIG, shape_issue
from store import Store


class AlertPolicyTests(unittest.TestCase):
    def test_no_avisa_si_el_error_no_persiste(self):
        p = AlertPolicy()
        self.assertFalse(p.update("Extiende el índice", 100.0))
        self.assertFalse(p.update("Extiende el índice", 100.0 + PERSIST_S / 2))

    def test_avisa_cuando_persiste(self):
        p = AlertPolicy()
        p.update("Extiende el índice", 100.0)
        self.assertTrue(p.update("Extiende el índice", 100.0 + PERSIST_S + 0.01))

    def test_no_repite_el_mismo_error_antes_de_tiempo(self):
        p = AlertPolicy()
        p.update("x", 100.0)
        self.assertTrue(p.update("x", 101.0))
        self.assertFalse(p.update("x", 101.0 + MIN_GAP_S + 0.1))
        self.assertTrue(p.update("x", 101.0 + REPEAT_S))

    def test_corregir_rearma_el_aviso(self):
        p = AlertPolicy()
        p.update("x", 100.0)
        self.assertTrue(p.update("x", 101.0))
        self.assertFalse(p.update(None, 102.0))
        p.update("x", 103.0)
        self.assertTrue(p.update("x", 103.0 + PERSIST_S + 0.01))


class ShapeIssueTests(unittest.TestCase):
    def test_forma_reconocida(self):
        self.assertIsNone(shape_issue({"A": 0.8, "Y": 0.2}, "A"))

    def test_probabilidad_repartida_entre_letras_parecidas_no_rechaza(self):
        self.assertIsNone(shape_issue({"A": 0.35, "S": 0.3, "T": 0.2, "E": 0.15}, "A"))

    def test_se_parece_a_otra_letra(self):
        issue = shape_issue({"C": 0.7, "Y": 0.3}, "Y")
        self.assertEqual(issue.parameter, CONFIG)
        self.assertEqual(issue.action, "Ajusta la forma: se parece más a una C")


    def test_error_tipico_de_la_misma_letra(self):
        issue = shape_issue({"B": 0.2, "B_mal": 0.7, "C": 0.1}, "B")
        self.assertEqual(issue.action, "Revisa la B: dedos separados o semiflexionados")

    def test_error_de_otra_letra_se_reporta_como_esa_letra(self):
        issue = shape_issue({"Y": 0.1, "C": 0.2, "C_mal": 0.7}, "Y")
        self.assertEqual(issue.action, "Ajusta la forma: se parece más a una C")

    def test_letra_que_el_modelo_no_conoce(self):
        self.assertIsNone(shape_issue({"A": 1.0}, "J"))

    def test_misma_forma_otra_orientacion_no_rechaza(self):
        # la G es una L horizontal: la orientación la juzga la muñequera, no la forma
        self.assertIsNone(shape_issue({"G": 0.6, "L": 0.4}, "L"))
        self.assertIsNotNone(shape_issue({"D": 0.6, "L": 0.4}, "L"))

    def test_c_cerrada_leida_como_o_la_decide_el_hueco(self):
        self.assertIsNone(shape_issue({"O": 0.6, "C": 0.4}, "C"))
        self.assertIsNotNone(shape_issue({"C": 0.6, "O": 0.4}, "O"))  # al revés no

    def test_umbral_de_error(self):
        self.assertIsNotNone(shape_issue({"B": 0.55, "B_mal": 0.45}, "B"))
        self.assertIsNone(shape_issue({"B": 0.65, "B_mal": 0.35}, "B"))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # sin .env y sin variables: las pruebas nunca tocan la nube real
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_KEY", None)
        self.store = Store(os.path.join(self.tmp.name, "t.db"), env_file=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_guarda_local_sin_nube(self):
        self.store.add_sample("ray", "A", {"pulgar": 170.0}, [0.1] * 63, None)
        self.store.add_sample("ray", "A", {"pulgar": 90.0}, [0.1] * 63, {"roll": 1, "pitch": 2, "yaw": 3},
                              is_error=True)
        self.store.add_attempt("ray", "A", 3.2, ["Configuración"])
        self.assertEqual(self.store.count_samples("A", False), 1)
        self.assertEqual(self.store.count_samples("A", True), 1)
        self.assertEqual(self.store.status(), {"cloud": False, "pending": 3, "error": None})
        self.assertEqual(self.store.sync_once(), 0)  # sin nube no intenta subir

    def test_sync_sube_y_marca(self):
        self.store.url, self.store.key = "https://x.supabase.co", "sb_secret_test"
        sent = []
        self.store._post = lambda table, rows: sent.append((table, rows))
        self.store.add_sample("ray", "B", {"pulgar": 120.0}, None, {"roll": 1.0, "pitch": 2.0, "yaw": 3.0})
        self.store.add_attempt("ray", "B", 2.0, [])
        self.assertEqual(self.store.sync_once(), 2)
        self.assertEqual(self.store.pending(), 0)
        table, rows = sent[0]
        self.assertEqual(table, "samples")
        self.assertIs(rows[0]["is_error"], False)
        self.assertEqual(rows[0]["imu"], {"roll": 1.0, "pitch": 2.0, "yaw": 3.0})  # JSON, no texto
        self.assertEqual(self.store.sync_once(), 0)  # nada pendiente

    def test_sync_fallido_deja_pendiente(self):
        self.store.url, self.store.key = "https://x.supabase.co", "k"
        def fail(table, rows):
            raise OSError("sin red")
        self.store._post = fail
        self.store.add_attempt("ray", "C", 1.0, [])
        self.assertEqual(self.store.sync_once(), 0)
        self.assertEqual(self.store.pending(), 1)
        self.assertIn("sin conexión", self.store.last_error)


if __name__ == "__main__":
    unittest.main()
