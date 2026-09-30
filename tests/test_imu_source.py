"""Pruebas de las estadísticas del enlace con la muñequera (sin Bluetooth).

    python -m unittest discover -s tests -t .
"""
import struct
import unittest

from imu_source import STALE_S, BLEIMU, MockIMU


class LinkStatsTests(unittest.TestCase):
    def test_sin_conexion_no_hay_lecturas(self):
        imu = BLEIMU()
        self.assertEqual(imu.stats(now=100.0),
                         {"connected": False, "address": None, "hz": 0.0, "age_ms": None, "stale": False})

    def test_una_notificacion_actualiza_orientacion_y_edad(self):
        imu = BLEIMU()
        imu.connected = True
        imu._got(10.0, -5.0, 90.0, now=100.0)
        self.assertEqual(imu.latest, {"roll": 10.0, "pitch": -5.0, "yaw": 90.0})
        st = imu.stats(now=100.25)
        self.assertTrue(st["connected"])
        self.assertEqual(st["age_ms"], 250)
        self.assertFalse(st["stale"])

    def test_tasa_de_lecturas_por_segundo(self):
        imu = BLEIMU()
        t = 100.0
        for i in range(41):  # 20 Hz durante 2 s
            imu._got(0.0, 0.0, 0.0, now=t + i * 0.05)
        self.assertAlmostEqual(imu.stats(now=t + 2.0)["hz"], 20.0, delta=1.0)

    def test_sin_lecturas_un_rato_se_marca_sin_datos(self):
        imu = BLEIMU()
        imu.connected = True
        for i in range(25):
            imu._got(1.0, 2.0, 3.0, now=100.0 + i * 0.05)
        st = imu.stats(now=100.0 + 24 * 0.05 + STALE_S + 0.1)
        self.assertTrue(st["stale"])
        self.assertEqual(st["hz"], 0.0)  # la tasa vieja no se sigue mostrando

    def test_desconectar_reinicia_el_enlace(self):
        imu = BLEIMU()
        imu.connected, imu.address = True, "AA:BB"
        imu._got(1.0, 2.0, 3.0, now=100.0)
        imu._reset_link()
        st = imu.stats(now=101.0)
        self.assertFalse(st["connected"])
        self.assertEqual(st["address"], "AA:BB")  # la dirección se conserva para reconectar
        self.assertIsNone(st["age_ms"])

    def test_paquete_ble_se_desempaca(self):
        imu = BLEIMU()
        imu._on_data(None, bytearray(struct.pack("<3f", 12.5, -30.0, 181.0)))
        self.assertEqual(imu.latest, {"roll": 12.5, "pitch": -30.0, "yaw": 181.0})
        self.assertIsNotNone(imu.last_t)

    def test_simulada_siempre_conectada(self):
        imu = MockIMU()
        self.assertTrue(imu.connected)
        imu.handle_key("d")
        self.assertEqual(imu.latest["roll"], MockIMU.STEP)
        self.assertIsNotNone(imu.stats()["age_ms"])


if __name__ == "__main__":
    unittest.main()
