"""Pruebas del seguimiento de cámara (sin servos ni router).

    python -m unittest discover -s tests -t .
"""
import json
import socket
import threading
import unittest

import pan_tilt
import pan_tilt_server
from pan_tilt import DEADBAND, MAX_STEP, PAN_CENTER, PAN_RANGE, PAN_SIGN, TILT_CENTER, UPDATE_S, NetClient, Tracker


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


class TargetPointTests(unittest.TestCase):
    def test_mano_manda_y_el_tilt_reparte_con_el_rostro(self):
        p = pan_tilt.target_point((0.8, 0.7), 10.0, (0.5, 0.3), 10.0, now=10.1)
        self.assertEqual(p, (0.8, 0.5))

    def test_sin_rostro_se_sigue_la_mano(self):
        self.assertEqual(pan_tilt.target_point((0.8, 0.7), 10.0, None, 0.0, now=10.1), (0.8, 0.7))

    def test_sin_mano_reciente_se_sigue_el_rostro(self):
        self.assertEqual(pan_tilt.target_point((0.8, 0.7), 5.0, (0.5, 0.3), 10.0, now=10.1), (0.5, 0.3))

    def test_nada_reciente(self):
        self.assertIsNone(pan_tilt.target_point((0.8, 0.7), 5.0, (0.5, 0.3), 5.0, now=10.1))


class FakeRouter:
    """Hace de RouterClient: apunta los servos en memoria."""

    def __init__(self):
        self.pan, self.tilt, self.notified = 90, 90, []
        self.connected = True

    def call(self, method, *args):
        if method == "aim":
            self.pan, self.tilt = args
            return 1
        if method == "center":
            self.pan, self.tilt = 90, 90
            return 1
        return f"{self.pan},{self.tilt}"

    def notify(self, method, *args):
        self.notified.append((method, args))
        self.call(method, *args)
        return True


class NetBridgeTests(unittest.TestCase):
    """NetClient (Pi) <-> pan_tilt_server.handle (UNO Q) por UDP en localhost, sin hardware."""

    def setUp(self):
        self.router = FakeRouter()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.settimeout(2.0)
        self.port = self.srv.getsockname()[1]
        self.alive = True
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while self.alive:
            try:
                data, addr = self.srv.recvfrom(512)
            except OSError:
                continue
            reply = pan_tilt_server.handle(self.router, data)
            if reply is not None:
                self.srv.sendto(reply, addr)

    def tearDown(self):
        self.alive = False
        self.srv.close()

    def test_conecta_y_pregunta_el_estado(self):
        client = NetClient("127.0.0.1", self.port)
        self.assertTrue(client.connect())
        self.assertEqual(client.call("status"), "90,90")
        client.close()

    def test_aim_por_notify_llega_al_router(self):
        client = NetClient("127.0.0.1", self.port)
        self.assertTrue(client.connect())
        client.notify("aim", 60, 100)
        self.assertEqual(client.call("status"), "60,100")  # UDP conserva el orden en localhost
        self.assertIn(("aim", (60, 100)), self.router.notified)
        client.close()

    def test_ordenes_desconocidas_o_basura_se_ignoran(self):
        self.assertIsNone(pan_tilt_server.handle(self.router, b"no es json"))
        self.assertIsNone(pan_tilt_server.handle(self.router, json.dumps({"m": "reboot", "id": 1}).encode()))
        self.assertEqual(self.router.notified, [])

    def test_sin_uno_q_no_conecta(self):
        client = NetClient("127.0.0.1", self.port)
        self.alive = False
        self.srv.close()
        self.assertFalse(client.connect(timeout=0.3))
        self.assertFalse(client.connected)
        client.close()

    def test_discover_encuentra_el_anuncio(self):
        port = 18766
        found = {}

        def listen():
            found["ip"] = pan_tilt.discover(timeout=2.0, port=port)
        t = threading.Thread(target=listen)
        t.start()
        ann = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(10):
            ann.sendto(json.dumps({"lsm": "pan_tilt", "port": 1}).encode(), ("127.0.0.1", port))
            t.join(0.1)
            if not t.is_alive():
                break
        t.join(2.5)
        ann.close()
        self.assertEqual(found.get("ip"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
