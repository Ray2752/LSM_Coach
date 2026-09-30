"""Seguimiento de la persona con la cámara motorizada (UNO Q + 2 servos SG90).

La visión (Linux de la UNO Q) detecta el rostro; aquí se decide hacia dónde girar la cámara
para mantenerlo centrado y se manda "aim(pan, tilt)" a la MCU, que mueve los servos
(uno_q/pan_tilt/sketch.ino). La orden va por el arduino-router: un socket Unix con
MessagePack-RPC en /var/run/arduino-router.sock, así que no depende del Python de App Lab.

Reglas del seguimiento (Tracker):
  - control proporcional con zona muerta: si el rostro está cerca del centro, no se mueve
  - pasos limitados: la cámara gira despacio para que la imagen no brinque
  - NO se mueve mientras se está grabando/evaluando un trazo (moverla alteraría la trayectoria)
  - sin rostro un rato, se queda quieta (no "busca")

Pruebas del hardware, en la UNO Q:
    python pan_tilt.py --centrar
    python pan_tilt.py --barrido       # recorre pan y tilt para ver que se mueven
    python pan_tilt.py --apuntar 60 100
"""
import argparse
import socket
import sys
import threading
import time

SOCKET_PATH = "/var/run/arduino-router.sock"

# --- Ajustes del seguimiento -----------------------------------------------------------
PAN_CENTER, TILT_CENTER = 90, 90
PAN_RANGE, TILT_RANGE = (15, 165), (40, 140)
DEADBAND = 0.08     # fracción de la imagen: dentro de esta zona alrededor del centro no se mueve
GAIN = 40.0         # grados por unidad de error (error = desplazamiento del rostro, -0.5..0.5)
MAX_STEP = 4.0      # grados como máximo por actualización
UPDATE_S = 0.2      # actualizaciones por segundo (5 Hz)
FACE_TIMEOUT_S = 1.0  # sin rostro más reciente que esto, no se mueve
# Signo de cada eje: depende de cómo esté montado el servo y de que la imagen va en espejo.
# Si la cámara "huye" de la persona en vez de seguirla, cambiar el signo de ese eje.
PAN_SIGN, TILT_SIGN = -1, 1
# ---------------------------------------------------------------------------------------


class RouterClient:
    """Cliente mínimo de MessagePack-RPC para el arduino-router (Linux <-> MCU)."""

    def __init__(self, path=SOCKET_PATH):
        import msgpack  # solo hace falta en la UNO Q: pip install msgpack
        self.msgpack = msgpack
        self.path = path
        self.sock = None
        self._id = 0
        self._pending = {}
        self._lock = threading.Lock()
        self.connected = False

    def connect(self, timeout=3.0):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(self.path)
            self.sock.settimeout(None)
            self.connected = True
            threading.Thread(target=self._receive, daemon=True).start()
            return True
        except OSError as e:
            print(f"Router de la UNO Q no disponible ({e}); la cámara no se moverá.", file=sys.stderr)
            self.connected = False
            return False

    def _receive(self):
        unpacker = self.msgpack.Unpacker(raw=False)
        while self.connected:
            try:
                data = self.sock.recv(4096)
            except OSError:
                break
            if not data:
                break
            unpacker.feed(data)
            for msg in unpacker:
                if isinstance(msg, (list, tuple)) and len(msg) == 4 and msg[0] == 1:  # RESPONSE
                    ev = self._pending.get(msg[1])
                    if ev:
                        ev["error"], ev["result"] = msg[2], msg[3]
                        ev["done"].set()
        self.connected = False

    def notify(self, method, *args):
        """Orden sin respuesta (lo más rápido: ~1 ms)."""
        if not self.connected:
            return False
        try:
            with self._lock:
                self.sock.sendall(self.msgpack.packb([2, method, list(args)], use_bin_type=True))
            return True
        except OSError:
            self.connected = False
            return False

    def call(self, method, *args, timeout=2.0):
        """Llamada con respuesta. Devuelve el resultado o None si falla."""
        if not self.connected:
            return None
        with self._lock:
            self._id += 1
            msgid = self._id
            ev = {"done": threading.Event(), "error": None, "result": None}
            self._pending[msgid] = ev
            try:
                self.sock.sendall(self.msgpack.packb([0, msgid, method, list(args)], use_bin_type=True))
            except OSError:
                self.connected = False
                return None
        ok = ev["done"].wait(timeout)
        self._pending.pop(msgid, None)
        if not ok or ev["error"]:
            print(f"Router: {method} -> {'sin respuesta' if not ok else ev['error']}", file=sys.stderr)
            return None
        return ev["result"]

    def close(self):
        self.connected = False
        try:
            self.sock.close()
        except Exception:
            pass


class Tracker:
    """Decide pan/tilt a partir de la posición del rostro. `send(pan, tilt)` se llama solo
    cuando hay que mover; se inyecta para poder probarlo sin hardware."""

    def __init__(self, send):
        self.send = send
        self.pan, self.tilt = float(PAN_CENTER), float(TILT_CENTER)
        self._last_update = 0.0
        self.moves = 0

    def update(self, face, now, busy=False, face_t=None):
        """face: (cx, cy, w, h) del rostro en la imagen (0-1) o None. busy: no mover
        (se está grabando o evaluando un trazo). Devuelve True si mandó un movimiento."""
        if busy or face is None or now - self._last_update < UPDATE_S:
            return False
        if face_t is not None and now - face_t > FACE_TIMEOUT_S:
            return False
        self._last_update = now
        ex, ey = face[0] - 0.5, face[1] - 0.5  # + = rostro a la derecha / abajo de la imagen
        dpan = 0.0 if abs(ex) < DEADBAND else max(-MAX_STEP, min(MAX_STEP, PAN_SIGN * GAIN * ex))
        dtilt = 0.0 if abs(ey) < DEADBAND else max(-MAX_STEP, min(MAX_STEP, TILT_SIGN * GAIN * ey))
        if dpan == 0.0 and dtilt == 0.0:
            return False
        self.pan = max(PAN_RANGE[0], min(PAN_RANGE[1], self.pan + dpan))
        self.tilt = max(TILT_RANGE[0], min(TILT_RANGE[1], self.tilt + dtilt))
        self.moves += 1
        self.send(int(round(self.pan)), int(round(self.tilt)))
        return True

    def center(self):
        self.pan, self.tilt = float(PAN_CENTER), float(TILT_CENTER)
        self.send(PAN_CENTER, TILT_CENTER)


def make_tracker():
    """Tracker conectado al router de la UNO Q, o None si no hay router (Mac, sin servos)."""
    try:
        client = RouterClient()
    except ImportError:
        print("Falta msgpack (pip install msgpack): sin seguimiento de cámara.", file=sys.stderr)
        return None
    if not client.connect():
        return None
    tracker = Tracker(lambda p, t: client.notify("aim", p, t))
    tracker.client = client
    tracker.center()
    print("Cámara motorizada: conectada al router de la UNO Q.")
    return tracker


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--centrar", action="store_true")
    ap.add_argument("--barrido", action="store_true")
    ap.add_argument("--apuntar", nargs=2, type=int, metavar=("PAN", "TILT"))
    args = ap.parse_args()
    client = RouterClient()
    if not client.connect():
        sys.exit(1)
    print("estado:", client.call("status"))
    if args.apuntar:
        client.notify("aim", *args.apuntar)
    elif args.barrido:
        for p, t in ((40, 90), (140, 90), (90, 60), (90, 120), (90, 90)):
            client.notify("aim", p, t)
            time.sleep(1.2)
            print("->", p, t, "estado:", client.call("status"))
    else:
        client.call("center")
    time.sleep(0.5)
    print("estado:", client.call("status"))
    client.close()


if __name__ == "__main__":
    main()
