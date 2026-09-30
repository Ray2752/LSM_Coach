"""Seguimiento de la persona con la cámara motorizada (UNO Q + 2 servos SG90).

La visión detecta el rostro; aquí se decide hacia dónde girar la cámara para mantenerlo
centrado y se manda "aim(pan, tilt)" a la MCU de la UNO Q, que mueve los servos
(uno_q/pan_tilt/sketch.ino). La orden llega a la MCU por el arduino-router: un socket Unix
con MessagePack-RPC en /var/run/arduino-router.sock, así que no depende del Python de App Lab.

Dos formas de llegar al router:
  - RouterClient: la visión corre en la propia UNO Q.
  - NetClient: la visión corre en otra placa (Raspberry Pi 5). Manda las mismas órdenes por
    UDP a la UNO Q, donde `pan_tilt_server.py` (uno_q/servos.sh) las pasa al router. La UNO Q
    se anuncia por difusión cada segundo, así que la Pi la encuentra sola (`discover`).

Reglas del seguimiento (Tracker):
  - control proporcional con zona muerta: si el rostro está cerca del centro, no se mueve
  - pasos limitados: la cámara gira despacio para que la imagen no brinque
  - NO se mueve mientras se está grabando/evaluando un trazo (moverla alteraría la trayectoria)
  - sin rostro un rato, se queda quieta (no "busca")

Pruebas del hardware, en la UNO Q:
    python pan_tilt.py --centrar
    python pan_tilt.py --barrido       # recorre pan y tilt para ver que se mueven
    python pan_tilt.py --apuntar 60 100
Desde la Pi (con uno_q/servos.sh corriendo en la UNO Q):
    python pan_tilt.py --red --barrido            # encuentra la UNO Q sola
    python pan_tilt.py --red 172.20.10.9 --barrido
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

SOCKET_PATH = "/var/run/arduino-router.sock"
NET_PORT = 8765        # UDP: órdenes de la Pi a la UNO Q (pan_tilt_server.py)
DISCOVER_PORT = 8766   # UDP: la UNO Q anuncia "aquí estoy" por difusión cada segundo

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


def discover(timeout=5.0, port=DISCOVER_PORT):
    """IP de la UNO Q que corre pan_tilt_server.py, escuchando sus anuncios; None si no hay."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
        deadline = time.time() + timeout
        while time.time() < deadline:
            sock.settimeout(max(0.1, deadline - time.time()))
            try:
                data, addr = sock.recvfrom(256)
                if json.loads(data.decode()).get("lsm") == "pan_tilt":
                    return addr[0]
            except (OSError, ValueError):
                continue
    except OSError as e:
        print(f"No se pudo escuchar los anuncios de la UNO Q ({e}).", file=sys.stderr)
    finally:
        sock.close()
    return None


class NetClient:
    """Mismas órdenes que RouterClient (aim/center/status) pero por UDP a la UNO Q, para cuando
    la visión corre en otra placa. Cada datagrama es JSON: {"m": "aim", "a": [pan, tilt]};
    con "id" se espera respuesta {"id": ..., "r": resultado}."""

    def __init__(self, host="auto", port=NET_PORT):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._id = 0
        self._lock = threading.Lock()
        self.connected = False

    def connect(self, timeout=5.0):
        if self.host == "auto":
            self.host = discover(timeout)
            if not self.host:
                print("No se encontró la UNO Q en la red (¿corre uno_q/servos.sh?); la cámara no se moverá.",
                      file=sys.stderr)
                return False
        self.connected = True
        status = self.call("status")
        if status is None:
            self.connected = False
            print(f"La UNO Q en {self.host} no responde a los servos; la cámara no se moverá.", file=sys.stderr)
            return False
        print(f"Cámara motorizada: UNO Q en {self.host} (servos en {status}).")
        return True

    def notify(self, method, *args):
        if not self.connected:
            return False
        try:
            self.sock.sendto(json.dumps({"m": method, "a": list(args)}).encode(), (self.host, self.port))
            return True
        except OSError:
            return False

    def call(self, method, *args, timeout=2.0):
        if not self.connected:
            return None
        with self._lock:
            self._id += 1
            msgid = self._id
            try:
                self.sock.sendto(json.dumps({"m": method, "a": list(args), "id": msgid}).encode(),
                                 (self.host, self.port))
                deadline = time.time() + timeout
                while time.time() < deadline:
                    self.sock.settimeout(max(0.05, deadline - time.time()))
                    data, _ = self.sock.recvfrom(512)
                    reply = json.loads(data.decode())
                    if reply.get("id") == msgid:
                        return reply.get("r")
            except (OSError, ValueError):
                pass
        print(f"UNO Q ({self.host}): {method} -> sin respuesta", file=sys.stderr)
        return None

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


def make_client(remote="auto"):
    """Cliente hacia los servos: el router local si esta placa es la UNO Q, si no la UNO Q por
    red (remote = "auto" para encontrarla sola, o su IP). None si no hay forma de llegar."""
    if remote in (None, "auto", "local") and os.path.exists(SOCKET_PATH):
        try:
            client = RouterClient()
        except ImportError:
            print("Falta msgpack (pip install msgpack): sin seguimiento de cámara.", file=sys.stderr)
            return None
    elif remote == "local":
        print("No hay arduino-router en esta placa; la cámara no se moverá.", file=sys.stderr)
        return None
    else:
        client = NetClient("auto" if remote is None else remote)
    return client if client.connect() else None


def make_tracker(remote="auto"):
    """Tracker conectado a los servos (ver make_client), o None si no hay servos."""
    client = make_client(remote)
    if client is None:
        return None
    tracker = Tracker(lambda p, t: client.notify("aim", p, t))
    tracker.client = client
    tracker.center()
    return tracker


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--centrar", action="store_true")
    ap.add_argument("--barrido", action="store_true")
    ap.add_argument("--apuntar", nargs=2, type=int, metavar=("PAN", "TILT"))
    ap.add_argument("--red", nargs="?", const="auto", metavar="IP",
                    help="hablar con la UNO Q por red (desde la Pi); sin IP la busca sola")
    args = ap.parse_args()
    client = make_client(args.red if args.red else "local")
    if client is None:
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
