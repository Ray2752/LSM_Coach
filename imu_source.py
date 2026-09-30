"""Fuentes de orientación para LSM Coach.

Ambas clases exponen lo mismo:
    .latest    -> {"roll": float, "pitch": float, "yaw": float}
    .connected -> True cuando hay lecturas de verdad (el simulado siempre)
    .stats()   -> estado del enlace para la interfaz: conectada, dirección, Hz, edad de la lectura
    .start() / .stop()
    .vibrate(ms) -> hace vibrar la muñequera (el simulado solo lo imprime)

MockIMU  : simulado, se mueve con el teclado (para probar sin hardware).
BLEIMU   : lee el Nano 33 BLE Sense Rev2 por BLE (requiere: pip install bleak).
"""
import asyncio
import os
import struct
import threading
import time

DEVICE_NAME = "LSM-Wrist"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # orientación (notify)
VIB_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"   # vibración (write)
ADDR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lsm_wrist_addr")
STALE_S = 1.5  # sin lecturas este tiempo: el enlace se muestra como "sin datos" aunque siga conectado


def save_address(addr):
    try:
        with open(ADDR_FILE, "w", encoding="utf-8") as f:
            f.write(addr.strip())
    except OSError:
        pass


def load_address():
    try:
        with open(ADDR_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


class _IMU:
    """Lo común a las dos fuentes: la última orientación y las estadísticas del enlace,
    que la interfaz muestra en el recuadro de la muñequera."""

    def __init__(self):
        self.latest = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.connected = False
        self.address = None   # dirección BLE de la Nano (None si no se ha conectado)
        self.hz = 0.0         # lecturas por segundo que están llegando
        self.last_t = None    # cuándo llegó la última lectura (time.time())
        self._count, self._win_t = 0, None

    def _got(self, roll, pitch, yaw, now=None):
        """Registra una lectura nueva y actualiza la tasa (ventana de 1 s)."""
        now = time.time() if now is None else now
        self.latest.update(roll=roll, pitch=pitch, yaw=yaw)
        self.last_t = now
        if self._win_t is None:
            self._win_t, self._count = now, 0
        self._count += 1
        if now - self._win_t >= 1.0:
            self.hz = self._count / (now - self._win_t)
            self._win_t, self._count = now, 0

    def _reset_link(self):
        self.connected = False
        self.hz, self.last_t = 0.0, None
        self._win_t, self._count = None, 0

    def stats(self, now=None):
        """Estado del enlace: conectada, dirección, Hz y hace cuánto llegó la última lectura (ms)."""
        now = time.time() if now is None else now
        age = None if self.last_t is None else round((now - self.last_t) * 1000)
        stale = age is not None and age > STALE_S * 1000
        return {"connected": self.connected, "address": self.address,
                "hz": 0.0 if stale else round(self.hz, 1), "age_ms": age, "stale": stale}


class MockIMU(_IMU):
    STEP = 5.0

    def __init__(self):
        super().__init__()
        self.connected = True  # simulada: siempre "hay lectura"

    def start(self):
        pass

    def stop(self):
        pass

    def vibrate(self, ms: int = 200):
        print(f"[VIBRA {ms} ms]")

    def handle_key(self, ch: str):
        """a/d = roll -/+, w/s = pitch +/-, j/l = yaw -/+, r = reset."""
        moves = {
            "a": ("roll", -self.STEP), "d": ("roll", self.STEP),
            "w": ("pitch", self.STEP), "s": ("pitch", -self.STEP),
            "j": ("yaw", -self.STEP), "l": ("yaw", self.STEP),
        }
        if ch in moves:
            axis, delta = moves[ch]
            self.latest[axis] += delta
        elif ch == "r":
            self.latest.update(roll=0.0, pitch=0.0, yaw=0.0)
        self._got(**self.latest)


class BLEIMU(_IMU):
    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._thread = None
        self._loop = None
        self._client = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Pide desconectar y espera a que ocurra. Importante en Linux (UNO Q): si el programa
        muere sin desconectar, bluez deja la Nano "conectada" sin que nadie la escuche, y como
        conectada no se anuncia, el siguiente arranque no la encuentra."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def vibrate(self, ms: int = 200):
        """Manda la orden al Nano (1 byte = duración en unidades de 10 ms)."""
        if not (self._client and self._loop and self.connected):
            return
        units = max(1, min(255, ms // 10))
        try:
            asyncio.run_coroutine_threadsafe(
                self._client.write_gatt_char(VIB_UUID, bytes([units]), response=False),
                self._loop,
            )
        except Exception as e:
            print("BLE vibrar:", e)

    def _run(self):
        asyncio.run(self._main())

    def _on_data(self, _, data: bytearray):
        self._got(*struct.unpack("<3f", data))

    async def _main(self):
        from bleak import BleakScanner, BleakClient
        self._loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:  # si el Bluetooth falla al buscar, se reintenta en vez de matar el hilo
                device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=8)
            except Exception as e:
                print("BLE (búsqueda):", e)
                await asyncio.sleep(2)
                continue
            if device is None:
                # No se anuncia: puede que bluez (Linux) la tenga conectada de una sesión anterior.
                # Conectar por la dirección guardada funciona aunque ya esté "tomada".
                device = load_address()
                if device is None:
                    continue
                print(f"LSM-Wrist no se anuncia; intento por dirección {device}...")
            try:
                async with BleakClient(device) as client:
                    self._client = client
                    self.address = client.address
                    self.connected = True
                    save_address(client.address)
                    await client.start_notify(CHAR_UUID, self._on_data)
                    while client.is_connected and not self._stop.is_set():
                        await asyncio.sleep(0.2)
                    try:  # avisar a la Nano para que vuelva a anunciarse de inmediato
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass
            except Exception as e:
                print("BLE:", e)
                await asyncio.sleep(1)
            finally:
                self._client = None
                self._reset_link()