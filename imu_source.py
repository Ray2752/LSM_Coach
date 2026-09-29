"""Fuentes de orientación para LSM Coach.

Ambas clases exponen lo mismo:
    .latest    -> {"roll": float, "pitch": float, "yaw": float}
    .start() / .stop()
    .vibrate(ms) -> hace vibrar la muñequera (el simulado solo lo imprime)

MockIMU  : simulado, se mueve con el teclado (para probar sin hardware).
BLEIMU   : lee el Nano 33 BLE Sense Rev2 por BLE (requiere: pip install bleak).
"""
import asyncio
import struct
import threading

DEVICE_NAME = "LSM-Wrist"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # orientación (notify)
VIB_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"   # vibración (write)


class MockIMU:
    STEP = 5.0

    def __init__(self):
        self.latest = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

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


class BLEIMU:
    def __init__(self):
        self.latest = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.connected = False
        self._stop = threading.Event()
        self._thread = None
        self._loop = None
        self._client = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

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
        roll, pitch, yaw = struct.unpack("<3f", data)
        self.latest.update(roll=roll, pitch=pitch, yaw=yaw)

    async def _main(self):
        from bleak import BleakScanner, BleakClient
        self._loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
            if device is None:
                continue
            try:
                async with BleakClient(device) as client:
                    self._client = client
                    self.connected = True
                    await client.start_notify(CHAR_UUID, self._on_data)
                    while client.is_connected and not self._stop.is_set():
                        await asyncio.sleep(0.2)
            except Exception as e:
                print("BLE:", e)
            finally:
                self._client = None
                self.connected = False