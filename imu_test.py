"""Prueba rápida de la muñequera: muestra roll/pitch/yaw en vivo por BLE.

    python imu_test.py            # lee 30 s
    python imu_test.py --vibrar   # además hace vibrar la muñequera al conectar

Sirve para comprobar el sensor real (y para grabar la evidencia del 2do corte).
"""
import argparse
import time

from imu_source import BLEIMU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=float, default=30)
    ap.add_argument("--vibrar", action="store_true", help="vibra 300 ms al conectar")
    args = ap.parse_args()

    imu = BLEIMU()
    imu.start()
    print("Buscando 'LSM-Wrist' por BLE (enciende la Nano)...")
    start, was_connected, packets = time.time(), False, 0
    last = None
    try:
        while time.time() - start < args.segundos:
            if imu.connected and not was_connected:
                print("Conectado.")
                if args.vibrar:
                    imu.vibrate(300)
            was_connected = imu.connected
            if imu.connected:
                v = dict(imu.latest)
                packets += v != last
                last = v
                print(f"roll={v['roll']:7.1f}  pitch={v['pitch']:7.1f}  yaw={v['yaw']:7.1f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    imu.stop()
    if not was_connected:
        print("No se conectó. Revisa que el sketch esté cargado y que la Nano tenga el LED parpadeando.")
    else:
        print(f"Lecturas distintas recibidas: {packets}")


if __name__ == "__main__":
    main()
