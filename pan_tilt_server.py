"""Puente de red para la cámara motorizada. Corre en la UNO Q cuando la visión va en otra
placa (Raspberry Pi 5): recibe por UDP las órdenes aim/center/status (pan_tilt.NetClient) y
las pasa al arduino-router, es decir, a la MCU que mueve los servos (uno_q/pan_tilt/sketch.ino).
Se anuncia por difusión cada segundo para que la Pi lo encuentre sola.

    python pan_tilt_server.py            # o: bash uno_q/servos.sh
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time

from pan_tilt import DISCOVER_PORT, NET_PORT, RouterClient

METHODS = ("aim", "center", "status")
ANNOUNCE_S = 1.0


def handle(client, data):
    """Procesa un datagrama. Devuelve la respuesta (bytes) si la orden llevaba "id", si no None."""
    try:
        msg = json.loads(data.decode())
        method, args = msg.get("m"), list(msg.get("a", []))
    except (ValueError, AttributeError, TypeError):
        return None
    if method not in METHODS:
        return None
    if "id" in msg:
        return json.dumps({"id": msg["id"], "r": client.call(method, *args)}).encode()
    client.notify(method, *args)
    return None


HEALTH_S = 10  # cada tanto se comprueba que la MCU siga contestando
# App de App Lab con el sketch de los servos. Al encender la placa, la MCU registra sus
# funciones antes de que el router de Linux exista y se pierden ("method not available"); el
# botón RESET reinicia toda la placa y repite el problema. Lo que sí funciona es volver a
# cargar el sketch con Linux ya arriba (Run en App Lab), y eso se puede hacer desde aquí.
APP_DIR = os.path.expanduser("~/ArduinoApps/seguimiento_servos")
RELOAD_AFTER_S = 20    # sin respuesta de la MCU este tiempo: recargar el sketch
RELOAD_EVERY_S = 180   # y no insistir más seguido que esto


def find_app_cli():
    """arduino-app-cli: en el PATH o en las rutas habituales (el servicio de systemd tiene un
    PATH corto)."""
    found = shutil.which("arduino-app-cli")
    if found:
        return found
    for p in ("/usr/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin"), "/opt/arduino/bin"):
        if os.path.exists(os.path.join(p, "arduino-app-cli")):
            return os.path.join(p, "arduino-app-cli")
    return None


def reload_sketch():
    """Vuelve a cargar el sketch en la MCU con arduino-app-cli (lo mismo que Run en App Lab)."""
    cli = find_app_cli()
    if not os.path.isdir(APP_DIR) or cli is None:
        print(f"No se puede recargar el sketch solo (app en {APP_DIR}: {os.path.isdir(APP_DIR)}, "
              f"arduino-app-cli: {cli}); pulsa Run en App Lab.", file=sys.stderr, flush=True)
        return False
    print(f"Recargando el sketch de los servos en la MCU ({cli} app start {APP_DIR})...", flush=True)
    try:
        r = subprocess.run([cli, "app", "start", APP_DIR], capture_output=True, text=True, timeout=240)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print("  " + " | ".join(tail), flush=True)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  no se pudo: {e}", file=sys.stderr, flush=True)
        return False


def connect_router():
    """Se conecta al router y comprueba que la MCU tenga registradas las órdenes. Si no
    ("method status not available"), tras RELOAD_AFTER_S recarga el sketch (ver arriba) y
    sigue reintentando cada 3 s hasta que conteste."""
    warned, first_fail, last_reload = 0.0, None, 0.0
    while True:
        client = RouterClient()
        if client.connect():
            if client.call("status") is not None:
                client.call("center")
                return client
            client.close()
            now = time.time()
            first_fail = first_fail or now
            if now - warned > 30:
                warned = now
                print("La MCU no tiene registradas las órdenes de los servos: se recargará el sketch "
                      "(o pulsa Run en App Lab). Reintentando...", file=sys.stderr, flush=True)
            if now - first_fail >= RELOAD_AFTER_S and now - last_reload >= RELOAD_EVERY_S:
                last_reload = now
                reload_sketch()
        time.sleep(3)  # el router arranca después del inicio de sesión; se reintenta


def main():
    client = connect_router()
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", NET_PORT))
    srv.settimeout(0.5)
    announcer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    announcer.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    announce = json.dumps({"lsm": "pan_tilt", "port": NET_PORT}).encode()
    print(f"Puente de servos: órdenes por UDP {NET_PORT}, anuncios por difusión en {DISCOVER_PORT}. "
          f"Servos en {client.call('status')}", flush=True)
    last_announce, last_health, n, last_from = 0.0, time.time(), 0, None
    while True:
        now = time.time()
        if now - last_announce >= ANNOUNCE_S:
            last_announce = now
            try:
                announcer.sendto(announce, ("255.255.255.255", DISCOVER_PORT))
            except OSError:
                pass
        if now - last_health >= HEALTH_S:
            last_health = now
            if client.connected and client.call("status") is None:
                client.close()  # la MCU dejó de contestar (reinicio del router o de la MCU)
        if not client.connected:
            print("Router o MCU sin respuesta: reconectando...", file=sys.stderr, flush=True)
            client = connect_router()
            print(f"Servos de nuevo en {client.call('status')}", flush=True)
        try:
            data, addr = srv.recvfrom(512)
        except socket.timeout:
            continue
        reply = handle(client, data)
        if reply is not None:
            srv.sendto(reply, addr)
        n += 1
        if addr[0] != last_from:
            last_from = addr[0]
            print(f"Órdenes desde {addr[0]}", flush=True)
        if n % 100 == 0:
            print(f"{n} órdenes; servos en {client.call('status')}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Puente de servos cerrado.")
