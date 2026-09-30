"""Puente de red para la cámara motorizada. Corre en la UNO Q cuando la visión va en otra
placa (Raspberry Pi 5): recibe por UDP las órdenes aim/center/status (pan_tilt.NetClient) y
las pasa al arduino-router, es decir, a la MCU que mueve los servos (uno_q/pan_tilt/sketch.ino).
Se anuncia por difusión cada segundo para que la Pi lo encuentre sola.

    python pan_tilt_server.py            # o: bash uno_q/servos.sh
"""
import json
import socket
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


def connect_router():
    while True:
        client = RouterClient()
        if client.connect():
            client.call("center")
            return client
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
    last_announce, n, last_from = 0.0, 0, None
    while True:
        now = time.time()
        if now - last_announce >= ANNOUNCE_S:
            last_announce = now
            try:
                announcer.sendto(announce, ("255.255.255.255", DISCOVER_PORT))
            except OSError:
                pass
        if not client.connected:
            print("Router desconectado: reconectando...", file=sys.stderr)
            client = connect_router()
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
