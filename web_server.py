"""Servidor de la interfaz web de LSM Coach (para el monitor HDMI de 10.1").

La laptop hace la visión (cámara + MediaPipe), la fusiona con la muñequera y sirve:
    /            la interfaz (carpeta web/)
    /video/0     cámara principal con la mano dibujada (MJPEG)
    /video/1     segunda cámara, si se indica con --cam2
    /ws          estado en vivo (JSON ~10 veces por segundo) y órdenes de la interfaz

Uso:
    python web_server.py --list-cams             # qué índice tiene cada cámara
    python web_server.py                         # cámara 0, muñequera por BLE
    python web_server.py --cam 1 --cam2 2        # elige cámaras
    python web_server.py --imu none              # sin muñequera (solo pruebas)
Luego abre http://localhost:8000 en el monitor y ponlo en pantalla completa.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from coach_engine import Coach
from imu_source import BLEIMU, MockIMU
from signs import NIVEL_1, SIGNS
from store import Store
from tolerance_calculator import OUTPUT_FILE, append_sample

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VIBRATE_MS = 250
STATE_HZ = 10
VIDEO_FPS = 25
JPEG_QUALITY = 80


def load_tolerances():
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_model(path="model.joblib"):
    """Clasificador de landmarks (train_classifier.py). Sin él, solo se usan las reglas."""
    if not os.path.exists(path):
        print(f"Aviso: no hay {path}; se evalúa solo con los rangos por dedo.")
        return None
    import joblib
    model = joblib.load(path)
    final = model[-1] if hasattr(model, "steps") else model  # Pipeline o clasificador solo
    final.n_jobs = 1  # un fotograma a la vez: los hilos en paralelo solo añaden retraso
    print(f"Clasificador cargado: {', '.join(map(str, model.classes_))}")
    return model


class Camera:
    """Lee una cámara en su propio hilo y guarda el último fotograma (ya en espejo)."""

    def __init__(self, index, width=1280, height=720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit(f"No se pudo abrir la cámara {index}. Prueba: python web_server.py --list-cams")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.index, self.frame, self.seq = index, None, 0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = cv2.flip(frame, 1)  # igual que al grabar
                self.seq += 1

    def read(self):
        with self._lock:
            return (None if self.frame is None else self.frame.copy()), self.seq


class Runtime:
    """Estado compartido entre el hilo de visión y las conexiones web."""

    def __init__(self, args):
        self.args = args
        self.store = Store()
        if self.store.cloud_enabled:
            self.store.start_sync()
        self.tolerances = load_tolerances()
        self.coach = Coach(self.tolerances, target=args.sign, require_imu=args.imu != "none",
                           model=load_model())
        self.imu = {"mock": MockIMU, "ble": BLEIMU}.get(args.imu, lambda: None)()
        if self.imu:
            self.imu.start()
        self.cams = [Camera(args.cam)] + ([Camera(args.cam2)] if args.cam2 is not None else [])
        self.jpeg = [None] * len(self.cams)
        self.person = "invitado"
        self.state = {}
        self.fps = 0.0
        self.message = None  # aviso breve para la interfaz: {"id", "text", "kind"}
        self._msg_id = 0
        self._lock = threading.Lock()  # protege al Coach (hilo de visión vs. órdenes web)
        threading.Thread(target=self._loop, daemon=True).start()

    def imu_values(self):
        if self.imu is None or not getattr(self.imu, "connected", True):
            return None  # sin conexión: nunca ceros falsos
        return dict(self.imu.latest)

    def _loop(self):
        last_seq, t_prev = -1, time.time()
        while True:
            frame, seq = self.cams[0].read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                self._step(frame)
            except Exception as e:  # un fotograma malo no debe congelar la interfaz
                print(f"Error procesando fotograma: {e!r}", file=sys.stderr)
                time.sleep(0.1)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

    def _step(self, frame):
        with self._lock:
            state, alert, achievement = self.coach.process(frame, self.imu_values())
        if alert and self.imu:
            self.imu.vibrate(VIBRATE_MS)
        if achievement:
            self.store.add_attempt(person=self.person, **achievement)
        self.jpeg[0] = encode(frame)
        for i, cam in enumerate(self.cams[1:], start=1):
            other, _ = cam.read()
            if other is not None:
                self.jpeg[i] = encode(other)
        self.state = {**state, **self.status()}

    def status(self):
        v = self.imu_values()
        return {
            "signs": NIVEL_1,
            "imu": {"mode": self.args.imu, "connected": v is not None,
                    **({k: round(x, 1) for k, x in v.items()} if v else {})},
            "fps": round(self.fps, 1),
            "cameras": len(self.cams),
            "person": self.person,
            "progress": dict(self.coach.progress),
            "sync": self.store.status(),
            "message": self.message,
        }

    def notify(self, text, kind="ok"):
        self._msg_id += 1
        self.message = {"id": self._msg_id, "text": text, "kind": kind}

    # --- órdenes que llegan desde la interfaz ---

    def command(self, msg):
        kind = msg.get("type")
        if kind == "target" and msg.get("sign") in SIGNS:
            with self._lock:
                self.coach.set_target(msg["sign"])
            print(f"Seña objetivo: {msg['sign']}")
        elif kind == "person":
            self.person = (msg.get("name") or "invitado").strip()[:30] or "invitado"
        elif kind == "record":
            self.record(bool(msg.get("is_error")))
        elif kind == "reload":
            self.tolerances = load_tolerances()
            model = load_model()
            with self._lock:
                self.coach.tolerances = self.tolerances
                self.coach.model = model
            self.notify(f"Rangos recargados: {', '.join(self.tolerances) or 'ninguno'}")

    def record(self, is_error):
        with self._lock:
            sample, sign = self.coach.last_sample, self.coach.target
        if sample is None:
            return self.notify("No veo la mano: no se guardó", "bad")
        if self.imu and sample["imu"] is None:
            return self.notify("La muñequera no está conectada: no se guardó", "bad")
        append_sample(sign, self.person, sample["angles"], sample["landmarks"], sample["imu"],
                      errores=is_error)
        self.store.add_sample(self.person, sign, sample["angles"], sample["landmarks"],
                              sample["imu"], is_error=is_error, camera=str(self.args.cam))
        n = self.store.count_samples(sign, is_error)
        self.notify(f"Guardada: {sign} {'(error intencional)' if is_error else '(correcta)'} "
                    f"— llevas {n}")


def encode(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


app = FastAPI(title="LSM Coach")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
rt: Runtime = None  # se crea en main()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/signs")
def api_signs():
    out = []
    for sign in NIVEL_1:
        tol = rt.tolerances.get(sign)
        info = {k: v for k, v in SIGNS[sign].items() if k != "forma"}
        out.append({
            **info, "sign": sign, "calibrated": tol is not None,
            "shape": SIGNS[sign].get("forma", {}),
            "orientation": bool(tol and tol.get("orientacion")),
        })
    return out


@app.get("/video/{index}")
async def video(index: int):
    async def frames():
        last = None
        while True:
            jpg = rt.jpeg[index] if index < len(rt.jpeg) else None
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1 / VIDEO_FPS)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    async def sender():
        while True:
            await ws.send_text(json.dumps(rt.state))
            await asyncio.sleep(1 / STATE_HZ)

    async def receiver():
        while True:
            rt.command(await ws.receive_json())

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()


def list_cams(max_index=6):
    """Muestra cada cámara y guarda una foto de cada una para saber cuál es cuál."""
    import tempfile
    out_dir = os.path.join(tempfile.gettempdir(), "lsm_cams")
    os.makedirs(out_dir, exist_ok=True)
    print("Índice  resolución   foto")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():  # los índices son consecutivos: al primer hueco ya no hay más
            break
        for _ in range(5):  # las primeras imágenes suelen salir oscuras
            ok, frame = cap.read()
        cap.release()
        if ok:
            path = os.path.join(out_dir, f"camara_{i}.jpg")
            cv2.imwrite(path, frame)
            print(f"  {i}     {frame.shape[1]}x{frame.shape[0]}   {path}")
    if sys.platform == "darwin":
        print(f"\nAbre las fotos con:  open {out_dir}")


def main():
    global rt
    ap = argparse.ArgumentParser(description="Interfaz web de LSM Coach")
    ap.add_argument("--cam", type=int, default=0, help="cámara principal (evalúa la seña)")
    ap.add_argument("--cam2", type=int, help="segunda cámara (vista adicional)")
    ap.add_argument("--sign", default=NIVEL_1[0], choices=NIVEL_1, help="seña inicial")
    ap.add_argument("--imu", choices=["ble", "none", "mock"], default="ble",
                    help="ble = Nano real (por defecto); mock solo para pruebas, NO en la demo")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 para abrir la interfaz desde otro equipo del hotspot")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--list-cams", action="store_true", help="muestra las cámaras y sale")
    args = ap.parse_args()
    if args.list_cams:
        return list_cams()

    rt = Runtime(args)
    print(f"Abre http://localhost:{args.port} en el monitor (pantalla completa).")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
