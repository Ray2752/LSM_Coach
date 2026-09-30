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
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from coach_engine import Coach
from imu_source import DEVICE_NAME, BLEIMU, MockIMU
from signs import ABECEDARIO, DYNAMIC, NIVEL_1, SIGNS, WORDS, level, ref_name
from store import Store
from tolerance_calculator import OUTPUT_FILE, append_sample

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VIBRATE_MS = 250
MIN_SAVE_GAP_S = 0.7  # entre dos muestras: evita ráfagas de fotos casi idénticas
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


def load_word_model(path="model_words.joblib"):
    """Reconocedor de señas de palabras (train_words.py). Sin él, las palabras se practican
    deletreadas y se pueden grabar muestras desde Calibración."""
    if not os.path.exists(path):
        print(f"Aviso: no hay {path}; las palabras del Nivel 3 aún no se evalúan como seña.")
        return None
    import joblib
    bundle = joblib.load(path)
    bundle["model"].n_jobs = 1
    print(f"Reconocedor de palabras cargado: {', '.join(bundle['signs'])}")
    return bundle


def load_dynamic_model(path="model_dynamic.joblib"):
    """Clasificador de trazos (train_dynamic.py). Sin él, las letras con movimiento solo
    muestran la animación de referencia."""
    if not os.path.exists(path):
        print(f"Aviso: no hay {path}; las letras con movimiento no se evalúan.")
        return None
    import joblib
    bundle = joblib.load(path)
    bundle["model"].n_jobs = 1
    print(f"Clasificador de movimiento cargado: {', '.join(bundle['signs'])}")
    return bundle


class RpicamCapture:
    """Cámara CSI de la Raspberry Pi (p. ej. la AI Camera IMX500). Va por libcamera, así que
    OpenCV no la abre por V4L2: se lanza `rpicam-vid` y se leen sus fotogramas YUV420 crudos
    por una tubería (sin comprimir: convertir a BGR cuesta ~1 ms). Misma interfaz que
    cv2.VideoCapture (isOpened, read, release). El ancho debe ser múltiplo de 64 (640, 1280):
    es el paso de línea del ISP de la Pi; con otros anchos la imagen saldría desplazada."""

    def __init__(self, width, height, fps=30):
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3 // 2
        cmd = ["rpicam-vid", "-t", "0", "-n", "--codec", "yuv420", "--width", str(width),
               "--height", str(height), "--framerate", str(fps), "--flush", "-o", "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        buf = self.proc.stdout.read(self.frame_bytes)  # bloquea hasta tener el fotograma entero
        if len(buf) < self.frame_bytes:
            return False, None
        yuv = np.frombuffer(buf, np.uint8).reshape(self.height * 3 // 2, self.width)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Camera:
    """Lee una cámara en su propio hilo y guarda el último fotograma (ya en espejo).
    Si deja de mandar imagen (se desconectó o cambió de índice), la vuelve a abrir."""

    REOPEN_AFTER_S = 2.0

    def __init__(self, index, width=1280, height=720):
        # index: número (Mac/Windows), ruta "/dev/videoN" (Linux) o "auto" (la primera que dé imagen)
        self.index, self.width, self.height = index, width, height
        self.device = None  # qué se abrió de verdad (con "auto", la ruta encontrada)
        self.size = None    # resolución real de los fotogramas (ancho, alto)
        self.cap = self._open()
        if not self.cap.isOpened():
            sys.exit(f"No se pudo abrir la cámara {index}. Prueba: python web_server.py --list-cams")
        self.frame, self.seq, self.last_t = None, 0, time.time()
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        if self.index == "rpicam":
            self.device = "rpicam"
            return RpicamCapture(self.width, self.height)
        if self.index == "auto":
            for dev in linux_video_devices():
                cap = self._open_device(dev)
                if cap.isOpened() and cap.read()[0]:
                    print(f"Cámara: {dev}")
                    self.device = dev
                    return cap
                cap.release()
            if shutil.which("rpicam-vid"):  # sin cámara USB: la CSI de la Raspberry Pi (AI Camera)
                cap = RpicamCapture(self.width, self.height)
                if cap.isOpened() and cap.read()[0]:
                    print("Cámara: CSI de la Raspberry Pi (rpicam-vid)")
                    self.device = "rpicam"
                    return cap
                cap.release()
            return cv2.VideoCapture()  # ninguna: queda "sin abrir"
        self.device = self.index
        return self._open_device(self.index)

    def _open_device(self, dev):
        # En Linux se abre por V4L2 y en MJPG: en la UNO Q, YUYV solo llega a 30 fps hasta 640x480
        cap = cv2.VideoCapture(v4l_index(dev), cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(dev)
        if sys.platform.startswith("linux"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    @property
    def live(self):
        return time.time() - self.last_t < 1.5

    def _run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if time.time() - self.last_t > self.REOPEN_AFTER_S:
                    print(f"Cámara {self.index} sin imagen: reabriendo...", file=sys.stderr)
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self._open()
                    self.last_t = time.time()  # espera otra vez antes de reintentar
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = cv2.flip(frame, 1)  # igual que al grabar
                self.seq += 1
                self.last_t = time.time()
                self.size = (frame.shape[1], frame.shape[0])

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
                           model=load_model(), dyn_model=load_dynamic_model(),
                           word_model=load_word_model(),
                           hands_complexity=0 if getattr(args, "modelo_ligero", False) else 1,
                           classify_every=2 if getattr(args, "lite", False) else 1)
        self.coach.notify = self.notify
        self.tracker = None
        if args.seguir:  # cámara motorizada (servos en la UNO Q): sigue el rostro de la persona
            from pan_tilt import make_tracker
            self.tracker = make_tracker(args.seguir)  # "auto": router local o la UNO Q por red
            self.coach.track_face = self.tracker is not None
        self.imu = {"mock": MockIMU, "ble": BLEIMU}.get(args.imu, lambda: None)()
        if self.imu:
            self.imu.start()
        w, h = (int(v) for v in args.res.lower().split("x"))
        self.cams = [Camera(args.cam, w, h)] + ([Camera(args.cam2, w, h)] if args.cam2 is not None else [])
        self.jpeg = [None] * len(self.cams)
        self.person = "invitado"
        self.state = {}
        self.fps = 0.0
        self.message = None  # aviso breve para la interfaz: {"id", "text", "kind"}
        self.achievement = None  # última seña lograda: {"id", "sign", "seconds", ...} (celebración)
        self.alert = None        # último aviso de vibración: {"id", "action"} (icono de la muñequera)
        self._last_save = 0.0
        self._msg_id = 0
        self._event_id = 0
        self._lock = threading.Lock()  # protege al Coach (hilo de visión vs. órdenes web)
        threading.Thread(target=self._loop, daemon=True).start()

    def imu_values(self):
        if self.imu is None or not getattr(self.imu, "connected", True):
            return None  # sin conexión: nunca ceros falsos
        return dict(self.imu.latest)

    def _loop(self):
        last_seq, t_prev, t_log = -1, time.time(), 0.0
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
            if sys.platform.startswith("linux") and now - t_log > 15:  # diagnóstico en la placa
                t_log = now
                print(f"Visión: {self.fps:.1f} fps", flush=True)

    def _step(self, frame):
        with self._lock:
            state, alert, achievement = self.coach.process(frame, self.imu_values())
        if alert:
            if self.imu:
                self.imu.vibrate(VIBRATE_MS)
            self._event_id += 1
            self.alert = {"id": self._event_id, "action": state["issues"][0]["action"] if state["issues"] else ""}
            print(f"Vibra ({state['target']}): {self.alert['action'] or state['verdict']}", flush=True)
        if achievement:
            self.store.add_attempt(person=self.person, **achievement)
            self._event_id += 1
            self.achievement = {"id": self._event_id, **achievement}
        if self.tracker:
            self.tracker.update(self.coach._face, time.time(), busy=self.coach.busy,
                                face_t=self.coach.face_t)
        self.jpeg[0] = encode(frame)
        for i, cam in enumerate(self.cams[1:], start=1):
            other, _ = cam.read()
            if other is not None:
                self.jpeg[i] = encode(other)
        self.state = {**state, **self.status()}

    def status(self):
        v = self.imu_values()
        cam = self.cams[0]
        return {
            # datos de la Nano (muñequera): orientación + estado del enlace BLE (Hz, dirección, edad)
            "imu": {"mode": self.args.imu, "name": DEVICE_NAME,
                    **(self.imu.stats() if self.imu else {"connected": False}),
                    **({k: round(x, 1) for k, x in v.items()} if v else {})},
            "camera": {"device": str(cam.device), "width": cam.size[0] if cam.size else None,
                       "height": cam.size[1] if cam.size else None},
            "fps": round(self.fps, 1) if cam.live else 0,
            "camera_live": cam.live,
            "cameras": len(self.cams),
            "person": self.person,
            "progress": dict(self.coach.progress),
            "sync": self.store.status(),
            "message": self.message,
            "achievement": self.achievement,
            "alert": self.alert,
            "lite": bool(getattr(self.args, "lite", False)),  # la interfaz quita efectos costosos
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
            model, dyn_model, word_model = load_model(), load_dynamic_model(), load_word_model()
            with self._lock:
                self.coach.tolerances = self.tolerances
                self.coach.model = model
                self.coach.dyn_model = dyn_model
                self.coach.word_model = word_model
            self.notify(f"Rangos recargados: {', '.join(self.tolerances) or 'ninguno'}")

    def record(self, is_error):
        with self._lock:
            sample, sign = self.coach.last_sample, self.coach.target
        if time.time() - self._last_save < MIN_SAVE_GAP_S:
            return self.notify("Muy rápido: cambia un poco la mano y vuelve a guardar", "bad")
        if sign in WORDS:  # palabra: se graba la próxima seña completa (con rostro y movimiento)
            if is_error:
                return self.notify("Las palabras solo se graban correctas", "bad")
            with self._lock:
                self.coach.arm_word_record(self.person)
            return self.notify(f"Haz la seña de {sign} y detén la mano: se guardará al terminar")
        if sign in DYNAMIC:  # una sola imagen no captura el movimiento: dañaría el modelo
            return self.notify(f"La {sign} lleva movimiento: todavía no se puede grabar", "bad")
        if sample is None:
            return self.notify("No veo la mano: no se guardó", "bad")
        if self.imu and sample["imu"] is None:
            return self.notify("La muñequera no está conectada: no se guardó", "bad")
        self._last_save = time.time()
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
server: uvicorn.Server = None  # también en main(); sirve para saber si se está apagando


def stopping():
    """True al pulsar Ctrl+C: el video y el WebSocket (conexiones sin fin) se cierran solos
    para que el servidor se apague con un solo Ctrl+C y sin errores."""
    return server is not None and server.should_exit


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # sin ícono: evita un 404 en cada carga


@app.get("/api/signs")
def api_signs():
    out = []
    for sign in ABECEDARIO:
        tol = rt.tolerances.get(sign)
        dynamic = sign in DYNAMIC
        info = {k: v for k, v in SIGNS[sign].items() if k != "forma"}
        out.append({
            **info, "sign": sign, "level": level(sign), "dynamic": dynamic,
            "calibrated": tol is not None or (dynamic and rt.coach.dyn_model is not None),
            "shape": SIGNS[sign].get("forma", {}),
            "orientation": bool(tol and tol.get("orientacion")),
            "ref": f"/static/ref/{ref_name(sign, 'gif' if dynamic else 'jpg')}",
        })
    known = set(rt.coach.word_model["signs"]) if rt.coach.word_model else set()
    for word in WORDS:  # Nivel 3: señas de palabras (referencia opcional en web/ref/words/)
        ref = f"words/{word.replace(' ', '_')}.gif"
        out.append({
            **SIGNS[word], "sign": word, "level": 3, "dynamic": True, "word": True,
            "calibrated": word in known, "shape": {}, "orientation": False,
            "ref": f"/static/ref/{ref}" if os.path.exists(os.path.join(WEB_DIR, "ref", ref)) else None,
        })
    return out


@app.get("/video/{index}")
async def video(index: int):
    async def frames():
        last = None
        while not stopping():
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
        while not stopping():  # el estado de los dispositivos siempre fresco
            await ws.send_text(json.dumps({**rt.state, **rt.status()}))
            await asyncio.sleep(1 / STATE_HZ)

    async def receiver():
        while True:
            rt.command(await ws.receive_json())

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    # termina cuando el navegador se va (receiver falla) o el servidor se apaga (sender acaba)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        t.exception()  # el navegador se fue (WebSocketDisconnect): no es un error
    if stopping():
        try:
            await ws.close()
        except Exception:  # el navegador ya había cerrado por su lado: nada que hacer
            pass


def linux_video_devices():
    """Rutas /dev/video* que pueden ser cámaras, las USB primero. En la UNO Q, /dev/video0 y 1
    son el códec de video del procesador; en la Raspberry Pi 5 hay decenas de nodos del ISP
    (pispbe, rp1-cfe, hevc) que no dan imagen y se saltan. La cámara USB cambia de número
    entre reinicios: por eso existe --cam auto, que prueba cada una hasta la que sí da imagen."""
    import glob
    devices = []
    for path in glob.glob("/dev/video*"):
        num = path[len("/dev/video"):]
        if not num.isdigit():
            continue
        sysfs = f"/sys/class/video4linux/video{num}"
        usb = "/usb" in os.path.realpath(f"{sysfs}/device") if os.path.exists(f"{sysfs}/device") else False
        if not usb:
            try:
                name = open(f"{sysfs}/name").read().strip().lower()
            except OSError:
                name = ""
            if any(k in name for k in ("pispbe", "rp1-cfe", "hevc", "unicam", "codec", "isp")):
                continue
        devices.append((0 if usb else 1, int(num), path))
    return [p for _, _, p in sorted(devices)]


def v4l_index(dev):
    """El backend V4L2 de OpenCV en la UNO Q no abre por nombre ("/dev/video2"), solo por
    número (2 -> /dev/video2). Convierte la ruta a número; lo demás se deja igual."""
    if isinstance(dev, str) and dev.startswith("/dev/video") and dev[len("/dev/video"):].isdigit():
        return int(dev[len("/dev/video"):])
    return dev


def cam_arg(value):
    """--cam acepta un número, una ruta /dev/videoN o 'auto'."""
    return int(value) if value.isdigit() else value


def list_cams(max_index=6):
    """Muestra cada cámara y guarda una foto de cada una para saber cuál es cuál."""
    import tempfile
    out_dir = os.path.join(tempfile.gettempdir(), "lsm_cams")
    os.makedirs(out_dir, exist_ok=True)
    linux = sys.platform.startswith("linux")
    devices = linux_video_devices() if linux else range(max_index)
    print(("Ruta" if linux else "Índice") + "   resolución   foto")
    for i in devices:
        cap = cv2.VideoCapture(v4l_index(i), cv2.CAP_V4L2) if linux else cv2.VideoCapture(i)
        if not cap.isOpened():
            if linux:
                continue
            break  # en Mac los índices son consecutivos: al primer hueco ya no hay más
        ok = False
        for _ in range(5):  # las primeras imágenes suelen salir oscuras
            ok, frame = cap.read()
        cap.release()
        if ok:
            name = os.path.basename(str(i)) if linux else f"camara_{i}"
            path = os.path.join(out_dir, f"{name}.jpg")
            cv2.imwrite(path, frame)
            print(f"  {i}     {frame.shape[1]}x{frame.shape[0]}   {path}")
    if sys.platform == "darwin":
        print(f"\nAbre las fotos con:  open {out_dir}")
    elif linux:
        if shutil.which("rpicam-hello"):  # Raspberry Pi: cámaras CSI (AI Camera) vía libcamera
            print("\nCámaras CSI de la Raspberry Pi (--cam rpicam):")
            subprocess.run(["rpicam-hello", "--list-cameras"], timeout=20)
        print("\nEn la placa puedes usar --cam auto para no depender del número (USB primero, luego CSI).")


def main():
    global rt, server
    ap = argparse.ArgumentParser(description="Interfaz web de LSM Coach")
    ap.add_argument("--cam", type=cam_arg, default=0,
                    help="cámara principal: número, /dev/videoN, rpicam (CSI de la Raspberry Pi) "
                         "o auto (Linux: la primera USB con imagen, si no la CSI)")
    ap.add_argument("--cam2", type=cam_arg, help="segunda cámara (vista adicional)")
    ap.add_argument("--res", default="1280x720", metavar="ANCHOxALTO",
                    help="resolución de captura (640x480 en la UNO Q: menos trabajo para el procesador)")
    ap.add_argument("--sign", default=NIVEL_1[0], choices=ABECEDARIO, help="seña inicial")
    ap.add_argument("--imu", choices=["ble", "none", "mock"], default="ble",
                    help="ble = Nano real (por defecto); mock solo para pruebas, NO en la demo")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 para abrir la interfaz desde otro equipo del hotspot")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lite", action="store_true",
                    help="UNO Q: video MJPEG más ligero (calidad 55, 12 fps) e interfaz sin efectos")
    ap.add_argument("--modelo-ligero", action="store_true",
                    help="modelo ligero de manos de MediaPipe (~2x más rápido; ver uno_q/bench.py)")
    ap.add_argument("--seguir", nargs="?", const="auto", metavar="IP",
                    help="cámara motorizada (servos en la UNO Q, uno_q/pan_tilt): sigue el rostro. Sin valor "
                         "usa el router local (UNO Q) o busca la UNO Q en la red (Pi); o su IP")
    ap.add_argument("--list-cams", action="store_true", help="muestra las cámaras y sale")
    args = ap.parse_args()
    if args.list_cams:
        return list_cams()
    if args.lite:
        global JPEG_QUALITY, VIDEO_FPS
        JPEG_QUALITY, VIDEO_FPS = 55, 12  # menos trabajo para codificar y para que el navegador decodifique

    rt = Runtime(args)
    print(f"Abre http://localhost:{args.port} en el monitor (pantalla completa). Ctrl+C para salir.")
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning",
                                           timeout_graceful_shutdown=2))
    try:
        server.run()
    except KeyboardInterrupt:  # segundo Ctrl+C mientras cierra: salir sin traceback
        pass
    if rt.imu:
        rt.imu.stop()
    print("Servidor cerrado.")


if __name__ == "__main__":
    main()
