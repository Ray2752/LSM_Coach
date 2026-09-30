#include <Arduino_RouterBridge.h>
#include <Servo.h>

const int PAN_PIN = 9, TILT_PIN = 10;
const int PAN_MIN = 15, PAN_MAX = 165;     // topes mecánicos (ajustar al soporte)
const int TILT_MIN = 40, TILT_MAX = 140;
const int PAN_CENTER = 90, TILT_CENTER = 90;
const int STEP_DEG = 2;                    // grados por paso...
const unsigned long STEP_MS = 20;          // ...cada 20 ms = 100 °/s como máximo

// Registro con el router de Linux. Bridge.provide() solo funciona si el router ya arrancó:
// al encender la placa la MCU arranca en segundos y Linux tarda ~1 min, así que se reintenta
// hasta lograrlo. Y si el router se reinicia olvida los métodos: si no llega ninguna orden en
// REANNOUNCE_MS (Linux pregunta "status" cada 10 s), se vuelven a anunciar.
const unsigned long REGISTER_RETRY_MS = 2000;
const unsigned long REANNOUNCE_MS = 25000;

Servo pan, tilt;
int panTarget = PAN_CENTER, tiltTarget = TILT_CENTER;
int panNow = PAN_CENTER, tiltNow = TILT_CENTER;
unsigned long lastStep = 0, lastCommand = 0, lastRegisterTry = 0;
bool registered = false;

int clampInt(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// Linux -> MCU: nuevo objetivo. Devuelve 1. (No llamar a Bridge.call aquí: bloquea el IPC.)
int aim(int p, int t) {
  panTarget = clampInt(p, PAN_MIN, PAN_MAX);
  tiltTarget = clampInt(t, TILT_MIN, TILT_MAX);
  lastCommand = millis();
  return 1;
}

int center() { return aim(PAN_CENTER, TILT_CENTER); }

// "pan,tilt" actuales, por si la visión quiere saber dónde está la cámara
String status() {
  lastCommand = millis();
  return String(panNow) + "," + String(tiltNow);
}

// Registra las tres funciones (router + MCU). Devuelve true si las tres quedaron.
bool registerAll() {
  bool ok = true;
  ok = Bridge.provide("aim", aim) && ok;
  ok = Bridge.provide("center", center) && ok;
  ok = Bridge.provide("status", status) && ok;
  return ok;
}

// Solo vuelve a anunciar los nombres al router (ya están enlazados en la MCU).
void reannounce() {
  const char* names[] = {"aim", "center", "status"};
  for (int i = 0; i < 3; i++) {
    bool res = false;
    Bridge.call("$/register", MsgPack::str_t(names[i])).result(res);
  }
}

void setup() {
  Bridge.begin();
  pan.attach(PAN_PIN);
  tilt.attach(TILT_PIN);
  pan.write(PAN_CENTER);
  tilt.write(TILT_CENTER);
  registered = registerAll();
  lastRegisterTry = lastCommand = millis();
}

void loop() {
  unsigned long now = millis();
  if (!registered) {
    if (now - lastRegisterTry >= REGISTER_RETRY_MS) {
      lastRegisterTry = now;
      registered = registerAll();
      lastCommand = now;
    }
  } else if (now - lastCommand >= REANNOUNCE_MS) {
    lastCommand = now;
    reannounce();
  }
  if (now - lastStep >= STEP_MS) {
    lastStep = now;
    if (panNow != panTarget) {
      panNow += (panTarget > panNow) ? min(STEP_DEG, panTarget - panNow) : -min(STEP_DEG, panNow - panTarget);
      pan.write(panNow);
    }
    if (tiltNow != tiltTarget) {
      tiltNow += (tiltTarget > tiltNow) ? min(STEP_DEG, tiltTarget - tiltNow) : -min(STEP_DEG, tiltNow - tiltTarget);
      tilt.write(tiltNow);
    }
  }
}
