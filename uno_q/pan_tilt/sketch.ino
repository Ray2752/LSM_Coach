#include <Arduino_RouterBridge.h>
#include <Servo.h>

const int PAN_PIN = 9, TILT_PIN = 10;
const int PAN_MIN = 15, PAN_MAX = 165;     // topes mecánicos (ajustar al soporte)
const int TILT_MIN = 40, TILT_MAX = 140;
const int PAN_CENTER = 90, TILT_CENTER = 90;
const int STEP_DEG = 2;                    // grados por paso...
const unsigned long STEP_MS = 20;          // ...cada 20 ms = 100 °/s como máximo

Servo pan, tilt;
int panTarget = PAN_CENTER, tiltTarget = TILT_CENTER;
int panNow = PAN_CENTER, tiltNow = TILT_CENTER;
unsigned long lastStep = 0, lastCommand = 0;

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
String status() { return String(panNow) + "," + String(tiltNow); }

void setup() {
  Bridge.begin();
  pan.attach(PAN_PIN);
  tilt.attach(TILT_PIN);
  pan.write(PAN_CENTER);
  tilt.write(TILT_CENTER);
  Bridge.provide("aim", aim);
  Bridge.provide("center", center);
  Bridge.provide("status", status);
}

void loop() {
  unsigned long now = millis();
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
