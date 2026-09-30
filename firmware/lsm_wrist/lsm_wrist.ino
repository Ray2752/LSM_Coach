#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <MadgwickAHRS.h>
#include <Wire.h>

const char *DEVICE_NAME = "LSM-Wrist";
const char *SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char *ORIENT_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214";
const char *VIB_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214";

const int VIB_PIN = 4;
const int VIB_ON = HIGH;
const unsigned long VIB_MAX_MS = 2000;

const unsigned long TEST_EVERY_MS = 0;
const unsigned long TEST_ON_MS = 1000;

const unsigned long SEND_MS = 50;
const unsigned long DEBUG_MS = 200;

const int AXIS_MAP[3][2] = {{0, 1}, {1, 1}, {2, 1}};

BLEService lsmService(SERVICE_UUID);
BLECharacteristic orientChar(ORIENT_UUID, BLERead | BLENotify, 12);
BLEByteCharacteristic vibChar(VIB_UUID, BLEWrite | BLEWriteWithoutResponse);

Madgwick filter;
unsigned long lastSend = 0, lastDebug = 0, lastTest = 0;
unsigned long lastImuUs = 0, imuCount = 0;

int pulsesLeft = 0;
bool vibOn = false;
unsigned long onMs = 0, offMs = 0, phaseStart = 0;

void motor(bool on) {
  vibOn = on;
  int level = on ? VIB_ON : !VIB_ON;
  digitalWrite(VIB_PIN, level);
  if (TEST_EVERY_MS && Serial) {
    Serial.print(on ? "motor ON  (D" : "motor OFF (D");
    Serial.print(VIB_PIN);
    Serial.println(level == HIGH ? "=HIGH)" : "=LOW)");
  }
}

void vibratePattern(int pulses, unsigned long on, unsigned long off) {
  onMs = min(on, VIB_MAX_MS);
  pulsesLeft = onMs > 0 ? pulses : 0;
  offMs = off;
  phaseStart = millis();
  motor(pulses > 0 && onMs > 0);
}

void updateVibration() {
  if (pulsesLeft == 0) return;
  unsigned long elapsed = millis() - phaseStart;
  if (vibOn && elapsed >= onMs) {
    motor(false);
    pulsesLeft--;
    phaseStart = millis();
  } else if (!vibOn && elapsed >= offMs) {
    motor(true);
    phaseStart = millis();
  }
}

void mapAxes(float in[3], float out[3]) {
  for (int i = 0; i < 3; i++) out[i] = in[AXIS_MAP[i][0]] * AXIS_MAP[i][1];
}

bool readImu(float a[3], float g[3]) {
  if (!IMU.accelerationAvailable()) return false;
  float araw[3], graw[3];
  IMU.readAcceleration(araw[0], araw[1], araw[2]);
  IMU.readGyroscope(graw[0], graw[1], graw[2]);
  mapAxes(araw, a);
  mapAxes(graw, g);
  return true;
}

void warmUpFilter() {
  float a[3], g[3];
  unsigned long start = millis();
  while (!readImu(a, g)) {
    if (millis() - start > 500) return;
  }
  filter.begin(2.0f);
  for (int i = 0; i < 200; i++) filter.updateIMU(0, 0, 0, a[0], a[1], a[2]);
  lastImuUs = micros();
}

bool beginWithRetries(int (*beginFn)(), void (*endFn)()) {
  for (int i = 0; i < 3; i++) {
    if (beginFn()) return true;
    if (endFn) endFn();
    delay(200);
  }
  return false;
}

int imuBegin() { return IMU.begin(); }
int bleBegin() { return BLE.begin(); }
void bleEnd() { BLE.end(); }

void onConnected(BLEDevice) {
  vibratePattern(2, 80, 120);
}

void onDisconnected(BLEDevice) {
  vibratePattern(0, 0, 0);
  BLE.advertise();
}

void failBlink() {
  motor(false);
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(150);
  }
}

void setup() {
  pinMode(VIB_PIN, OUTPUT);
  motor(false);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);

  if (!beginWithRetries(imuBegin, nullptr)) failBlink();
  Wire1.setClock(400000);
  if (!beginWithRetries(bleBegin, bleEnd)) failBlink();

  warmUpFilter();

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  lsmService.addCharacteristic(orientChar);
  lsmService.addCharacteristic(vibChar);
  BLE.addService(lsmService);
  BLE.setAdvertisedService(lsmService);
  BLE.setEventHandler(BLEConnected, onConnected);
  BLE.setEventHandler(BLEDisconnected, onDisconnected);
  BLE.advertise();

  vibratePattern(1, 150, 0);
}

void loop() {
  BLE.poll();

  if (vibChar.written()) vibratePattern(1, (unsigned long)vibChar.value() * 10, 0);
  updateVibration();

  float a[3], g[3];
  if (readImu(a, g)) {
    unsigned long us = micros();
    float dt = (us - lastImuUs) * 1e-6f;
    lastImuUs = us;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.01f;
    filter.begin(1.0f / dt);
    filter.updateIMU(g[0], g[1], g[2], a[0], a[1], a[2]);
    imuCount++;
  }

  unsigned long now = millis();
  if (TEST_EVERY_MS && now - lastTest >= TEST_EVERY_MS) {
    lastTest = now;
    vibratePattern(1, TEST_ON_MS, 0);
  }

  bool connected = BLE.connected();
  digitalWrite(LED_BUILTIN, (connected || (now / 500) % 2) ? HIGH : LOW);

  if (connected && now - lastSend >= SEND_MS) {
    lastSend = now;
    float rpy[3] = {filter.getRoll(), filter.getPitch(), filter.getYaw()};
    orientChar.writeValue((uint8_t *)rpy, sizeof(rpy));
  }

  if (DEBUG_MS && now - lastDebug >= DEBUG_MS) {
    unsigned long hz = imuCount * 1000 / (now - lastDebug);
    imuCount = 0;
    lastDebug = now;
    if (Serial) {
      Serial.print("roll=");  Serial.print(filter.getRoll(), 1);
      Serial.print(" pitch="); Serial.print(filter.getPitch(), 1);
      Serial.print(" yaw=");   Serial.print(filter.getYaw(), 1);
      Serial.print(" imu_hz="); Serial.print(hz);
      Serial.println(connected ? " BLE=conectado" : " BLE=esperando");
    }
  }
}