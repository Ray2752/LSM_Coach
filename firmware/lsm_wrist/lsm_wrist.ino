#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <MadgwickAHRS.h>

const char *DEVICE_NAME = "LSM-Wrist";
const char *SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char *ORIENT_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214";
const char *VIB_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214";

const int VIB_PIN = 3;                 // D3, hacia la base del transistor (con resistencia ~1 kΩ)
const unsigned long SEND_MS = 50;      // 20 envíos por segundo
const unsigned long DEBUG_MS = 200;    // salida por serial para depurar (0 = apagada)

// Reasigna ejes si hace falta: {índice de eje (0=x,1=y,2=z), signo}
const int AXIS_MAP[3][2] = {{0, 1}, {1, 1}, {2, 1}};

BLEService lsmService(SERVICE_UUID);
BLECharacteristic orientChar(ORIENT_UUID, BLERead | BLENotify, 12);
BLEByteCharacteristic vibChar(VIB_UUID, BLEWrite | BLEWriteWithoutResponse);

Madgwick filter;
unsigned long lastSend = 0, lastDebug = 0, vibUntil = 0;

void mapAxes(float in[3], float out[3]) {
  for (int i = 0; i < 3; i++) out[i] = in[AXIS_MAP[i][0]] * AXIS_MAP[i][1];
}

void vibrate(unsigned long ms) {
  digitalWrite(VIB_PIN, HIGH);
  vibUntil = millis() + ms;
}

void onDisconnected(BLEDevice) {
  BLE.advertise();  // vuelve a anunciarse para que la app pueda reconectar
}

void failBlink() {  // error fatal: LED parpadeando
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(150);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(VIB_PIN, OUTPUT);
  digitalWrite(VIB_PIN, LOW);
  Serial.begin(115200);  // no se espera al monitor serie: funciona sin USB

  if (!IMU.begin()) failBlink();
  if (!BLE.begin()) failBlink();

  filter.begin(IMU.accelerationSampleRate());

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  lsmService.addCharacteristic(orientChar);
  lsmService.addCharacteristic(vibChar);
  BLE.addService(lsmService);
  BLE.setAdvertisedService(lsmService);
  BLE.setEventHandler(BLEDisconnected, onDisconnected);
  BLE.advertise();

  vibrate(150);  // pulso corto: el IMU y el BLE arrancaron bien
}

void loop() {
  BLE.poll();

  if (vibChar.written()) vibrate((unsigned long)vibChar.value() * 10);
  if (vibUntil && millis() >= vibUntil) {
    digitalWrite(VIB_PIN, LOW);
    vibUntil = 0;
  }

  float a[3], g[3];
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float araw[3], graw[3];
    IMU.readAcceleration(araw[0], araw[1], araw[2]);
    IMU.readGyroscope(graw[0], graw[1], graw[2]);
    mapAxes(araw, a);
    mapAxes(graw, g);
    filter.updateIMU(g[0], g[1], g[2], a[0], a[1], a[2]);
  }

  unsigned long now = millis();
  bool connected = BLE.connected();
  digitalWrite(LED_BUILTIN, connected ? HIGH : (now / 500) % 2);  // fijo = conectado

  if (connected && now - lastSend >= SEND_MS) {
    lastSend = now;
    float rpy[3] = {filter.getRoll(), filter.getPitch(), filter.getYaw()};
    orientChar.writeValue((uint8_t *)rpy, sizeof(rpy));  // 3 float little-endian
  }

  if (DEBUG_MS && Serial && now - lastDebug >= DEBUG_MS) {
    lastDebug = now;
    Serial.print("roll=");  Serial.print(filter.getRoll(), 1);
    Serial.print(" pitch="); Serial.print(filter.getPitch(), 1);
    Serial.print(" yaw=");   Serial.println(filter.getYaw(), 1);
  }
}
