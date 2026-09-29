# LSM Coach

Retroalimentación en tiempo real para quienes aprenden Lengua de Señas Mexicana (LSM).
Una cámara analiza la configuración de la mano y una muñequera con IMU mide su orientación.
El sistema fusiona ambas fuentes para decidir si la seña es correcta y dice **qué parámetro
corregir y cómo** ("Extiende el índice", "Baja la mano"). Avisa en pantalla, con voz y
con vibración en la muñequera.

Proyecto para el reto *Seña Correcta* de Indivisa Ingenium 2026. Nivel 1: letras A, B, C, L, Y.

## Componentes

| Parte | Qué hace | Archivos |
|---|---|---|
| Muñequera (Arduino Nano 33 BLE Sense Rev2 + motor de vibración en D3) | Mide roll/pitch con el IMU BMI270 (filtro Madgwick) y los envía por BLE; vibra al recibir una orden | `firmware/lsm_wrist/` |
| Visión y fusión (laptop) | MediaPipe Hands → ángulos de los 5 dedos; los combina con la orientación de la muñeca | `coach_engine.py`, `evaluation.py` |
| Interfaz web (monitor de 10.1") | Seña objetivo, veredicto, qué corregir, medidores por dedo y por eje, calibración | `web_server.py`, `web/` |
| Datos | SQLite local + sincronización con Supabase; CSV por seña para calibrar | `store.py`, `db/`, `samples/` |

## Instalación (una vez)

Requiere Python 3.12 (MediaPipe 0.10.21 no está disponible para 3.13+).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Muñequera: en el Arduino IDE, instala la placa "Arduino Mbed OS Nano Boards" y las
bibliotecas `ArduinoBLE`, `Arduino_BMI270_BMM150` y `MadgwickAHRS`. Luego sube
`firmware/lsm_wrist/lsm_wrist.ino`.

## Uso

```bash
python imu_test.py --vibrar              # 1. comprobar la muñequera (lecturas + vibración)
python web_server.py --list-cams         # 2. ver qué índice tiene cada cámara
python web_server.py --cam 0             # 3. arrancar; abrir http://localhost:8000
```

Pon el navegador del monitor en pantalla completa. Teclas: `1`–`5` cambian de seña y
`Espacio` dice en voz alta qué corregir.

Opciones: `--cam2 N` agrega una segunda cámara; `--imu none` corre sin muñequera, solo para pruebas.

## Calibrar una seña

1. En la interfaz, abre **Calibración**, escribe tu nombre y, con la muñequera puesta,
   haz la seña correctamente y pulsa **Guardar muestra correcta** (~20 veces, varias
   personas, mano derecha).
2. Graba también **errores intencionales** (dedo mal flexionado, muñeca girada).
3. Calcula los rangos, recárgalos desde la interfaz y mide la exactitud:

```bash
python tolerance_calculator.py analyze --sign A
python evaluate_accuracy.py --signs A B C L Y     # meta de la rúbrica: >= 90 %
```

## Datos públicos y clasificador

El reconocimiento de la forma de la mano usa un dataset público (ver Declaración de recursos).
Con el dataset descomprimido:

```bash
python import_msl.py --root ~/Downloads/datos_ent/MSL-ABC   # ~3 min, crea samples_public/
python evaluate_public.py                                   # mide con personas no vistas
python evaluate_public.py --percentiles 2 98 --guardar      # rangos base -> tolerances.json
python train_classifier.py --publicos --signs A B C L Y     # clasificador -> model.joblib
```

Una seña es correcta solo si **los 5 dedos están en rango**, **el clasificador reconoce la
letra** (confianza ≥ 0.5) y, si la seña tiene orientación calibrada, **la muñeca está en rango**.

## Base de datos en la nube (Supabase)

1. Crea un proyecto en supabase.com y ejecuta `db/supabase_schema.sql` en el *SQL Editor*.
2. Copia `.env.example` como `.env` y pon la URL y la clave secreta del proyecto.
3. `python store.py status` muestra lo pendiente; la app sincroniza sola cada 30 s.

La evaluación nunca depende de internet: todo se guarda primero en `lsm_coach.db` y
se sube cuando hay conexión.

## Pruebas

```bash
python -m unittest discover -s tests -t .
```

## Declaración de recursos

- **MediaPipe Hands** (Google, Apache 2.0): detección de 21 puntos de la mano.
- **OpenCV, FastAPI, Uvicorn, Bleak, Pillow, scikit-learn**: bibliotecas de código abierto.
- **Arduino ArduinoBLE, Arduino_BMI270_BMM150, MadgwickAHRS**: firmware de la muñequera.
- **Asistente de IA**: parte del código se desarrolló con apoyo de Claude (Anthropic).
- **Dataset público:** Morfín, R. *Mexican Sign Language Alphabet (static signs only)*. Zenodo.
  https://doi.org/10.5281/zenodo.10067508 (CC BY 4.0). Se usan las letras A, B, C, L y Y:
  landmarks extraídos con MediaPipe para entrenar el clasificador y estimar el ancho de los rangos.
- Los datos de calibración con nuestra cámara y la muñequera se graban durante el evento.
