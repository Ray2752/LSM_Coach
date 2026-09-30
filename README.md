# LSM Coach

Retroalimentación en tiempo real para quienes aprenden Lengua de Señas Mexicana (LSM).
Una cámara analiza la configuración de la mano y una muñequera con IMU mide su orientación.
El sistema fusiona ambas fuentes para decidir si la seña es correcta y dice **qué parámetro
corregir y cómo** ("Extiende el índice", "Baja la mano"). Avisa en pantalla, con voz y
con vibración en la muñequera.

Proyecto para el reto *Seña Correcta* de Indivisa Ingenium 2026. Nivel 1: letras A, B, C, L, Y
(obligatorio); además se pueden practicar las 21 letras estáticas del abecedario. Las letras con
movimiento (J, K, Ñ, Q, X, Z; Nivel 2) muestran su animación de referencia y aún no se evalúan.

## Componentes

| Parte | Qué hace | Archivos |
|---|---|---|
| Muñequera (Arduino Nano 33 BLE Sense Rev2 + motor de vibración en D2, vía transistor 2N2222) | Mide roll/pitch con el IMU BMI270 (filtro Madgwick) y los envía por BLE; vibra al recibir una orden | `firmware/lsm_wrist/` |
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

## Demo en la Arduino UNO Q (todo en la placa, sin laptop)

La UNO Q corre todo: cámara USB, visión, servidor web, Bluetooth con el guante y el
navegador en la pantalla HDMI. Hardware: hub USB-C con entrada de energía (power bank
5 V/3 A), monitor HDMI con su propia alimentación, cámara InnoMaker y teclado para
configurar. La visión va a ~12 fotogramas por segundo en su procesador (suficiente).

1. Primer arranque en modo computadora (hub + pantalla + teclado): crear la contraseña,
   conectar el Wi-Fi con internet.
2. En una terminal de la UNO Q:

```bash
curl -fsSL https://raw.githubusercontent.com/Ray2752/LSM_Coach/main/uno_q/install.sh | bash
```

   Instala Python 3.12 (MediaPipe no existe para ARM con el 3.13 que trae la placa),
   el navegador, Bluetooth, clona el repo, crea el entorno con las versiones de
   `uno_q/requirements-uno-q.txt`, activa 2 GB de swap y corre las pruebas.
3. Copiar `.env` (clave de Supabase) a `~/LSM_Coach/.env` si se quiere sincronizar; sin
   él todo funciona igual, solo local.
4. Arrancar: `bash ~/LSM_Coach/uno_q/run.sh` (abre el navegador en pantalla completa; al
   cerrarlo con Alt+F4 se apaga el servidor). `--instalar-autostart` lo deja arrancando
   solo al encender. `--sin-guante` para probar sin la muñequera.

En Linux `--cam` acepta `/dev/videoN` o `auto` (la cámara USB cambia de número entre
reinicios; `/dev/video0` y `1` son el códec del procesador, no cámaras).

## Calibrar con su cámara (muestras correctas y errores)

1. En la interfaz, abre **Calibración**, escribe el nombre de la persona y, con la muñequera
   puesta, pulsa **Guardar muestra correcta** (~20 por seña, mano derecha, bajando la mano
   entre capturas). Idealmente, personas que saben LSM.
2. Graba **errores intencionales** con **Guardar error intencional**: los errores típicos
   de cada letra (dedos separados en la B, C demasiado cerrada o abierta, índice a medio
   extender en la Y, etc.).
3. Entrena y mide:

```bash
python curate.py exp3:Y --aplicar                 # aparta muestras mal etiquetadas (persona:seña)
python evaluate_public.py --percentiles 1 99 --incluir-propias \
    --expertos invitado exp4 exp5 exp6 exp7 exp8 --guardar                 # rangos de los dedos
python train_classifier.py --publicos --errores --errores-de invitado exp4 exp8   # errores de las referencias
for s in A B C L Y; do python tolerance_calculator.py analyze --sign $s; done   # orientación (IMU)
python evaluate_accuracy.py --por-persona         # cifra honesta: deja fuera a cada persona
```

Los errores los definen personas de referencia que saben LSM (`--errores-de`; hoy invitado,
exp4 y exp8, que deben grabar los mismos errores típicos del reto): si cada
quien graba sus propios "errores", se contradicen (lo que uno graba como error otro lo hace
como correcto) y el modelo aprende cosas opuestas.

Pulsa **Recargar rangos** en la interfaz para usar lo nuevo.

`analyze` conserva los rangos de los dedos del dataset público: centrarlos en pocas personas
hacía que gente nueva fuera rechazada. Con `--centrar` se centran en las muestras propias.

## Datos públicos y clasificador

El reconocimiento de la forma de la mano usa un dataset público (ver Declaración de recursos).
Con el dataset descomprimido:

```bash
python import_msl.py --root ~/Downloads/datos_ent/MSL-ABC --signs A B C D E F G H I L M N O P R S T U V W Y   # ~13 min
python make_refs.py --root ~/Downloads/datos_ent            # imágenes de referencia -> web/ref/
python evaluate_public.py                                   # mide con personas no vistas
python evaluate_public.py --percentiles 1 99 --guardar      # rangos de dedos -> tolerances.json
python train_classifier.py --publicos --errores             # clasificador -> model.joblib
```

Una seña es correcta solo si **los 5 dedos están en rango**, **el clasificador la reconoce
como la letra más probable** y **no la clasifica como su error típico** (el modelo aprende
los errores intencionales grabados como clases "B_mal", etc.) y, si la seña tiene
orientación calibrada, **la muñeca está en rango**.

Resultados:
- Clasificador de las 21 letras estáticas con personas que no vio: 97.5 % (la V se confunde
  con la U en 14 % de los casos).
- Reglas + clasificador con personas del dataset no vistas: acepta el 94 % de las señas
  correctas y rechaza el 100 % de las otras letras.
- Errores intencionales de una persona que el modelo no vio (entrenando con los de otra):
  acepta el 98-100 % de sus señas correctas y rechaza el 73-85 % de sus errores
  (85-91 % de aciertos correctas vs. errores). Mejora al grabar errores de más personas.

## Letras con movimiento (Nivel 2: J, K, Ñ, Q, X, Z)

Se entrenan con los videos del dataset público de señas dinámicas (Zenodo
10.5281/zenodo.14689869: 621 videos frontales, 20 personas). El flujo:

```bash
python extract_dynamic.py --root ~/Downloads/datos_ent/MSL-dynamic-signs   # trayectorias -> samples_dynamic/ (~8 min)
python train_dynamic.py --por-persona                                      # model_dynamic.joblib
```

Cada video se remuestrea a 32 pasos con la forma de la mano, el recorrido de la muñeca, la
velocidad y el giro de la mano; un ExtraTrees clasifica el trazo. Además aprende una clase
**"quieta"** (la forma de la letra sostenida con temblor, sintetizada de cada video): sin
ella, bastaba poner la forma de la mano para que la diera por correcta. Resultado: **98 %** de
los trazos del split de prueba (personas nuevas) y **87 %** dejando fuera a cada persona;
las quietas se detectan al 100 %. Las confusiones quedan entre K, Q y X.

En la app (`dynamic.py`): con una letra con movimiento elegida, se espera a que la muñeca
(suavizada) supere ~1.6 tamaños de mano por segundo, se graba el trazo hasta que se detiene
0.45 s, sale de cuadro o pasan 3.5 s, y se clasifica. Antes de clasificar, el trazo debe
recorrer ≥ 0.8 tamaños de mano y abarcar ≥ 0.3 (el ruido de los landmarks recorre mucho pero
no se extiende). Si la letra pedida es la más probable con ≥ 0.25, cuenta como lograda; si
no, vibra una vez y dice a qué se pareció o que falta el movimiento. Las palabras se
practican deletreándolas (pestaña **Palabras**), con las letras estáticas.

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
- **Datasets públicos (CC BY 4.0):**
  - Morfín, R. *Mexican Sign Language Alphabet (static signs only)*. Zenodo.
    https://doi.org/10.5281/zenodo.10067508 — 21 letras estáticas: landmarks extraídos con
    MediaPipe para el clasificador y los rangos de los dedos; fotos de referencia.
  - Navarrete-López, J. A. y López-Nava, I. H. *Mexican Sign Language Alphabet (dynamic signs
    only)*. Zenodo. https://doi.org/10.5281/zenodo.14689869 — animaciones de referencia de
    J, K, Ñ, Q, X, Z.
- Los datos de calibración con nuestra cámara y la muñequera se graban durante el evento.
