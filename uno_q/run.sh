#!/usr/bin/env bash
# Arranca LSM Coach en la UNO Q: servidor (cámara + visión + guante por BLE) y el navegador
# en pantalla completa (modo quiosco). Al cerrar el navegador se apaga el servidor.
#
#   bash ~/LSM_Coach/uno_q/run.sh                       # demo
#   bash ~/LSM_Coach/uno_q/run.sh --sin-guante          # pruebas sin la muñequera
#   bash ~/LSM_Coach/uno_q/run.sh --seguir              # con la cámara motorizada (servos)
#   bash ~/LSM_Coach/uno_q/run.sh --instalar-autostart  # que arranque sola al encender
set -uo pipefail

DIR="$HOME/LSM_Coach"
URL="http://localhost:8000"
IMU="ble"

if [ "${1:-}" = "--instalar-autostart" ]; then
  mkdir -p "$HOME/.config/autostart"
  cat > "$HOME/.config/autostart/lsm-coach.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LSM Coach
Exec=bash $DIR/uno_q/run.sh
X-GNOME-Autostart-Delay=8
EOF
  echo "Listo: LSM Coach arrancará al iniciar sesión. Para quitarlo: rm ~/.config/autostart/lsm-coach.desktop"
  exit 0
fi
EXTRA=""; RES=""
for arg in "$@"; do
  case "$arg" in
    --sin-guante) IMU="none" ;;
    --seguir) EXTRA="$EXTRA --seguir" ;;   # cámara motorizada (servos en D9/D10, uno_q/pan_tilt)
    --modelo-ligero) ;;                    # ya va por defecto (ver abajo)
    --res=*) RES="${arg#--res=}" ;;        # p. ej. --res=480x360
  esac
done

cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

# Si se lanza por SSH no hay DISPLAY: se usa la pantalla principal del monitor
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# Que la pantalla no se apague ni bloquee durante la demo ("Power Saving Mode" del monitor)
if command -v gsettings >/dev/null 2>&1; then
  gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null
  gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type nothing 2>/dev/null
fi
command -v xset >/dev/null 2>&1 && xset s off -dpms 2>/dev/null

# --cam auto: la cámara USB cambia de /dev/videoN entre reinicios. 640x480 alivia al procesador.
# --lite: video MJPEG más ligero e interfaz sin efectos (el navegador de la UNO Q no tiene GPU).
# shellcheck disable=SC2086
# --modelo-ligero: manos con el modelo ligero de MediaPipe (97 ms vs 162 ms por fotograma en la UNO Q)
python web_server.py --cam auto --res "${RES:-640x480}" --imu "$IMU" --lite --modelo-ligero $EXTRA &
SERVER=$!

# espera a que el servidor responda (máx. 60 s: MediaPipe tarda en cargar la primera vez)
for _ in $(seq 1 60); do
  curl -fs "$URL/api/signs" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "El servidor no arrancó"; exit 1; }
  sleep 1
done

BROWSER=$(command -v chromium || command -v chromium-browser || command -v firefox-esr || command -v firefox)
# nice: la visión (MediaPipe) tiene prioridad sobre el navegador. Las banderas de GPU intentan
# usar la aceleración del QRB2210 (si no existe, Chromium cae a software sin fallar).
if [[ "$BROWSER" == *chromium* ]]; then
  nice -n 5 "$BROWSER" --kiosk --noerrdialogs --disable-infobars --no-first-run --disable-session-crashed-bubble \
             --autoplay-policy=no-user-gesture-required --ignore-gpu-blocklist --enable-gpu-rasterization \
             --enable-zero-copy --disable-smooth-scrolling "$URL/?lite=1"
else
  nice -n 5 "$BROWSER" --kiosk "$URL/?lite=1"
fi

kill $SERVER 2>/dev/null
wait $SERVER 2>/dev/null
