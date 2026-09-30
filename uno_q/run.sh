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
EXTRA=""
for arg in "$@"; do
  case "$arg" in
    --sin-guante) IMU="none" ;;
    --seguir) EXTRA="$EXTRA --seguir" ;;   # cámara motorizada (servos en D9/D10, uno_q/pan_tilt)
  esac
done

cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

# --cam auto: la cámara USB cambia de /dev/videoN entre reinicios. 640x480 alivia al procesador.
# shellcheck disable=SC2086
python web_server.py --cam auto --res 640x480 --imu "$IMU" $EXTRA &
SERVER=$!

# espera a que el servidor responda (máx. 60 s: MediaPipe tarda en cargar la primera vez)
for _ in $(seq 1 60); do
  curl -fs "$URL/api/signs" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "El servidor no arrancó"; exit 1; }
  sleep 1
done

BROWSER=$(command -v chromium || command -v chromium-browser || command -v firefox-esr || command -v firefox)
if [[ "$BROWSER" == *chromium* ]]; then
  "$BROWSER" --kiosk --noerrdialogs --disable-infobars --no-first-run --disable-session-crashed-bubble \
             --autoplay-policy=no-user-gesture-required "$URL"
else
  "$BROWSER" --kiosk "$URL"
fi

kill $SERVER 2>/dev/null
wait $SERVER 2>/dev/null
