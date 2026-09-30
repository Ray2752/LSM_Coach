#!/usr/bin/env bash
# Arranca LSM Coach en la placa (Arduino UNO Q o Raspberry Pi 5): servidor (cámara + visión +
# guante por BLE) y el navegador en pantalla completa (modo quiosco). Al cerrar el navegador
# se apaga el servidor.
#
#   bash ~/LSM_Coach/uno_q/run.sh                       # demo
#   bash ~/LSM_Coach/uno_q/run.sh --sin-guante          # pruebas sin la muñequera
#   bash ~/LSM_Coach/uno_q/run.sh --seguir              # con la cámara motorizada (servos, solo UNO Q)
#   bash ~/LSM_Coach/uno_q/run.sh --instalar-autostart  # que arranque sola al encender
#   bash ~/LSM_Coach/uno_q/run.sh --modelo-ligero       # en la Pi: manos con el modelo ligero (más fps)
#   bash ~/LSM_Coach/uno_q/run.sh --ligero --res=480x360   # en la Pi: forzar el modo de la UNO Q
#   bash ~/LSM_Coach/uno_q/run.sh --completo            # en la UNO Q: sin modo ligero (lento)
set -uo pipefail

DIR="$HOME/LSM_Coach"
URL="http://localhost:8000"
IMU="ble"

# Raspberry Pi 5: 3-4 veces más rápida que la UNO Q y con GPU en el navegador: va sin modo ligero
# y con el modelo completo de manos (más preciso). En la UNO Q, 480x360 + ligero dan ~7 fps.
if grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  RES="640x480"; LITE=""; MODEL=""
else
  RES="480x360"; LITE="--lite"; MODEL="--modelo-ligero"
fi

if [ "${1:-}" = "--instalar-autostart" ]; then
  # ~/.config/autostart lo respetan el escritorio de la UNO Q y Raspberry Pi OS (Wayfire y labwc
  # lo procesan con lxsession-xdg-autostart). Los 5 s dan tiempo a que la cámara y el BT estén.
  mkdir -p "$HOME/.config/autostart"
  cat > "$HOME/.config/autostart/lsm-coach.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LSM Coach
Exec=bash -c "sleep 5; bash $DIR/uno_q/run.sh"
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
    --ligero) LITE="--lite" ;;             # video e interfaz ligeros (navegador sin GPU)
    --modelo-ligero) MODEL="--modelo-ligero" ;;   # MediaPipe complexity 0 (97 ms vs 162 ms en la UNO Q)
    --completo) LITE=""; MODEL="" ;;
    --res=*) RES="${arg#--res=}" ;;        # p. ej. --res=480x360
  esac
done

cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

# Si se lanza por SSH no hay pantalla asignada: se usa el escritorio abierto en el monitor.
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
# Raspberry Pi OS usa Wayland: el navegador necesita saber dónde está el escritorio.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -S "$XDG_RUNTIME_DIR/wayland-0" ]; then
  export WAYLAND_DISPLAY=wayland-0
fi

# Que la pantalla no se apague ni bloquee durante la demo ("Power Saving Mode" del monitor)
if command -v gsettings >/dev/null 2>&1; then
  gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null
  gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type nothing 2>/dev/null
fi
command -v xset >/dev/null 2>&1 && xset s off -dpms 2>/dev/null

# --cam auto: la cámara USB cambia de /dev/videoN entre reinicios.
# shellcheck disable=SC2086
python web_server.py --cam auto --res "$RES" --imu "$IMU" $LITE $MODEL $EXTRA &
SERVER=$!

# espera a que el servidor responda (máx. 60 s: MediaPipe tarda en cargar la primera vez)
for _ in $(seq 1 60); do
  curl -fs "$URL/api/signs" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "El servidor no arrancó"; exit 1; }
  sleep 1
done

PAGE="$URL/"
[ -n "$LITE" ] && PAGE="$URL/?lite=1"
BROWSER=$(command -v chromium || command -v chromium-browser || command -v firefox-esr || command -v firefox)
# nice: la visión (MediaPipe) tiene prioridad sobre el navegador. Las banderas de GPU usan la
# aceleración si existe (Pi 5: sí; UNO Q: cae a software sin fallar). ozone auto: Wayland o X11.
if [[ "$BROWSER" == *chromium* ]]; then
  nice -n 5 "$BROWSER" --kiosk --noerrdialogs --disable-infobars --no-first-run --disable-session-crashed-bubble \
             --autoplay-policy=no-user-gesture-required --ignore-gpu-blocklist --enable-gpu-rasterization \
             --enable-zero-copy --disable-smooth-scrolling --ozone-platform-hint=auto "$PAGE"
else
  nice -n 5 "$BROWSER" --kiosk "$PAGE"
fi

kill $SERVER 2>/dev/null
wait $SERVER 2>/dev/null
