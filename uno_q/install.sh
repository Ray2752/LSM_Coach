#!/usr/bin/env bash
# Instala LSM Coach en la placa de la demo: Arduino UNO Q (Debian ARM64) o Raspberry Pi 5
# (Raspberry Pi OS de 64 bits). Se corre UNA vez, en una terminal de la placa con internet:
#
#   curl -fsSL https://raw.githubusercontent.com/Ray2752/LSM_Coach/main/uno_q/install.sh | bash
#
# Tarda ~10-20 min (descarga ~400 MB). Al final corre las pruebas y lista las cámaras.
# Se puede volver a correr sin problema: cada paso salta lo que ya está hecho.
set -euo pipefail

REPO="https://github.com/Ray2752/LSM_Coach.git"
DIR="$HOME/LSM_Coach"
# Sin cuadros de diálogo de apt (p. ej. "qué servicios reiniciar"): por adb las flechas no
# funcionan y el instalador se quedaba esperando. Los servicios se reinician solos.
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1

if [ "$(uname -m)" != "aarch64" ]; then
  echo "Se necesita un sistema de 64 bits (aarch64) y este es $(uname -m)." >&2
  echo "En la Raspberry Pi hay que grabar 'Raspberry Pi OS (64-bit)' con Raspberry Pi Imager." >&2
  exit 1
fi
if grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then PLACA="Raspberry Pi"; else PLACA="UNO Q"; fi
echo "Placa: $PLACA ($(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo desconocida))"

echo "== 1/7 Paquetes del sistema (git, navegador, bluetooth, v4l2, librerías de OpenCV)"
# Algunas redes (hotspot de celular, campus) bloquean HTTP sin cifrar y apt recibe "403
# Forbidden" de deb.debian.org. Los repositorios de Debian y de Raspberry Pi aceptan HTTPS.
sudo sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g; s|http://archive.raspberrypi.com|https://archive.raspberrypi.com|g' \
  /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true
sudo apt-get update
sudo apt-get install -y git curl v4l-utils bluez x11-xserver-utils
# el nombre de libglib cambia entre versiones de Debian (bookworm: libglib2.0-0, trixie: -0t64)
sudo apt-get install -y libgl1 libglib2.0-0 libportaudio2 || sudo apt-get install -y libgl1 libglib2.0-0t64 libportaudio2
sudo apt-get install -y chromium || sudo apt-get install -y firefox-esr
sudo usermod -aG video,bluetooth "$USER" || true
sudo systemctl enable --now bluetooth || true

echo "== 2/7 Python 3.12 (la placa trae otra versión y MediaPipe en ARM solo existe hasta 3.12)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

echo "== 3/7 Código"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"

echo "== 4/7 Entorno y dependencias"
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r uno_q/requirements-uno-q.txt

echo "== 5/7 Memoria de intercambio (escritorio + navegador + visión van justos con 2-4 GB)"
if [ "$(awk '/MemTotal/{print $2}' /proc/meminfo)" -lt 6000000 ] && ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== 6/7 Ajustes de la placa (procesador a tope, sin apagado de pantalla, sin blueman)"
# blueman (gestor de Bluetooth del escritorio) se comía un núcleo entero en la UNO Q.
if dpkg -s blueman >/dev/null 2>&1; then
  pkill -f blueman || true
  sudo apt-get remove -y blueman || true
fi
# Gobernador "performance": la visión no espera a que el procesador suba de frecuencia.
sudo tee /etc/systemd/system/cpu-performance.service >/dev/null <<'EOF'
[Unit]
Description=CPU governor performance (LSM Coach)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $$g; done'

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-performance.service || true
# Raspberry Pi OS (Wayland): que la pantalla no se apague a los 10 min de demo.
if command -v raspi-config >/dev/null 2>&1; then
  sudo raspi-config nonint do_blanking 1 || true
fi

echo "== 7/7 Verificación"
python -c "import mediapipe as mp, cv2, sklearn, bleak; print('MediaPipe', mp.__version__, '| OpenCV', cv2.__version__, '| sklearn', sklearn.__version__)"
python -m unittest discover -s tests -t .
python -c "import joblib; m = joblib.load('model.joblib'); print('Modelo OK:', len(m.classes_), 'clases')"
echo
echo "Cámaras conectadas:"
python web_server.py --list-cams || true

echo
echo "Listo. Para arrancar la demo:   bash ~/LSM_Coach/uno_q/run.sh"
echo "Para medir la velocidad de la visión:   python ~/LSM_Coach/uno_q/bench.py"
echo "Para que arranque sola al encender:   bash ~/LSM_Coach/uno_q/run.sh --instalar-autostart"
echo "Si el usuario acaba de entrar a los grupos video/bluetooth, cierra sesión y vuelve a entrar."
