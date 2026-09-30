#!/usr/bin/env bash
# Instala LSM Coach en la Arduino UNO Q (Debian, ARM64). Se corre UNA vez, en una terminal
# de la propia UNO Q (modo computadora: hub + pantalla + teclado) con internet:
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

echo "== 1/6 Paquetes del sistema (git, navegador, bluetooth, v4l2, librerías de OpenCV)"
# Algunas redes (hotspot de celular, campus) bloquean HTTP sin cifrar y apt recibe "403
# Forbidden" de deb.debian.org. Los repositorios de Debian aceptan HTTPS: se cambian.
sudo sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' \
  /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true
sudo apt-get update
sudo apt-get install -y git curl v4l-utils bluez libportaudio2 libgl1 libglib2.0-0
sudo apt-get install -y chromium || sudo apt-get install -y firefox-esr
sudo usermod -aG video,bluetooth "$USER" || true
sudo systemctl enable --now bluetooth || true

echo "== 2/6 Python 3.12 (la UNO Q trae 3.13 y MediaPipe no lo soporta en ARM)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

echo "== 3/6 Código"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"

echo "== 4/6 Entorno y dependencias"
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r uno_q/requirements-uno-q.txt

echo "== 5/6 Memoria de intercambio (la UNO Q de 2 GB va justa con escritorio + navegador + visión)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== 6/6 Verificación"
python -c "import mediapipe as mp, cv2, sklearn, bleak; print('MediaPipe', mp.__version__, '| OpenCV', cv2.__version__, '| sklearn', sklearn.__version__)"
python -m unittest discover -s tests -t .
python -c "import joblib; m = joblib.load('model.joblib'); print('Modelo OK:', len(m.classes_), 'clases')"
echo
echo "Cámaras conectadas:"
python web_server.py --list-cams || true

echo
echo "Listo. Para arrancar la demo:   bash ~/LSM_Coach/uno_q/run.sh"
echo "Para que arranque sola al encender:   bash ~/LSM_Coach/uno_q/run.sh --instalar-autostart"
echo "Si el usuario acaba de entrar a los grupos video/bluetooth, cierra sesión y vuelve a entrar."
