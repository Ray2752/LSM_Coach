#!/usr/bin/env bash
# UNO Q como puente de los servos (cámara motorizada) cuando la visión corre en la Raspberry Pi:
# recibe las órdenes por la red del hotspot y las pasa a la MCU (uno_q/pan_tilt/sketch.ino).
#
#   bash ~/LSM_Coach/uno_q/servos.sh                      # en una terminal (Ctrl+C para parar)
#   bash ~/LSM_Coach/uno_q/servos.sh --instalar-servicio  # que arranque solo al encender
#   bash ~/LSM_Coach/uno_q/servos.sh --barrido            # prueba: mueve los servos
set -uo pipefail
DIR="$HOME/LSM_Coach"

if [ "${1:-}" = "--instalar-servicio" ]; then
  sudo tee /etc/systemd/system/lsm-servos.service >/dev/null <<EOF
[Unit]
Description=LSM Coach: puente de servos (UNO Q)
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python $DIR/pan_tilt_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now lsm-servos.service
  echo "Listo: el puente de servos arranca al encender. Ver:  journalctl -u lsm-servos -f"
  echo "Para quitarlo:  sudo systemctl disable --now lsm-servos"
  exit 0
fi

cd "$DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "${1:-}" = "--barrido" ]; then
  exec python pan_tilt.py --barrido
fi
exec python pan_tilt_server.py
