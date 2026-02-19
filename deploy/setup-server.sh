#!/bin/bash
# =============================================================
# Hetzner VPS Setup für Paper-Trading-Bot
# Einmal auf dem Server ausführen nach erstem Login
# =============================================================
set -e

echo "================================================"
echo "  Paper-Trading-Bot - Server Setup"
echo "================================================"

# System updaten
echo "[1/5] System Update..."
apt update && apt upgrade -y

# Python 3.11 + Tools installieren
echo "[2/5] Python installieren..."
apt install -y python3 python3-pip python3-venv git tmux htop

# Bot-User anlegen (nicht als root laufen)
echo "[3/5] Bot-User anlegen..."
if ! id "botuser" &>/dev/null; then
    useradd -m -s /bin/bash botuser
    echo "User 'botuser' erstellt"
else
    echo "User 'botuser' existiert bereits"
fi

# Projekt-Verzeichnis vorbereiten
echo "[4/5] Verzeichnisse anlegen..."
BOT_DIR="/home/botuser/trading-bot"
mkdir -p "$BOT_DIR"
mkdir -p /home/botuser/logs

# Systemd Service erstellen
echo "[5/5] Systemd Service erstellen..."
cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=Paper-Trading-Bot (Momentum)
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/trading-bot
ExecStart=/home/botuser/trading-bot/venv/bin/python3 main.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

# Automatischer Neustart bei Crash, aber max 5x in 5 Min
StartLimitIntervalSec=300
StartLimitBurst=5

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "================================================"
echo "  Server-Setup fertig!"
echo "================================================"
echo ""
echo "Naechste Schritte:"
echo "  1. Code deployen: ./deploy.sh <SERVER-IP>"
echo "  2. Secrets anlegen: ssh botuser@<IP>"
echo "     nano ~/trading-bot/config/secrets.env"
echo "  3. Bot starten: ssh root@<IP>"
echo "     systemctl start trading-bot"
echo ""
