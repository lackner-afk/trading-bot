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
Description=Trading-Bot (Bitpanda Fusion / One Trading) - Paper + Live Mode supported
After=network.target

# Automatischer Neustart bei Crash, aber max 5x in 5 Min.
# Diese beiden Keys gehoeren in [Unit] — vorher standen sie in [Service],
# wo systemd sie als unbekannt ignoriert hat. Das Rate-Limit griff also nicht.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/trading-bot
ExecStart=/home/botuser/trading-bot/venv/bin/python3 main.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

# Der Bot faehrt jetzt geordnet herunter (Stop-Event statt blindem sleep),
# storniert offene Orders und schreibt den Abschlussreport. 30s reichen dafuer.
TimeoutStopSec=30

# Environment
Environment=PYTHONUNBUFFERED=1
# Für Live-Modus: Hier kannst du LIVE_TRADING_ENABLED=1 setzen (aber NUR nach mehrfacher Bestätigung!)
# Environment=LIVE_TRADING_ENABLED=1

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
