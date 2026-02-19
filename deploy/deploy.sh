#!/bin/bash
# =============================================================
# Deploy-Script: Laptop → Hetzner VPS
# Usage: ./deploy/deploy.sh <SERVER-IP>
# =============================================================
set -e

SERVER_IP="${1:?Fehler: Server-IP angeben! Usage: ./deploy.sh 123.45.67.89}"
BOT_USER="botuser"
BOT_DIR="/home/$BOT_USER/trading-bot"

echo "================================================"
echo "  Deploy nach $SERVER_IP"
echo "================================================"

# Code per rsync hochladen (ohne Secrets, DB, Cache)
echo "[1/4] Code hochladen..."
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude 'config/secrets.env' \
    --exclude '*.db' \
    --exclude '*.sqlite*' \
    --exclude '*.log' \
    --exclude 'logs/' \
    --exclude 'models/' \
    --exclude '*.pkl' \
    --exclude '*.pt' \
    --exclude '.DS_Store' \
    --exclude 'bot.pid' \
    --exclude 'deploy/' \
    -e ssh \
    /Users/nici/Projects/trading-bot/ \
    root@${SERVER_IP}:${BOT_DIR}/

# Ownership fixen
echo "[2/4] Berechtigungen setzen..."
ssh root@${SERVER_IP} "chown -R ${BOT_USER}:${BOT_USER} ${BOT_DIR}"

# Venv + Dependencies installieren
echo "[3/4] Dependencies installieren..."
ssh root@${SERVER_IP} "su - ${BOT_USER} -c '
    cd ${BOT_DIR}
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt --no-cache-dir
'"

# Bot neustarten
echo "[4/4] Bot neustarten..."
ssh root@${SERVER_IP} "systemctl restart trading-bot"

# Status pruefen
echo ""
echo "================================================"
echo "  Deploy fertig! Status:"
echo "================================================"
ssh root@${SERVER_IP} "systemctl status trading-bot --no-pager -l | head -15"

echo ""
echo "Nuetzliche Befehle:"
echo "  Logs live:    ssh root@${SERVER_IP} 'journalctl -u trading-bot -f'"
echo "  Status:       ssh root@${SERVER_IP} 'systemctl status trading-bot'"
echo "  Stop:         ssh root@${SERVER_IP} 'systemctl stop trading-bot'"
echo "  Restart:      ssh root@${SERVER_IP} 'systemctl restart trading-bot'"
echo ""
