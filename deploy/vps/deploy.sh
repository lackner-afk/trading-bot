#!/bin/bash
# =============================================================
# Paper-Bot (mit Panda-Pro-Dashboard) und Datensammler auf den VPS bringen.
#   ./deploy/vps/deploy.sh          (aus dem Repo-Wurzelverzeichnis)
#
# Secrets liegen NUR auf dem Server und werden nie übertragen:
#   config/secrets.env     BITPANDA_API_KEY=... (Read-only-Key reicht)
#   config/dashboard.env   DASHBOARD_USER / DASHBOARD_PASSWORD
# Fehlt dashboard.env, legt dieses Skript sie mit Zufallspasswort an.
# =============================================================
set -euo pipefail

SERVER="deploy@91.98.234.212"
APP_DIR="/home/deploy/trading-bot"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Prüfen …"
python3 -c "import ast; [ast.parse(open(f).read(), f) for f in ['$ROOT/main.py', '$ROOT/dashboard/server.py', '$ROOT/tools/data_recorder.py']]"
grep -q "^  mode: paper" "$ROOT/config/settings.yaml" || { echo "FEHLER: settings.yaml steht nicht auf mode: paper — Abbruch."; exit 1; }

echo "==> Dateien übertragen …"
ssh "$SERVER" "mkdir -p $APP_DIR/config"
# --delete: Gelöschtes darf auf dem Server nicht liegen bleiben.
# Ausgenommen: Secrets, Laufzeitdaten, Mac-Spezifisches.
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude 'venv/' --exclude '.venv/' \
  --exclude 'config/secrets.env' --exclude 'config/dashboard.env' \
  --exclude '*.db' --exclude '*.log' --exclude '*.pkl' --exclude 'bot.pid' \
  --exclude 'models/' --exclude 'logs/' \
  "$ROOT/" "$SERVER:$APP_DIR/"

echo "==> Bauen und starten …"
ssh "$SERVER" 'bash -s' << 'REMOTE'
set -euo pipefail
cd /home/deploy/trading-bot
umask 077
[ -f config/secrets.env ] || { echo "# BITPANDA_API_KEY=..." > config/secrets.env; echo "  config/secrets.env angelegt (ohne Key → Kraken-Kurse)"; }
if [ ! -f config/dashboard.env ]; then
  PW="$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 24)"
  printf 'DASHBOARD_USER=nici\nDASHBOARD_PASSWORD=%s\n' "$PW" > config/dashboard.env
  echo "  config/dashboard.env angelegt — Passwort: $PW"
fi
cd deploy/vps
docker compose up -d --build --remove-orphans
# Nur verwaiste Images dieses Baus — andere Projekte bleiben unberührt.
docker image prune -f > /dev/null
for i in $(seq 1 30); do
  if curl -fsS --max-time 3 http://127.0.0.1:3400/health > /dev/null; then
    echo "==> Dashboard antwortet auf 127.0.0.1:3400"
    docker compose ps --format 'table {{.Name}}\t{{.Status}}'
    exit 0
  fi
  sleep 1
done
echo "FEHLER: Das Dashboard antwortet nicht."
docker compose logs --tail=40
exit 1
REMOTE
