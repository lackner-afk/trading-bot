#!/bin/bash
# =============================================================
# Live Trading Bot Status Check (Phase 6)
# Usage: ./deploy/status.sh <SERVER-IP>
#
# Zeigt:
# - Service Status
# - Ob der Bot im LIVE MODE läuft (wichtig!)
# - Letzte kritische Fehler / Reconciliation
# - Ressourcen
# - Letzte Logs (mit Fokus auf Live-Events)
# =============================================================

SERVER_IP="${1:?Fehler: Server-IP angeben! Usage: ./status.sh 123.45.67.89}"

echo "================================================"
echo "  🚀 TRADING-BOT LIVE STATUS  |  $SERVER_IP"
echo "================================================"
echo ""

# 1. Service Status
echo "=== SERVICE STATUS ==="
SERVICE_STATUS=$(ssh root@${SERVER_IP} "systemctl is-active trading-bot")
echo "Status: $SERVICE_STATUS"

if [ "$SERVICE_STATUS" != "active" ]; then
    echo "⚠️  WARNUNG: Service ist NICHT aktiv!"
fi
echo ""

# 2. Erkenne LIVE MODE im Log (sehr wichtig für Sicherheit)
echo "=== LIVE MODE CHECK ==="
LIVE_LINES=$(ssh root@${SERVER_IP} "journalctl -u trading-bot --no-pager -n 100 | grep -i 'LIVE MODE' | tail -5" || echo "")
if [ -n "$LIVE_LINES" ]; then
    echo "🔴 LIVE MODE wurde kürzlich aktiviert:"
    echo "$LIVE_LINES"
else
    echo "✅ Kein aktiver LIVE MODE in den letzten 100 Logzeilen gefunden (vermutlich Paper)."
fi
echo ""

# 3. Letzte Reconciliation (Phase 3/4)
echo "=== LETZTE RECONCILIATION ==="
ssh root@${SERVER_IP} "journalctl -u trading-bot --no-pager -n 50 | grep -i 'Reconciliation' | tail -3" || echo "Keine Reconciliation Logs gefunden"
echo ""

# 4. Server Ressourcen
echo "=== SERVER RESSOURCEN ==="
ssh root@${SERVER_IP} "uptime && echo '--- Memory ---' && free -h | head -2"
echo ""

# 5. Letzte kritische / Error Logs (live-relevant)
echo "=== LETZTE KRITISCHE / ERROR LOGS ==="
ssh root@${SERVER_IP} "journalctl -u trading-bot --no-pager -p err -n 15 --no-hostname" || echo "Keine Fehler in den letzten Logs"
echo ""

# 6. Letzte 30 allgemeine Logs (kompakt)
echo "=== LETZTE LOGS (letzte 30 Zeilen) ==="
ssh root@${SERVER_IP} "journalctl -u trading-bot --no-pager -n 30 --no-hostname | tail -30"
echo ""

echo "================================================"
echo "Nützliche Live-Befehle:"
echo "  Live Logs:        ssh root@${SERVER_IP} 'journalctl -u trading-bot -f'"
echo "  Nur Errors:       ssh root@${SERVER_IP} 'journalctl -u trading-bot -p err -f'"
echo "  Bot stoppen:      ssh root@${SERVER_IP} 'systemctl stop trading-bot'"
echo "  Sofort neustart:  ssh root@${SERVER_IP} 'systemctl restart trading-bot'"
echo "================================================"
