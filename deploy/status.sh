#!/bin/bash
# =============================================================
# Quick Status-Check vom Laptop aus
# Usage: ./deploy/status.sh <SERVER-IP>
# =============================================================

SERVER_IP="${1:?Fehler: Server-IP angeben! Usage: ./status.sh 123.45.67.89}"

echo "================================================"
echo "  Bot Status auf $SERVER_IP"
echo "================================================"
echo ""

# Service Status
echo "--- Service ---"
ssh root@${SERVER_IP} "systemctl is-active trading-bot"
echo ""

# Uptime + Memory
echo "--- Server ---"
ssh root@${SERVER_IP} "uptime && free -h | head -2"
echo ""

# Letzte 20 Log-Zeilen
echo "--- Letzte Logs ---"
ssh root@${SERVER_IP} "journalctl -u trading-bot --no-pager -n 20"
echo ""
