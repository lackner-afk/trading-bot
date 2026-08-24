#!/bin/bash
# =============================================================
# Trading-Bot als LaunchAgent auf dem Mac verwalten.
#
#   ./deploy/mac/launchagent.sh install    einmalig einrichten + starten
#   ./deploy/mac/launchagent.sh start      starten
#   ./deploy/mac/launchagent.sh stop       stoppen (bleibt installiert)
#   ./deploy/mac/launchagent.sh status     laeuft er?
#   ./deploy/mac/launchagent.sh uninstall  komplett entfernen
#
# Warum ueberhaupt: ohne LaunchAgent stirbt der Bot mit dem Terminal und
# niemand startet ihn neu. KeepAlive startet ihn nach Absturz und Reboot,
# caffeinate verhindert das Einfrieren im Idle-Sleep.
# =============================================================
set -euo pipefail

LABEL="at.silvertoast.tradingbot"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SRC="${PROJ}/deploy/mac/${LABEL}.plist"
DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

case "${1:-}" in
  install)
    [ -f "${PROJ}/config/secrets.env" ] || { echo "FEHLER: config/secrets.env fehlt."; exit 1; }
    mkdir -p "${HOME}/Library/LaunchAgents"
    # Pfad zur Laufzeit einsetzen, damit ein verschobenes Repo nicht still bricht
    sed "s|<string>/Users/[^<]*trading-bot</string>|<string>${PROJ}</string>|g; \
         s|<string>/Users/[^<]*trading-bot/bot_console.log</string>|<string>${PROJ}/bot_console.log</string>|g" \
        "${SRC}" > "${DEST}"
    plutil -lint "${DEST}" >/dev/null
    launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/${UID_NUM}" "${DEST}"
    echo "Installiert und gestartet: ${LABEL}"
    echo "Logs:  tail -f ${PROJ}/bot.log"
    ;;
  start)
    launchctl bootstrap "gui/${UID_NUM}" "${DEST}" 2>/dev/null || launchctl kickstart "gui/${UID_NUM}/${LABEL}"
    echo "Gestartet."
    ;;
  stop)
    launchctl bootout "gui/${UID_NUM}/${LABEL}"
    echo "Gestoppt (bleibt installiert, startet beim naechsten Login wieder)."
    ;;
  status)
    if launchctl print "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1; then
      launchctl print "gui/${UID_NUM}/${LABEL}" | grep -E "state|pid|last exit" | sed 's/^[[:space:]]*/  /'
    else
      echo "  nicht geladen"
    fi
    pgrep -fl "python3 main.py" || echo "  kein Bot-Prozess"
    ;;
  uninstall)
    launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
    rm -f "${DEST}"
    echo "Entfernt."
    ;;
  *)
    sed -n '3,12p' "$0"; exit 1
    ;;
esac
