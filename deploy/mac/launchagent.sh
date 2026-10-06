#!/bin/bash
# =============================================================
# Trading-Bot und Datensammler als LaunchAgents auf dem Mac verwalten.
#
#   ./deploy/mac/launchagent.sh install  [bot|recorder|all]
#   ./deploy/mac/launchagent.sh start    [bot|recorder|all]
#   ./deploy/mac/launchagent.sh stop     [bot|recorder|all]
#   ./deploy/mac/launchagent.sh status   [bot|recorder|all]
#   ./deploy/mac/launchagent.sh uninstall [bot|recorder|all]
#
# Ohne zweites Argument gilt "all".
#
#   bot       — der Trading-Bot selbst
#   recorder  — sammelt Orderbuch- und Orderflow-Daten (tools/data_recorder.py).
#               Laeuft unabhaengig vom Bot: die Daten sind historisch nicht
#               beschaffbar, jede Stunde ohne Sammler ist dauerhaft verloren.
#
# Beide starten bei Login und nach jedem Ende neu (KeepAlive) und halten den Mac
# mit caffeinate wach — sonst friert macOS den Prozess im Idle ein.
# =============================================================
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
UID_NUM="$(id -u)"
ACTION="${1:-}"
TARGET="${2:-all}"

case "${TARGET}" in
  bot)      LABELS=("at.silvertoast.tradingbot") ;;
  recorder) LABELS=("at.silvertoast.datarecorder") ;;
  all)      LABELS=("at.silvertoast.tradingbot" "at.silvertoast.datarecorder") ;;
  *)        echo "Unbekanntes Ziel: ${TARGET} (bot|recorder|all)"; exit 1 ;;
esac

install_one() {
  local label="$1"
  local src="${PROJ}/deploy/mac/${label}.plist"
  local dest="${HOME}/Library/LaunchAgents/${label}.plist"
  [ -f "${src}" ] || { echo "FEHLER: ${src} fehlt."; exit 1; }
  mkdir -p "${HOME}/Library/LaunchAgents"
  # Projektpfad zur Laufzeit einsetzen, damit ein verschobenes Repo nicht still bricht
  sed "s|<string>/Users/[^<]*trading-bot\(/[^<]*\)\?</string>|<string>${PROJ}\1</string>|g" \
      "${src}" > "${dest}"
  plutil -lint "${dest}" >/dev/null
  launchctl bootout "gui/${UID_NUM}/${label}" 2>/dev/null || true
  launchctl bootstrap "gui/${UID_NUM}" "${dest}"
  echo "  installiert und gestartet: ${label}"
}

case "${ACTION}" in
  install)
    [ -f "${PROJ}/config/secrets.env" ] || { echo "FEHLER: config/secrets.env fehlt."; exit 1; }
    for l in "${LABELS[@]}"; do install_one "$l"; done
    echo
    echo "Logs:  tail -f ${PROJ}/bot.log"
    echo "       tail -f ${PROJ}/recorder.log"
    ;;
  start)
    for l in "${LABELS[@]}"; do
      launchctl bootstrap "gui/${UID_NUM}" "${HOME}/Library/LaunchAgents/${l}.plist" 2>/dev/null \
        || launchctl kickstart "gui/${UID_NUM}/${l}"
      echo "  gestartet: ${l}"
    done
    ;;
  stop)
    for l in "${LABELS[@]}"; do
      launchctl bootout "gui/${UID_NUM}/${l}" 2>/dev/null && echo "  gestoppt: ${l}" \
        || echo "  war nicht geladen: ${l}"
    done
    ;;
  status)
    for l in "${LABELS[@]}"; do
      echo "${l}:"
      if launchctl print "gui/${UID_NUM}/${l}" >/dev/null 2>&1; then
        launchctl print "gui/${UID_NUM}/${l}" | grep -E "^\s+(state|pid|last exit code)" | sed 's/^[[:space:]]*/    /'
      else
        echo "    nicht geladen"
      fi
    done
    echo "Prozesse:"
    pgrep -fl "python3 main.py|data_recorder.py" | sed 's/^/    /' || echo "    keine"
    if [ -f "${PROJ}/market_data.db" ]; then
      echo "Datensammlung:"
      sqlite3 "${PROJ}/market_data.db" \
        "SELECT '    ' || COUNT(*) || ' Orderbuch-Zeilen über ' ||
                round((MAX(ts)-MIN(ts))/86400.0, 1) || ' Tage' FROM orderbook;" 2>/dev/null \
        || echo "    noch keine Daten"
    fi
    ;;
  uninstall)
    for l in "${LABELS[@]}"; do
      launchctl bootout "gui/${UID_NUM}/${l}" 2>/dev/null || true
      rm -f "${HOME}/Library/LaunchAgents/${l}.plist"
      echo "  entfernt: ${l}"
    done
    ;;
  *)
    sed -n '3,17p' "$0"; exit 1
    ;;
esac
