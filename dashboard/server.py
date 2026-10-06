#!/usr/bin/env python3
"""
Dashboard für den Paper-Bot — nur lesend.

Liest, was Bot und Datensammler ins gemeinsame Verzeichnis schreiben:
  status.json      Zustand des Bots (alle 15 s, siehe main.py _status_loop)
  equity.jsonl     Equity-Verlauf (alle 5 min)
  trades.db        abgeschlossene Trades
  bot.log          Log des Bots
  market_data.db   Orderbuch-/Orderflow-Sammlung (tools/data_recorder.py)

Es gibt keinen Knopf, der den Bot steuert: wer das Passwort errät, sieht
Zahlen, kann aber nichts auslösen.

Aufruf:
    DASHBOARD_PASSWORD=... BOT_STATE_DIR=/state python3 dashboard/server.py
"""
import asyncio
import base64
import hmac
import json
import logging
import os
import sqlite3
from collections import deque
from datetime import datetime
from pathlib import Path

from aiohttp import web

STATE_DIR = Path(os.environ.get('BOT_STATE_DIR', '.'))
USER = os.environ.get('DASHBOARD_USER', 'nici')
PASSWORD = os.environ.get('DASHBOARD_PASSWORD', '')
PORT = int(os.environ.get('PORT', 8080))
HERE = Path(__file__).parent

# Logzeilen, die im Dashboard landen — der Rest ist Rauschen für den Verlauf
LOG_MARKERS = ('[CLOSE]', '[SPOT]', '[CONFLUENCE]', 'Trade blockiert', 'Cooldown',
               '[DATEN', 'WARNING', 'ERROR', 'CRITICAL', 'startet', 'FusionFeed', 'nehme Kraken')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Dashboard] %(levelname)s: %(message)s')
log = logging.getLogger('dashboard')


# ---------- Zugang ----------

@web.middleware
async def basic_auth(request, handler):
    if request.path == '/health':
        return await handler(request)
    header = request.headers.get('Authorization', '')
    if header.startswith('Basic '):
        try:
            user, _, pw = base64.b64decode(header[6:]).decode('utf-8').partition(':')
        except Exception:
            user, pw = '', ''
        # Zeitkonstant vergleichen — beide Teile, damit die Laufzeit nichts verrät
        ok_user = hmac.compare_digest(user.encode(), USER.encode())
        ok_pw = hmac.compare_digest(pw.encode(), PASSWORD.encode())
        if ok_user and ok_pw:
            return await handler(request)
    # Bremse gegen Durchprobieren
    await asyncio.sleep(1)
    return web.Response(status=401, text='Anmeldung nötig',
                        headers={'WWW-Authenticate': 'Basic realm="Trading-Bot", charset="UTF-8"'})


# ---------- Daten ----------

def _ro(path: Path):
    """SQLite nur lesend öffnen — das Dashboard darf dem Bot nie etwas sperren."""
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=2)


def read_status():
    try:
        return json.loads((STATE_DIR / 'status.json').read_text())
    except Exception:
        return None


def read_trades():
    path = STATE_DIR / 'trades.db'
    if not path.exists():
        return []
    try:
        with _ro(path) as conn:
            rows = conn.execute(
                'SELECT id, symbol, side, size, entry_price, exit_price, pnl, fees, '
                'entry_time, exit_time FROM trades ORDER BY exit_time'
            ).fetchall()
    except sqlite3.Error as e:
        log.warning(f'trades.db nicht lesbar: {e}')
        return []
    keys = ('id', 'symbol', 'side', 'size', 'entry_price', 'exit_price', 'pnl', 'fees',
            'entry_time', 'exit_time')
    return [dict(zip(keys, r)) for r in rows]


def read_equity(max_points: int = 600):
    path = STATE_DIR / 'equity.jsonl'
    if not path.exists():
        return []
    points = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                p = json.loads(line)
                points.append([p['t'], p['equity']])
            except (ValueError, KeyError):
                continue  # halbe Zeile nach Absturz
    # Ausdünnen, damit die Kurve nach Monaten nicht Megabytes schickt
    if len(points) > max_points:
        step = len(points) / max_points
        thinned = [points[int(i * step)] for i in range(max_points)]
        thinned[-1] = points[-1]
        points = thinned
    return points


def read_log(n: int = 60):
    path = STATE_DIR / 'bot.log'
    if not path.exists():
        return []
    lines = deque(maxlen=n)
    last_cycle = None
    # errors='replace': nach kill -9 stehen NUL-Bytes im Log
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.replace('\x00', '').rstrip()
            if not line:
                continue
            if '[CONFLUENCE CYCLE]' in line:
                last_cycle = line[:300]
            elif any(m in line for m in LOG_MARKERS):
                lines.append(line[:300])
    # Ruhige Durchgänge sind Rauschen — aber der jüngste zeigt, dass der Bot lebt.
    # Ohne ihn wirkte das Log nach dem letzten Signal wie eingefroren.
    if last_cycle:
        lines.append(last_cycle)
    return list(lines)


def read_recorder():
    path = STATE_DIR / 'market_data.db'
    if not path.exists():
        return None
    try:
        with _ro(path) as conn:
            n, lo, hi = conn.execute('SELECT COUNT(*), MIN(ts), MAX(ts) FROM orderbook').fetchone()
            flow = conn.execute('SELECT COUNT(*) FROM flow').fetchone()[0]
    except sqlite3.Error as e:
        return {'error': str(e)}
    return {
        'orderbook_rows': n,
        'flow_rows': flow,
        'first': datetime.fromtimestamp(lo).isoformat(timespec='seconds') if lo else None,
        'last': datetime.fromtimestamp(hi).isoformat(timespec='seconds') if hi else None,
        'days': round((hi - lo) / 86400, 2) if lo and hi else 0,
        'size_mb': round(path.stat().st_size / 1e6, 1),
    }


def collect():
    return {
        'server_time': datetime.now().isoformat(timespec='seconds'),
        'status': read_status(),
        'trades': read_trades(),
        'equity': read_equity(),
        'log': read_log(),
        'recorder': read_recorder(),
    }


# ---------- Routen ----------

async def index(request):
    return web.FileResponse(HERE / 'index.html', headers={'Cache-Control': 'no-store'})


async def api(request):
    # SQLite und Dateizugriffe aus dem Event-Loop heraushalten
    data = await asyncio.get_running_loop().run_in_executor(None, collect)
    return web.json_response(data, headers={'Cache-Control': 'no-store'})


async def health(request):
    return web.Response(text='ok')


def main():
    if len(PASSWORD) < 12:
        raise SystemExit('DASHBOARD_PASSWORD fehlt oder ist kürzer als 12 Zeichen — starte nicht ungeschützt.')
    app = web.Application(middlewares=[basic_auth])
    app.router.add_get('/', index)
    app.router.add_get('/api/data', api)
    app.router.add_get('/health', health)
    log.info(f'Dashboard auf Port {PORT}, Daten aus {STATE_DIR.resolve()}')
    web.run_app(app, port=PORT, access_log=None, print=None)


if __name__ == '__main__':
    main()
