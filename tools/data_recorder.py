#!/usr/bin/env python3
"""
Sammelt Orderbuch- und Orderflow-Daten, die sich historisch nicht beschaffen
lassen.

Hintergrund: Binance liefert Open Interest, Taker-Ratio und Long/Short-Ratio nur
30 Tage rueckwirkend, Orderbuchtiefe ueberhaupt nicht. Genau diese Daten sind die
einzige Signalquelle, die bisher ungeprueft ist (siehe docs/BACKTEST_BEFUNDE.md).
Wer sie auswerten will, muss heute anfangen zu sammeln.

Nach 30 Tagen existiert eine Historie, die die API nicht mehr hergibt; nach 90
Tagen genug fuer eine Validierung nach denselben Regeln wie bisher (drei
Fenster, IC >= 0,03 mit stabilem Vorzeichen).

Aufruf:
    python3 tools/data_recorder.py                  # laeuft bis Strg-C
    INTERVAL_S=300 DB=market_data.db python3 tools/data_recorder.py

Der Dienst ist bewusst anspruchslos: ein Fehler in einer Quelle darf die
anderen nicht mitreissen, und ein Netzwerkausfall darf ihn nicht beenden.
"""
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

import ccxt

SYMBOLS = os.environ.get('SYMBOLS', 'BTC,ETH,SOL').split(',')
INTERVAL_S = int(os.environ.get('INTERVAL_S', 300))
DB_PATH = os.environ.get('DB', 'market_data.db')
DEPTH_LIMIT = 500
# Preisbaender um den Mittelkurs, in denen die Liquiditaet aufsummiert wird
BANDS = (0.001, 0.005, 0.01)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Recorder] %(levelname)s: %(message)s',
    handlers=[logging.FileHandler(os.environ.get('RECORDER_LOG', 'recorder.log')),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('recorder')

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True
    log.info("Beende nach dem laufenden Durchgang...")


class Recorder:
    def __init__(self):
        self.fut = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 20000,
        })
        self.conn = sqlite3.connect(DB_PATH)
        self._init_db()

    # ---------- Datenbank ----------

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS orderbook (
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                mid REAL, bid REAL, ask REAL, spread_bps REAL,
                bid_vol_10bps REAL, ask_vol_10bps REAL,
                bid_vol_50bps REAL, ask_vol_50bps REAL,
                bid_vol_100bps REAL, ask_vol_100bps REAL,
                imbalance_10bps REAL, imbalance_100bps REAL,
                PRIMARY KEY (ts, symbol)
            )''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS flow (
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                open_interest REAL,
                funding_rate REAL,
                taker_buy_sell REAL,
                ls_accounts REAL,
                ls_positions REAL,
                PRIMARY KEY (ts, symbol)
            )''')
        self.conn.commit()

    # ---------- Quellen ----------

    def _pair(self, sym):
        return f"{sym}/USDT:USDT"

    def orderbook(self, sym):
        """Tiefe und Ungleichgewicht — der Teil, den es historisch nirgends gibt."""
        ob = self.fut.fetch_order_book(self._pair(sym), limit=DEPTH_LIMIT)
        bids, asks = ob.get('bids') or [], ob.get('asks') or []
        if not bids or not asks:
            return None
        bid, ask = float(bids[0][0]), float(asks[0][0])
        mid = (bid + ask) / 2
        if mid <= 0:
            return None

        row = {'mid': mid, 'bid': bid, 'ask': ask,
               'spread_bps': (ask - bid) / mid * 10_000}
        for band in BANDS:
            bv = sum(float(p) * float(q) for p, q in bids if p >= mid * (1 - band))
            av = sum(float(p) * float(q) for p, q in asks if p <= mid * (1 + band))
            key = int(band * 10_000)
            row[f'bid_vol_{key}bps'] = bv
            row[f'ask_vol_{key}bps'] = av
            if key in (10, 100):
                total = bv + av
                row[f'imbalance_{key}bps'] = (bv - av) / total if total > 0 else 0.0
        return row

    def flow(self, sym):
        """Open Interest, Funding, Taker- und Long/Short-Verhaeltnisse."""
        row = {}
        pair = self._pair(sym)

        try:
            oi = self.fut.fetch_open_interest(pair)
            row['open_interest'] = oi.get('openInterestAmount') or oi.get('openInterestValue')
        except Exception as e:
            log.debug(f"{sym} OI: {e!r}")

        try:
            fr = self.fut.fetch_funding_rate(pair)
            row['funding_rate'] = fr.get('fundingRate')
        except Exception as e:
            log.debug(f"{sym} Funding: {e!r}")

        # Diese drei gibt es nur ueber die native API, ccxt kennt sie nicht
        for key, endpoint, field in [
            ('taker_buy_sell', 'takerlongshortRatio', 'buySellRatio'),
            ('ls_accounts', 'globalLongShortAccountRatio', 'longShortRatio'),
            ('ls_positions', 'topLongShortPositionRatio', 'longShortRatio'),
        ]:
            try:
                url = (f"https://fapi.binance.com/futures/data/{endpoint}"
                       f"?symbol={sym}USDT&period=5m&limit=1")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read())
                if data:
                    row[key] = float(data[-1][field])
            except Exception as e:
                log.debug(f"{sym} {endpoint}: {e!r}")
        return row

    # ---------- Durchgang ----------

    def tick(self):
        ts = int(time.time())
        ok, failed = 0, 0
        for sym in SYMBOLS:
            try:
                ob = self.orderbook(sym)
                if ob:
                    cols = ['mid', 'bid', 'ask', 'spread_bps',
                            'bid_vol_10bps', 'ask_vol_10bps',
                            'bid_vol_50bps', 'ask_vol_50bps',
                            'bid_vol_100bps', 'ask_vol_100bps',
                            'imbalance_10bps', 'imbalance_100bps']
                    self.conn.execute(
                        f"INSERT OR REPLACE INTO orderbook (ts, symbol, {','.join(cols)}) "
                        f"VALUES (?,?,{','.join('?' * len(cols))})",
                        [ts, sym] + [ob.get(c) for c in cols])
                    ok += 1
            except Exception as e:
                failed += 1
                log.warning(f"{sym} Orderbuch fehlgeschlagen: {e!r}")

            try:
                fl = self.flow(sym)
                if fl:
                    cols = ['open_interest', 'funding_rate', 'taker_buy_sell',
                            'ls_accounts', 'ls_positions']
                    self.conn.execute(
                        f"INSERT OR REPLACE INTO flow (ts, symbol, {','.join(cols)}) "
                        f"VALUES (?,?,{','.join('?' * len(cols))})",
                        [ts, sym] + [fl.get(c) for c in cols])
            except Exception as e:
                failed += 1
                log.warning(f"{sym} Flow fehlgeschlagen: {e!r}")

        self.conn.commit()
        return ok, failed

    def stats(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM orderbook")
        n, lo, hi = c.fetchone()
        if not n:
            return "noch keine Daten"
        days = (hi - lo) / 86400
        return f"{n} Orderbuch-Zeilen über {days:.1f} Tage"

    def run(self):
        log.info(f"Sammle {', '.join(SYMBOLS)} alle {INTERVAL_S}s nach {DB_PATH}")
        log.info(f"Bestand: {self.stats()}")
        # Wanduhr statt Zaehlschleife: auf macOS zaehlt ein schlafender Rechner
        # keine Wartezeit mit, ein reiner sleep-Takt driftet dadurch weg.
        next_run = datetime.now()
        while not _stop:
            now = datetime.now()
            if now < next_run:
                time.sleep(min(5, (next_run - now).total_seconds()))
                continue
            next_run = now + timedelta(seconds=INTERVAL_S)
            try:
                ok, failed = self.tick()
                if failed:
                    log.warning(f"Durchgang: {ok} ok, {failed} Fehler")
                else:
                    log.info(f"Durchgang: {ok} Symbole erfasst | {self.stats()}")
            except Exception as e:
                log.error(f"Durchgang fehlgeschlagen: {e!r}")
        self.conn.close()
        log.info("Beendet.")


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    Recorder().run()
