"""
Persistenz für das Dashboard: Equity-Verlauf, Aktivitäts-Feed und Bot-Flags.

Liegt in derselben SQLite-Datei wie das Portfolio (trades.db), aber in eigenen
Tabellen — das Portfolio-Schema bleibt unberührt.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ActivityEvent:
    """Ein Eintrag im Aktivitäts-Feed."""
    timestamp: datetime
    kind: str              # 'open' | 'close' | 'control'
    symbol: Optional[str]
    title: str
    subtitle: str
    amount: Optional[float]  # realisierter PnL in EUR, None = kein Betrag


class DashboardStore:
    """Schreibt und liest die Dashboard-Tabellen (synchron, kurz — Aufruf via to_thread)."""

    def __init__(self, db_path: str = 'trades.db', start_capital: float = 0.0):
        self.db_path = Path(db_path)
        self.start_capital = start_capital
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dashboard_events'")
            first_run = cur.fetchone() is None

            cur.execute('''
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    ts TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    balance REAL NOT NULL
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    symbol TEXT,
                    title TEXT NOT NULL,
                    subtitle TEXT NOT NULL,
                    amount REAL
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_dashboard_events_ts ON dashboard_events(ts)')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS bot_flags (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            conn.commit()

            if first_run:
                self._backfill_from_trades(cur)
                conn.commit()
        finally:
            conn.close()

    def _backfill_from_trades(self, cur: sqlite3.Cursor):
        """
        Beim allerersten Start Verlauf und Feed aus der bestehenden trades-Tabelle
        rekonstruieren — sonst wäre das Dashboard bis zum ersten neuen Trade leer.
        Die Equity-Kurve ist dabei nur realisiert (Startkapital + kumulierter PnL).
        """
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cur.fetchone() is None:
            return
        cur.execute('''
            SELECT symbol, side, leverage, pnl, entry_time, exit_time, strategy
            FROM trades ORDER BY exit_time
        ''')
        rows = cur.fetchall()
        if not rows:
            return

        equity = self.start_capital
        cur.execute(
            'INSERT OR IGNORE INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)',
            (rows[0][4], equity, equity)
        )
        for symbol, side, leverage, pnl, entry_time, exit_time, strategy in rows:
            equity += pnl
            cur.execute(
                'INSERT OR IGNORE INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)',
                (exit_time, equity, equity)
            )
            open_ev = open_event(symbol, side, leverage, strategy)
            close_ev = close_event(symbol, pnl, strategy, reason='')
            for ts, ev in ((entry_time, open_ev), (exit_time, close_ev)):
                cur.execute(
                    'INSERT INTO dashboard_events (ts, kind, symbol, title, subtitle, amount) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (ts, ev.kind, ev.symbol, ev.title, ev.subtitle, ev.amount)
                )

    # ------------------------------------------------------------------ Equity

    def add_snapshot(self, equity: float, balance: float, ts: Optional[datetime] = None):
        ts = ts or datetime.now()
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)',
                (ts.isoformat(timespec='seconds'), equity, balance)
            )
            conn.commit()
        finally:
            conn.close()

    def get_equity_series(self, since: Optional[datetime]) -> List[Tuple[datetime, float]]:
        conn = self._connect()
        try:
            if since is None:
                rows = conn.execute('SELECT ts, equity FROM equity_snapshots ORDER BY ts').fetchall()
            else:
                # Letzten Punkt VOR dem Fenster mitnehmen, damit die Veränderung
                # im Zeitraum gegen den Stand am Fensteranfang gerechnet wird.
                before = conn.execute(
                    'SELECT ts, equity FROM equity_snapshots WHERE ts < ? ORDER BY ts DESC LIMIT 1',
                    (since.isoformat(),)
                ).fetchall()
                rows = before + conn.execute(
                    'SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts',
                    (since.isoformat(),)
                ).fetchall()
        finally:
            conn.close()
        return [(datetime.fromisoformat(ts), eq) for ts, eq in rows]

    def get_first_timestamp(self) -> Optional[datetime]:
        conn = self._connect()
        try:
            row = conn.execute('SELECT MIN(ts) FROM equity_snapshots').fetchone()
        finally:
            conn.close()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    # ---------------------------------------------------------------- Aktivität

    def add_event(self, event: ActivityEvent):
        conn = self._connect()
        try:
            conn.execute(
                'INSERT INTO dashboard_events (ts, kind, symbol, title, subtitle, amount) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (event.timestamp.isoformat(timespec='seconds'), event.kind, event.symbol,
                 event.title, event.subtitle, event.amount)
            )
            conn.commit()
        finally:
            conn.close()

    def get_events(self, limit: int = 20) -> List[ActivityEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                'SELECT ts, kind, symbol, title, subtitle, amount FROM dashboard_events '
                'ORDER BY ts DESC, id DESC LIMIT ?', (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [ActivityEvent(datetime.fromisoformat(r[0]), *r[1:]) for r in rows]

    # ----------------------------------------------------------------- Trades

    def get_trade_stats(self, since: Optional[datetime]) -> List[Tuple[str, int, int, float]]:
        """Pro Strategie: (strategie, trades, gewinner, pnl) seit `since`."""
        conn = self._connect()
        try:
            query = ('SELECT strategy, COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl) '
                     'FROM trades {} GROUP BY strategy')
            if since is None:
                rows = conn.execute(query.format('')).fetchall()
            else:
                rows = conn.execute(query.format('WHERE exit_time >= ?'), (since.isoformat(),)).fetchall()
        finally:
            conn.close()
        return [(s, n, w or 0, p or 0.0) for s, n, w, p in rows]

    def count_trades_since(self, since: datetime) -> int:
        conn = self._connect()
        try:
            row = conn.execute('SELECT COUNT(*) FROM trades WHERE exit_time >= ?',
                               (since.isoformat(),)).fetchone()
        finally:
            conn.close()
        return row[0] if row else 0

    # ------------------------------------------------------------------- Flags

    def get_flag(self, key: str, default: str = '') -> str:
        conn = self._connect()
        try:
            row = conn.execute('SELECT value FROM bot_flags WHERE key = ?', (key,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else default

    def set_flag(self, key: str, value: str):
        conn = self._connect()
        try:
            conn.execute('INSERT OR REPLACE INTO bot_flags (key, value) VALUES (?, ?)', (key, value))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Event-Fabriken: einheitliche Texte für den Feed
# ---------------------------------------------------------------------------

COIN_NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'XRP': 'XRP',
    'ADA': 'Cardano', 'DOT': 'Polkadot', 'AVAX': 'Avalanche', 'LINK': 'Chainlink',
    'DOGE': 'Dogecoin', 'LTC': 'Litecoin',
}

STRATEGY_NAMES = {
    'confluence': 'Confluence', 'momentum': 'Momentum', 'scalper': 'Scalper',
    'ml': 'ML-Prognose', 'crypto': 'Manuell',
}


def coin_of(symbol: Optional[str]) -> str:
    """BTC_EUR → BTC"""
    if not symbol:
        return ''
    return symbol.replace('/', '_').split('_')[0]


def coin_name(symbol: Optional[str]) -> str:
    base = coin_of(symbol)
    return COIN_NAMES.get(base, base)


def strategy_name(key: Optional[str]) -> str:
    return STRATEGY_NAMES.get(key or '', (key or '').capitalize())


def open_event(symbol: str, side: str, leverage: float, strategy: str,
               ts: Optional[datetime] = None) -> ActivityEvent:
    direction = 'Long' if side == 'long' else 'Short'
    lever = f" · Hebel {leverage:g}x" if leverage > 1 else ''
    return ActivityEvent(
        timestamp=ts or datetime.now(),
        kind='open',
        symbol=symbol,
        title=f"Einstieg {coin_name(symbol)}",
        subtitle=f"{strategy_name(strategy)} · {direction}{lever}",
        amount=None,
    )


def close_event(symbol: str, pnl: float, strategy: str, reason: str,
                ts: Optional[datetime] = None) -> ActivityEvent:
    r = reason.lower()
    if 'trailing' in r:
        title, label = 'Trailing-Stop', 'Gewinn gesichert'
    elif 'stop' in r:
        title, label = 'Stop-Loss', 'Risikoschutz'
    elif 'take' in r:
        title, label = 'Take-Profit', 'Ziel erreicht'
    elif 'risk' in r or 'risiko' in r:
        title, label = 'Risiko-Exit', 'Risk-Limit'
    else:
        title, label = 'Ausstieg', 'Position geschlossen'
    return ActivityEvent(
        timestamp=ts or datetime.now(),
        kind='close',
        symbol=symbol,
        title=f"{title} {coin_name(symbol)}",
        subtitle=f"{label} · {strategy_name(strategy)}",
        amount=pnl,
    )


def control_event(title: str, subtitle: str) -> ActivityEvent:
    return ActivityEvent(datetime.now(), 'control', None, title, subtitle, None)


def window_start(range_key: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """1T / 1W / 1M / 1J / Max → Startzeitpunkt (None = alles)."""
    now = now or datetime.now()
    days = {'1T': 1, '1W': 7, '1M': 30, '1J': 365}.get(range_key)
    return now - timedelta(days=days) if days else None
