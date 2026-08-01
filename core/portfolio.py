"""
Portfolio-Management für Paper-Trading
Trackt Balances, Positionen, PNL und speichert alles in SQLite
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Eine offene Position"""
    symbol: str
    side: str  # 'long' oder 'short'
    size: float  # Positionsgröße in Base-Currency
    entry_price: float
    leverage: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0
    market_type: str = 'crypto'  # 'crypto' oder 'polymarket'

    def calculate_pnl(self, current_price: float) -> float:
        """Berechnet unrealized PNL"""
        if self.entry_price <= 0 or self.size <= 0:
            return 0.0

        price_change_pct = (current_price - self.entry_price) / self.entry_price

        if self.side == 'long':
            pnl = price_change_pct * self.size * self.leverage
        else:
            pnl = -price_change_pct * self.size * self.leverage

        self.unrealized_pnl = pnl
        return pnl


@dataclass
class Trade:
    """Ein abgeschlossener Trade"""
    id: int
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    leverage: float
    pnl: float
    fees: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    market_type: str = 'crypto'


@dataclass
class PortfolioState:
    """Aktueller Portfolio-Zustand"""
    balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    positions: Dict[str, Position] = field(default_factory=dict)
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0


class Portfolio:
    """
    Fake-Portfolio-Management mit SQLite-Persistenz
    """

    def __init__(self, start_capital: float = 10000.0, db_path: str = 'trades.db',
                 snapshot_interval_seconds: int = 60, constraints=None):
        # Letzte Verteidigungslinie gegen Positionen, die das Ziel-Venue nicht
        # ausführen kann (Spot: keine Shorts, kein Hebel).
        from core.market_constraints import MarketConstraints
        self.constraints = constraints or MarketConstraints()

        self.start_capital = start_capital
        self.balance = start_capital
        self.equity = start_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.db_path = Path(db_path)
        self.daily_start_balance = start_capital
        self.day_start = datetime.now().date()

        # PNL Tracking
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0

        # Historische Daten für Metriken
        self.equity_history: List[tuple] = []  # (timestamp, equity)
        self.pnl_history: List[float] = []

        # Equity-Snapshots werden gedrosselt geschrieben. update_position_prices()
        # läuft im 1s-Main-Loop; ungedrosselt entstünden 86.400 Punkte pro Tag,
        # und genau diese Sekunden-Auflösung hat die Sharpe-Berechnung verfälscht.
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self._last_snapshot: Optional[datetime] = None

        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialisiert SQLite-Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                leverage REAL NOT NULL,
                pnl REAL NOT NULL,
                fees REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                market_type TEXT DEFAULT 'crypto'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY,
                balance REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                win_count INTEGER NOT NULL,
                loss_count INTEGER NOT NULL,
                last_update TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        ''')

        # Equity-Verlauf dauerhaft speichern — Basis für Sharpe, Max-Drawdown
        # und das Profitabilitäts-Gate. Vorher lag die Kurve nur im RAM und
        # war nach jedem Neustart weg.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                equity REAL NOT NULL,
                balance REAL NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts)')

        # Migration: portfolio_state um Tagesstart-Felder erweitern (additiv,
        # damit bestehende trades.db auf dem Server nicht kaputtgeht).
        cursor.execute('PRAGMA table_info(portfolio_state)')
        existing_cols = {row[1] for row in cursor.fetchall()}
        for col, coltype in (('daily_start_balance', 'REAL'), ('day_start', 'TEXT')):
            if col not in existing_cols:
                cursor.execute(f'ALTER TABLE portfolio_state ADD COLUMN {col} {coltype}')

        conn.commit()
        conn.close()

    def _load_state(self):
        """Lädt Portfolio-Zustand aus DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Lade Portfolio-State (explizite Spalten, damit spätere Migrationen
        # die Indizes nicht verschieben)
        cursor.execute('''
            SELECT balance, realized_pnl, win_count, loss_count,
                   daily_start_balance, day_start
            FROM portfolio_state WHERE id = 1
        ''')
        row = cursor.fetchone()
        if row:
            self.balance = row[0]
            self.realized_pnl = row[1]
            self.win_count = row[2]
            self.loss_count = row[3]
            # Tagesstart weiterführen, sonst resettet der Daily-Drawdown-Guard
            # bei jedem Neustart und der Kill-Switch wäre umgehbar.
            if row[4] is not None and row[5]:
                saved_day = datetime.fromisoformat(row[5]).date()
                if saved_day == datetime.now().date():
                    self.daily_start_balance = row[4]
                    self.day_start = saved_day
                else:
                    self.daily_start_balance = self.balance
            else:
                self.daily_start_balance = self.balance

        # Lade offene Positionen
        cursor.execute('SELECT * FROM positions')
        for row in cursor.fetchall():
            data = json.loads(row[1])
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            self.positions[row[0]] = Position(**data)

        # Lade abgeschlossene Trades — ohne das sind get_avg_win_loss(),
        # get_recent_trades() und die Factor-Attribution nach jedem Neustart
        # leer, während win_count/loss_count überleben. Die Metriken haben
        # sich dadurch widersprochen.
        cursor.execute('''
            SELECT id, symbol, side, size, entry_price, exit_price, leverage,
                   pnl, fees, entry_time, exit_time, strategy, market_type
            FROM trades ORDER BY id
        ''')
        for r in cursor.fetchall():
            self.trades.append(Trade(
                id=r[0], symbol=r[1], side=r[2], size=r[3],
                entry_price=r[4], exit_price=r[5], leverage=r[6],
                pnl=r[7], fees=r[8],
                entry_time=datetime.fromisoformat(r[9]),
                exit_time=datetime.fromisoformat(r[10]),
                strategy=r[11], market_type=r[12] or 'crypto'
            ))

        # Equity-Kurve der letzten 24h in den RAM-Puffer zurückholen
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute(
            'SELECT ts, equity FROM equity_snapshots WHERE ts > ? ORDER BY ts',
            (cutoff,)
        )
        self.equity_history = [
            (datetime.fromisoformat(ts), eq) for ts, eq in cursor.fetchall()
        ]

        conn.close()
        self._update_equity()

    def _save_state(self):
        """Speichert Portfolio-Zustand in DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_state
                (id, balance, realized_pnl, win_count, loss_count, last_update,
                 daily_start_balance, day_start)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.balance, self.realized_pnl, self.win_count, self.loss_count,
              datetime.now().isoformat(),
              self.daily_start_balance, self.day_start.isoformat()))

        # Speichere Positionen
        cursor.execute('DELETE FROM positions')
        for symbol, pos in self.positions.items():
            data = asdict(pos)
            data['timestamp'] = pos.timestamp.isoformat()
            cursor.execute('INSERT INTO positions (symbol, data) VALUES (?, ?)',
                          (symbol, json.dumps(data)))

        conn.commit()
        conn.close()

    def _update_equity(self):
        """Aktualisiert Equity basierend auf offenen Positionen"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.balance + unrealized

        # Prüfe ob neuer Tag
        if datetime.now().date() != self.day_start:
            self.daily_start_balance = self.balance
            self.day_start = datetime.now().date()
            self.daily_pnl = 0.0

    def open_position(self, symbol: str, side: str, size: float, price: float,
                     leverage: float, strategy: str, stop_loss: float = None,
                     take_profit: float = None, market_type: str = 'crypto') -> Optional[Position]:
        """Öffnet eine neue Position"""
        if symbol in self.positions:
            return None  # Position existiert bereits

        # Spot-Venue: eine Short-Position kann physisch nicht gebucht werden.
        # Käme sie hier an, wäre oberhalb (Aggregator, Loop, RiskManager) etwas
        # durchgerutscht — deshalb CRITICAL statt stiller Ablehnung.
        if self.constraints.spot_only and side == 'short':
            logger.critical(
                f"SHORT-Position auf Spot-Venue abgelehnt: {symbol}. "
                f"Das haette der Aggregator bereits verhindern muessen."
            )
            return None

        if self.constraints.spot_only and leverage > 1.0:
            logger.critical(
                f"Gehebelte Position auf Spot-Venue abgelehnt: {symbol} "
                f"mit Leverage {leverage}."
            )
            return None

        # Margin berechnen
        margin_required = size / leverage
        if margin_required > self.balance:
            return None  # Nicht genug Balance

        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            leverage=leverage,
            timestamp=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_type=market_type
        )

        self.positions[symbol] = position
        self.balance -= margin_required
        self._save_state()

        return position

    def close_position(self, symbol: str, exit_price: float, fees: float,
                      strategy: str) -> Optional[Trade]:
        """Schließt eine Position und erstellt Trade-Record"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pnl = pos.calculate_pnl(exit_price) - fees

        # Margin zurückgeben + PNL
        margin = pos.size / pos.leverage
        self.balance += margin + pnl
        self.realized_pnl += pnl
        self.daily_pnl += pnl

        # Win/Loss Tracking
        if pnl > 0:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1

        # Trade Record erstellen. Die ID kommt nach dem Insert aus SQLite
        # (AUTOINCREMENT) — len(self.trades)+1 hat nach einem Neustart mit
        # bereits vorhandenen Trades kollidiert.
        trade = Trade(
            id=0,
            symbol=symbol,
            side=pos.side,
            size=pos.size,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            leverage=pos.leverage,
            pnl=pnl,
            fees=fees,
            entry_time=pos.timestamp,
            exit_time=datetime.now(),
            strategy=strategy,
            market_type=pos.market_type
        )

        trade.id = self._save_trade(trade)
        self.trades.append(trade)

        del self.positions[symbol]
        self._update_equity()
        self._save_state()

        return trade

    def _save_trade(self, trade: Trade) -> int:
        """Speichert Trade in DB und gibt die vergebene Zeilen-ID zurück"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (symbol, side, size, entry_price, exit_price, leverage, pnl, fees, entry_time, exit_time, strategy, market_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade.symbol, trade.side, trade.size, trade.entry_price, trade.exit_price,
              trade.leverage, trade.pnl, trade.fees, trade.entry_time.isoformat(),
              trade.exit_time.isoformat(), trade.strategy, trade.market_type))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def update_position_prices(self, prices: Dict[str, float]):
        """Aktualisiert unrealized PNL für alle Positionen"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.calculate_pnl(prices[symbol])
        self._update_equity()
        self._record_equity_snapshot()

    def _record_equity_snapshot(self, force: bool = False):
        """
        Schreibt einen Equity-Punkt — gedrosselt auf snapshot_interval_seconds.

        Wird aus dem 1s-Main-Loop aufgerufen; ohne Drosselung entstünden
        Sekundendaten, auf denen jede annualisierte Kennzahl unbrauchbar ist.
        """
        now = datetime.now()
        if not force and self._last_snapshot is not None:
            elapsed = (now - self._last_snapshot).total_seconds()
            if elapsed < self.snapshot_interval_seconds:
                return
        self._last_snapshot = now

        self.equity_history.append((now, self.equity))
        # RAM-Puffer auf 24h begrenzen; die Langfrist-Kurve liegt in der DB
        cutoff = now - timedelta(hours=24)
        self.equity_history = [(t, e) for t, e in self.equity_history if t > cutoff]

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            'INSERT INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)',
            (now.isoformat(), self.equity, self.balance)
        )
        conn.commit()
        conn.close()

    def get_equity_snapshots(self, since: Optional[datetime] = None) -> List[tuple]:
        """Liest die persistierte Equity-Kurve als [(datetime, equity), ...]"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if since is not None:
            cursor.execute(
                'SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts',
                (since.isoformat(),)
            )
        else:
            cursor.execute('SELECT ts, equity FROM equity_snapshots ORDER BY ts')
        rows = cursor.fetchall()
        conn.close()
        return [(datetime.fromisoformat(ts), eq) for ts, eq in rows]

    def get_daily_equity_curve(self) -> List[tuple]:
        """
        Verdichtet die Snapshots auf einen Punkt pro Kalendertag (letzter Wert
        des Tages). Grundlage für Sharpe und Drawdown auf Tagesbasis.
        """
        daily: Dict[object, tuple] = {}
        for ts, eq in self.get_equity_snapshots():
            daily[ts.date()] = (ts, eq)
        return [daily[d] for d in sorted(daily)]

    def get_state(self) -> PortfolioState:
        """Gibt aktuellen Portfolio-Zustand zurück"""
        return PortfolioState(
            balance=self.balance,
            equity=self.equity,
            unrealized_pnl=self.equity - self.balance,
            realized_pnl=self.realized_pnl,
            daily_pnl=self.daily_pnl,
            positions=self.positions.copy(),
            win_count=self.win_count,
            loss_count=self.loss_count,
            total_trades=self.win_count + self.loss_count
        )

    def get_win_rate(self) -> float:
        """Berechnet Win-Rate"""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.0
        return self.win_count / total

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe-Ratio auf Basis *täglicher* Returns, annualisiert mit sqrt(365).

        Vorher wurden Sekunden-Snapshots mit sqrt(24*365) annualisiert, also
        unter der Annahme stündlicher Returns — der Wert lag um Größenordnungen
        daneben. Braucht mindestens drei Tage Historie.
        """
        curve = self.get_daily_equity_curve()
        if len(curve) < 3:
            return 0.0

        returns = []
        for i in range(1, len(curve)):
            prev_eq = curve[i - 1][1]
            curr_eq = curve[i][1]
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)

        if len(returns) < 2:
            return 0.0

        returns = np.array(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0

        daily_rf = risk_free_rate / 365.0
        return float((np.mean(returns) - daily_rf) / std_return * np.sqrt(365))

    def get_max_drawdown(self) -> float:
        """
        Maximaler Peak-to-Trough-Drawdown über die gesamte persistierte
        Equity-Kurve (nicht mehr nur über die letzten 24h im RAM).
        """
        snapshots = self.get_equity_snapshots()
        equities = [e for _, e in snapshots]
        if len(equities) < 2:
            return 0.0

        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    def get_profit_factor(self) -> float:
        """
        Echter Profit Factor: Bruttogewinn / Bruttoverlust.

        Nicht zu verwechseln mit der Payoff-Ratio (avg_win/avg_loss), die an
        zwei Stellen im Reporter fälschlich als Profit Factor gemeldet wurde —
        die weist ein Verlustsystem als profitabel aus.
        Rückgabe inf, wenn es Gewinne, aber keine Verluste gibt.
        """
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def get_expectancy(self) -> float:
        """Durchschnittliches PNL pro abgeschlossenem Trade (nach Gebühren)"""
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    def get_net_pnl(self) -> float:
        """Netto-PNL aller abgeschlossenen Trades (Gebühren bereits abgezogen)"""
        return sum(t.pnl for t in self.trades)

    def get_total_fees(self) -> float:
        """Summe aller gezahlten Gebühren"""
        return sum(t.fees for t in self.trades)

    def get_daily_drawdown(self) -> float:
        """Berechnet täglichen Drawdown"""
        if self.daily_start_balance == 0:
            return 0.0
        return (self.daily_start_balance - self.equity) / self.daily_start_balance

    def get_avg_win_loss(self) -> tuple:
        """Berechnet durchschnittlichen Win und Loss"""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        return avg_win, avg_loss

    def get_recent_trades(self, n: int = 10) -> List[Trade]:
        """Gibt die letzten n Trades zurück"""
        return self.trades[-n:]

    def reset(self):
        """Setzt Portfolio auf Startwerte zurück"""
        self.balance = self.start_capital
        self.equity = self.start_capital
        self.positions.clear()
        self.trades.clear()
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.equity_history.clear()
        self.daily_start_balance = self.start_capital
        self.day_start = datetime.now().date()
        self._last_snapshot = None

        # DB leeren
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trades')
        cursor.execute('DELETE FROM portfolio_state')
        cursor.execute('DELETE FROM positions')
        cursor.execute('DELETE FROM equity_snapshots')
        conn.commit()
        conn.close()
