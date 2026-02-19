"""
Portfolio-Management für Paper-Trading
Trackt Balances, Positionen, PNL und speichert alles in SQLite
"""

import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np


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

    def __init__(self, start_capital: float = 10000.0, db_path: str = 'trades.db'):
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

        conn.commit()
        conn.close()

    def _load_state(self):
        """Lädt Portfolio-Zustand aus DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Lade Portfolio-State
        cursor.execute('SELECT * FROM portfolio_state WHERE id = 1')
        row = cursor.fetchone()
        if row:
            self.balance = row[1]
            self.realized_pnl = row[2]
            self.win_count = row[3]
            self.loss_count = row[4]

        # Lade offene Positionen
        cursor.execute('SELECT * FROM positions')
        for row in cursor.fetchall():
            data = json.loads(row[1])
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            self.positions[row[0]] = Position(**data)

        conn.close()
        self._update_equity()

    def _save_state(self):
        """Speichert Portfolio-Zustand in DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_state (id, balance, realized_pnl, win_count, loss_count, last_update)
            VALUES (1, ?, ?, ?, ?, ?)
        ''', (self.balance, self.realized_pnl, self.win_count, self.loss_count, datetime.now().isoformat()))

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

        # Trade Record erstellen
        trade = Trade(
            id=len(self.trades) + 1,
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

        self.trades.append(trade)
        self._save_trade(trade)

        del self.positions[symbol]
        self._update_equity()
        self._save_state()

        return trade

    def _save_trade(self, trade: Trade):
        """Speichert Trade in DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (symbol, side, size, entry_price, exit_price, leverage, pnl, fees, entry_time, exit_time, strategy, market_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade.symbol, trade.side, trade.size, trade.entry_price, trade.exit_price,
              trade.leverage, trade.pnl, trade.fees, trade.entry_time.isoformat(),
              trade.exit_time.isoformat(), trade.strategy, trade.market_type))

        conn.commit()
        conn.close()

    def update_position_prices(self, prices: Dict[str, float]):
        """Aktualisiert unrealized PNL für alle Positionen"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.calculate_pnl(prices[symbol])
        self._update_equity()

        # Equity-History für Sharpe-Berechnung
        self.equity_history.append((datetime.now(), self.equity))
        # Nur letzte 24h behalten
        cutoff = datetime.now() - timedelta(hours=24)
        self.equity_history = [(t, e) for t, e in self.equity_history if t > cutoff]

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
        """Berechnet Sharpe-Ratio basierend auf stündlichen Returns"""
        if len(self.equity_history) < 2:
            return 0.0

        # Stündliche Returns berechnen
        returns = []
        for i in range(1, len(self.equity_history)):
            prev_eq = self.equity_history[i-1][1]
            curr_eq = self.equity_history[i][1]
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)

        if len(returns) < 2:
            return 0.0

        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualisiert (24h * 365 Tage)
        return (mean_return - risk_free_rate) / std_return * np.sqrt(24 * 365)

    def get_max_drawdown(self) -> float:
        """Berechnet maximalen Drawdown"""
        if len(self.equity_history) < 2:
            return 0.0

        equities = [e for _, e in self.equity_history]
        peak = equities[0]
        max_dd = 0.0

        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

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

        # DB leeren
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trades')
        cursor.execute('DELETE FROM portfolio_state')
        cursor.execute('DELETE FROM positions')
        conn.commit()
        conn.close()
