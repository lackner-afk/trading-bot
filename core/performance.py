"""
Performance-Kennzahlen aus der Trade-Historie.

Eine Implementierung für alle Konsumenten (Reporter, Gate, Auswertung).
Vorher waren dieselben Kennzahlen an mehreren Stellen leicht unterschiedlich
gerechnet — der Profit Factor im Reporter war sogar in Wahrheit die
Payoff-Ratio und wies Verlustsysteme als profitabel aus.

Liest direkt aus trades.db, damit die Auswertung unabhängig von einem
laufenden Bot funktioniert.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Round-Trip-Kosten: Entry + Exit als Taker
# Round-Trip-Kosten: Entry + Exit, je inklusive Spread.
# Bitpanda Fusion Level 1: 0,25 % Gebühr + ~0,05 % Spread pro Seite.
# Wird von round_trip_fee_from_config() aus der settings.yaml abgeleitet —
# diese Konstante ist nur der Fallback.
DEFAULT_ROUND_TRIP_FEE = 0.006


def round_trip_fee_from_config(config: Dict = None) -> float:
    """
    Leitet die Round-Trip-Kosten aus dem `fees:`-Block ab.

    Wichtig für das Gate-Kriterium "Erwartungswert > 2x Round-Trip-Fee": eine
    zu optimistische Gebührenannahme lässt eine Strategie profitabel
    aussehen, die real Geld verbrennt.
    """
    fees = (config or {}).get("fees", {}) or {}
    taker = float(fees.get("crypto_taker", 0.0025))
    spread = float(fees.get("spread_estimate", 0.0005))
    return 2 * (taker + spread)


@dataclass
class TradeRow:
    """Ein abgeschlossener Trade, wie er in der DB liegt."""
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


@dataclass
class PerformanceReport:
    """Alle Kennzahlen eines Auswertungszeitraums."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    largest_win_share: float = 0.0
    positive_week_share: float = 0.0
    trading_days: int = 0
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    short_trades: int = 0
    leveraged_trades: int = 0
    per_strategy: Dict[str, Dict] = field(default_factory=dict)


def load_trades(db_path: str) -> List[TradeRow]:
    """Liest alle abgeschlossenen Trades aus der SQLite."""
    path = Path(db_path)
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cursor.fetchone() is None:
            return []

        cursor.execute("""
            SELECT id, symbol, side, size, entry_price, exit_price, leverage,
                   pnl, fees, entry_time, exit_time, strategy
            FROM trades ORDER BY exit_time
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    trades = []
    for r in rows:
        try:
            trades.append(TradeRow(
                id=r[0], symbol=r[1], side=r[2], size=r[3],
                entry_price=r[4], exit_price=r[5], leverage=r[6],
                pnl=r[7], fees=r[8],
                entry_time=datetime.fromisoformat(r[9]),
                exit_time=datetime.fromisoformat(r[10]),
                strategy=r[11],
            ))
        except (TypeError, ValueError):
            continue
    return trades


def load_daily_equity(db_path: str) -> List[Tuple[datetime, float]]:
    """
    Verdichtet die Equity-Snapshots auf einen Punkt pro Kalendertag.

    Grundlage für Sharpe und Max-Drawdown. Ein Sharpe auf Sekundendaten mit
    Tages-Annualisierung liegt um Größenordnungen daneben — genau das war
    der Fehler vor P0.
    """
    path = Path(db_path)
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='equity_snapshots'"
        )
        if cursor.fetchone() is None:
            return []
        cursor.execute("SELECT ts, equity FROM equity_snapshots ORDER BY ts")
        rows = cursor.fetchall()
    finally:
        conn.close()

    daily: Dict[object, Tuple[datetime, float]] = {}
    for ts_str, equity in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
        except (TypeError, ValueError):
            continue
        daily[ts.date()] = (ts, equity)

    return [daily[d] for d in sorted(daily)]


def compute_performance(db_path: str,
                        round_trip_fee: float = DEFAULT_ROUND_TRIP_FEE) -> PerformanceReport:
    """Berechnet alle Kennzahlen aus trades.db."""
    trades = load_trades(db_path)
    report = PerformanceReport()

    if not trades:
        return report

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    report.total_trades = len(trades)
    report.winning_trades = len(wins)
    report.losing_trades = len(losses)
    report.win_rate = len(wins) / len(trades)

    report.gross_profit = sum(t.pnl for t in wins)
    report.gross_loss = abs(sum(t.pnl for t in losses))
    report.net_pnl = sum(t.pnl for t in trades)
    report.total_fees = sum(t.fees for t in trades)

    # Echter Profit Factor, nicht die Payoff-Ratio
    if report.gross_loss > 0:
        report.profit_factor = report.gross_profit / report.gross_loss
    else:
        report.profit_factor = float('inf') if report.gross_profit > 0 else 0.0

    report.avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    report.avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    report.payoff_ratio = abs(report.avg_win / report.avg_loss) if report.avg_loss else 0.0

    report.expectancy = report.net_pnl / len(trades)
    # Erwartungswert relativ zum eingesetzten Notional — nur so lässt sich
    # gegen die Gebührenschwelle vergleichen.
    total_size = sum(t.size for t in trades)
    report.expectancy_pct = (report.net_pnl / total_size) if total_size > 0 else 0.0

    # Konzentrationstest: hängt der gesamte Gewinn an einem Glückstrade?
    if report.gross_profit > 0 and wins:
        report.largest_win_share = max(t.pnl for t in wins) / report.gross_profit

    report.first_trade = trades[0].exit_time
    report.last_trade = trades[-1].exit_time
    report.trading_days = max(1, (report.last_trade - report.first_trade).days)

    # Gegenprobe Spot-Modus
    report.short_trades = sum(1 for t in trades if t.side == 'short')
    report.leveraged_trades = sum(1 for t in trades if t.leverage and t.leverage > 1.0)

    # Konsistenz über die Zeit: Anteil profitabler Kalenderwochen
    weekly: Dict[Tuple[int, int], float] = {}
    for t in trades:
        key = t.exit_time.isocalendar()[:2]
        weekly[key] = weekly.get(key, 0.0) + t.pnl
    if weekly:
        report.positive_week_share = sum(1 for v in weekly.values() if v > 0) / len(weekly)

    # Sharpe und Max-Drawdown aus der Equity-Kurve
    curve = load_daily_equity(db_path)
    report.sharpe_ratio = _sharpe_from_curve(curve)
    report.max_drawdown = _max_drawdown_from_curve(curve)

    # Aufschlüsselung je Strategie
    for t in trades:
        s = report.per_strategy.setdefault(
            t.strategy or 'unbekannt',
            {'trades': 0, 'pnl': 0.0, 'wins': 0}
        )
        s['trades'] += 1
        s['pnl'] += t.pnl
        if t.pnl > 0:
            s['wins'] += 1

    return report


def _sharpe_from_curve(curve: List[Tuple[datetime, float]]) -> float:
    """Sharpe aus täglichen Returns, annualisiert mit sqrt(365)."""
    if len(curve) < 3:
        return 0.0

    returns = []
    for i in range(1, len(curve)):
        prev = curve[i - 1][1]
        if prev > 0:
            returns.append((curve[i][1] - prev) / prev)

    if len(returns) < 2:
        return 0.0

    arr = np.array(returns)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(365))


def _max_drawdown_from_curve(curve: List[Tuple[datetime, float]]) -> float:
    """Maximaler Peak-to-Trough-Drawdown."""
    if len(curve) < 2:
        return 0.0

    peak = curve[0][1]
    max_dd = 0.0
    for _, equity in curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd
