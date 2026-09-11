#!/usr/bin/env python3
"""
Backtester für die DailyTrendStrategy (Tageskerzen, Long/Flat, Spot).

Rechnet mit denselben Regeln wie der Live-Bot:
  * Signale aus `strategies.daily_trend.compute_state` — identischer Code.
  * Signal am Tagesschluss, Ausführung zur Eröffnung des NÄCHSTEN Tages.
  * Gebühr je Seite (Fusion Stufe 1: 0,25 %), Slippage, 25 € Mindestorder.
  * Harte RiskManager-Grenzen: max. 20 % des Kapitals je Position, max. 2 %
    Verlust bis zum Notstopp. Beides ist NICHT abschaltbar — genau wie live.
  * Notstopp intraday gegen das Tagestief.

Mehrere Zeitfenster (Gesamtzeitraum + Kalenderjahre), damit ein Parameter, der
nur in einem Fenster funktioniert, als das erkannt wird, was er ist:
Anpassung an ein Fenster, nicht Kalibrierung.

Beispiele:
    # Binance-Daten ab 2018, Fusion-Gebühr, 100 € Startkapital
    python tools/backtest_daily_trend.py

    # Eigene Parameter, mehrere SMA-Längen und Startkapitalien
    python tools/backtest_daily_trend.py --ma 50,100,150,200 --capital 100,250,500

    # Direkt gegen Fusion-Kerzen (bis 1440 Tage), Key aus der Umgebung
    FUSION_API_KEY=... python tools/backtest_daily_trend.py --source fusion

    # Offline aus CSV-Dateien (Spalten: timestamp,open,high,low,close,volume)
    python tools/backtest_daily_trend.py --source csv --csv-dir data/daily
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.daily_trend import TrendParams, compute_state  # noqa: E402


# ============================================================================
# Simulation
# ============================================================================

@dataclass
class SimConfig:
    start_capital: float = 100.0
    fee_pct: float = 0.0025          # je Seite (Fusion Stufe 1)
    slippage_pct: float = 0.0005     # je Seite
    min_order_amount: float = 25.0   # Fusion-Mindestorder BTC-EUR
    allocation_pct: float = 0.20     # Wunschgröße je Symbol
    max_position_pct: float = 0.20   # RiskManager.MAX_POSITION_SIZE (hart)
    max_risk_pct: float = 0.02       # RiskManager.MAX_RISK_PER_TRADE (hart)


@dataclass
class SimResult:
    equity_curve: pd.Series
    trades: List[dict]
    skipped_min_order: int
    buy_hold_return: float
    exposure: float
    metrics: Dict[str, float] = field(default_factory=dict)


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    dd = (curve / peak - 1.0).min()
    return float(-dd) if not math.isnan(dd) else 0.0


def _sharpe(curve: pd.Series) -> float:
    rets = curve.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(365))


def position_notional(equity: float, cash: float, params: TrendParams, cfg: SimConfig) -> float:
    """
    Positionsgröße wie im Live-Bot: Wunschanteil, dann die harten Grenzen des
    RiskManagers (20 % je Position, 2 % Risiko bis zum Notstopp), dann Cash.
    """
    notional = equity * cfg.allocation_pct
    notional = min(notional, equity * cfg.max_position_pct)
    if params.max_loss_pct > 0:
        notional = min(notional, equity * cfg.max_risk_pct / params.max_loss_pct)
    # Gebühr muss aus dem Cash bezahlt werden
    notional = min(notional, cash / (1.0 + cfg.fee_pct + cfg.slippage_pct))
    return max(0.0, notional)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if getattr(df['timestamp'].dt, 'tz', None) is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['date'] = df['timestamp'].dt.normalize()
    df = df.drop_duplicates('date').sort_values('date').set_index('date')
    for col in ('open', 'high', 'low', 'close'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['open', 'high', 'low', 'close'])


def simulate(candles: Dict[str, pd.DataFrame], params: TrendParams, cfg: SimConfig,
             start: Optional[pd.Timestamp] = None,
             end: Optional[pd.Timestamp] = None) -> SimResult:
    """
    Simuliert Long/Flat je Symbol über den gemeinsamen Kalender.

    `start`/`end` begrenzen den Handelszeitraum; die Signale werden trotzdem auf
    der gesamten Historie berechnet, damit die SMA am Fensteranfang bereits
    aufgewärmt ist.
    """
    data = {s: _prepare(df) for s, df in candles.items() if df is not None and len(df) > 0}
    states = {s: compute_state(df, params) for s, df in data.items()}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()])) if data else []
    if start is not None:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]
    if end is not None:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(end)]

    cash = cfg.start_capital
    positions: Dict[str, dict] = {}          # symbol -> {qty, entry_price, stop, entry_date, notional, fees}
    pending_entry: set = set()
    pending_exit: set = set()
    trades: List[dict] = []
    skipped = 0
    curve = {}
    invested_fraction = []

    buy_px = lambda p: p * (1.0 + cfg.slippage_pct)   # noqa: E731
    sell_px = lambda p: p * (1.0 - cfg.slippage_pct)  # noqa: E731

    def close_position(symbol: str, price: float, date: pd.Timestamp, reason: str):
        nonlocal cash
        pos = positions.pop(symbol)
        proceeds = pos['qty'] * price
        fee = proceeds * cfg.fee_pct
        cash += proceeds - fee
        pnl = proceeds - pos['notional'] - fee - pos['fees']
        trades.append({
            'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date,
            'entry_price': pos['entry_price'], 'exit_price': price,
            'notional': pos['notional'], 'pnl': pnl, 'fees': fee + pos['fees'],
            'reason': reason,
        })

    prev_close: Dict[str, float] = {}

    for date in all_dates:
        # --- 1. Am Vortag entschiedene Aktionen zur heutigen Eröffnung ausführen
        for symbol in list(pending_exit):
            pending_exit.discard(symbol)
            if symbol in positions and date in data[symbol].index:
                close_position(symbol, sell_px(float(data[symbol].at[date, 'open'])), date, 'trend')

        equity_prev = cash + sum(p['qty'] * prev_close.get(s, p['entry_price']) for s, p in positions.items())
        for symbol in sorted(pending_entry):
            pending_entry.discard(symbol)
            if symbol in positions or date not in data[symbol].index:
                continue
            price = buy_px(float(data[symbol].at[date, 'open']))
            notional = position_notional(equity_prev, cash, params, cfg)
            if notional < cfg.min_order_amount:
                skipped += 1
                continue
            fee = notional * cfg.fee_pct
            cash -= notional + fee
            positions[symbol] = {
                'qty': notional / price, 'entry_price': price,
                'stop': price * (1.0 - params.max_loss_pct) if params.max_loss_pct > 0 else 0.0,
                'entry_date': date, 'notional': notional, 'fees': fee,
            }

        # --- 2. Notstopp intraday
        for symbol in list(positions):
            if date not in data[symbol].index:
                continue
            pos = positions[symbol]
            row = data[symbol].loc[date]
            if pos['stop'] > 0 and float(row['low']) <= pos['stop']:
                # Gap unter den Stop? Dann gibt es nur den Eröffnungskurs.
                fill = min(pos['stop'], float(row['open']))
                close_position(symbol, sell_px(fill), date, 'stop')

        # --- 3. Tagesschluss: Zustand bewerten, Aktionen für morgen vormerken
        for symbol, df in data.items():
            if date not in df.index:
                continue
            prev_close[symbol] = float(df.at[date, 'close'])
            st = states[symbol].loc[date]
            if symbol in positions:
                if bool(st['exit_ok']):
                    pending_exit.add(symbol)
            elif bool(st['entry_ok']):
                pending_entry.add(symbol)

        market_value = sum(p['qty'] * prev_close.get(s, p['entry_price']) for s, p in positions.items())
        equity = cash + market_value
        curve[date] = equity
        invested_fraction.append(market_value / equity if equity > 0 else 0.0)

    equity_curve = pd.Series(curve, dtype=float)

    # Offene Positionen zum letzten Schluss bewerten (bleiben offen, kein Trade-Record)
    # --- Buy & Hold, gleichgewichtet, voll investiert, gleiche Gebühr
    bh = 0.0
    if all_dates and data:
        legs = []
        for symbol, df in data.items():
            window = df.loc[(df.index >= all_dates[0]) & (df.index <= all_dates[-1])]
            if len(window) >= 2:
                entry = buy_px(float(window['open'].iloc[0])) * (1 + cfg.fee_pct)
                exit_ = sell_px(float(window['close'].iloc[-1])) * (1 - cfg.fee_pct)
                legs.append(exit_ / entry - 1.0)
        bh = float(np.mean(legs)) if legs else 0.0

    n_days = len(equity_curve)
    total_return = float(equity_curve.iloc[-1] / cfg.start_capital - 1.0) if n_days else 0.0
    cagr = (1 + total_return) ** (365.0 / n_days) - 1.0 if n_days > 30 and total_return > -1 else total_return
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = -sum(t['pnl'] for t in losses)

    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'max_dd': _max_drawdown(equity_curve),
        'sharpe': _sharpe(equity_curve),
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0),
        'fees_paid': sum(t['fees'] for t in trades) + sum(p['fees'] for p in positions.values()),
        'stops_hit': sum(1 for t in trades if t['reason'] == 'stop'),
        'days': n_days,
    }
    return SimResult(
        equity_curve=equity_curve, trades=trades, skipped_min_order=skipped,
        buy_hold_return=bh, exposure=float(np.mean(invested_fraction)) if invested_fraction else 0.0,
        metrics=metrics,
    )


# ============================================================================
# Daten
# ============================================================================

CACHE_DIR = Path(__file__).resolve().parent.parent / '.cache'


def load_ccxt(exchange_id: str, symbol: str, since: str, use_cache: bool = True) -> pd.DataFrame:
    """Tageskerzen paginiert über CCXT (Binance liefert BTC/EUR ab 2019, USDT ab 2017)."""
    import ccxt  # lazy: nicht jeder Nutzer hat ccxt für CSV/Fusion-Läufe

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"daily_{exchange_id}_{symbol.replace('/', '-')}.csv"
    if use_cache and cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        return pd.read_csv(cache, parse_dates=['timestamp'])

    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    since_ms = int(pd.Timestamp(since).timestamp() * 1000)
    rows: List[list] = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, '1d', since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= since_ms or len(batch) < 2:
            break
        since_ms = last + 1
        if last >= int(time.time() * 1000) - 86_400_000:
            break
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    if use_cache:
        df.to_csv(cache, index=False)
    return df


def load_fusion(symbol: str, api_key: str, limit: int = 1440) -> pd.DataFrame:
    """Tageskerzen direkt von Bitpanda Fusion (max. 1440 Bars je Abfrage)."""
    import requests

    pair = symbol.replace('_', '-').replace('/', '-')
    resp = requests.get(
        f"https://api.fusion.bitpanda.com/v1/candles/{pair}",
        params={'interval': '1d', 'limit': limit},
        headers={'x-api-key': api_key, 'Accept': 'application/json'},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('timestamp')


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')


# ============================================================================
# Fenster + Ausgabe
# ============================================================================

def build_windows(dates: pd.DatetimeIndex, mode: str) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows = [('gesamt', dates.min(), dates.max())]
    if mode in ('yearly', 'all'):
        for year in sorted(set(dates.year)):
            sub = dates[dates.year == year]
            if len(sub) >= 200:  # Rumpfjahre verzerren, weglassen
                windows.append((str(year), sub.min(), sub.max()))
    return windows


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['ccxt', 'fusion', 'csv'], default='ccxt')
    ap.add_argument('--exchange', default='binance', help='CCXT-Börse (binance, bitstamp, kraken ...)')
    ap.add_argument('--symbols', default='BTC/EUR,ETH/EUR')
    ap.add_argument('--since', default='2018-01-01')
    ap.add_argument('--csv-dir', default='data/daily')
    ap.add_argument('--ma', default='50,100,150,200', help='SMA-Längen, kommagetrennt')
    ap.add_argument('--entry-buffer', type=float, default=0.01)
    ap.add_argument('--exit-buffer', type=float, default=0.02)
    ap.add_argument('--max-loss', type=float, default=0.08, help='Notstopp unter Einstieg')
    ap.add_argument('--no-rising', action='store_true', help='SMA-Steigung nicht verlangen')
    ap.add_argument('--fee', type=float, default=0.0025, help='Gebühr je Seite')
    ap.add_argument('--slippage', type=float, default=0.0005)
    ap.add_argument('--min-order', type=float, default=25.0)
    ap.add_argument('--capital', default='100', help='Startkapitalien, kommagetrennt (z.B. 100,250,500)')
    ap.add_argument('--allocation', type=float, default=0.20, help='Wunschanteil je Symbol')
    ap.add_argument('--windows', choices=['gesamt', 'yearly'], default='yearly')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--trades', action='store_true', help='Einzeltrades der letzten Kombination ausgeben')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    # --- Daten laden
    candles: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if args.source == 'ccxt':
            df = load_ccxt(args.exchange, symbol, args.since, use_cache=not args.no_cache)
        elif args.source == 'fusion':
            key = os.environ.get('FUSION_API_KEY') or os.environ.get('BITPANDA_API_KEY')
            if not key:
                sys.exit("FUSION_API_KEY fehlt in der Umgebung.")
            df = load_fusion(symbol, key)
        else:
            path = Path(args.csv_dir) / f"{symbol.replace('/', '_').replace('-', '_')}.csv"
            if not path.exists():
                sys.exit(f"CSV nicht gefunden: {path}")
            df = load_csv(path)
        print(f"{symbol}: {len(df)} Tageskerzen "
              f"{df['timestamp'].min():%Y-%m-%d} bis {df['timestamp'].max():%Y-%m-%d}")
        candles[symbol] = df

    all_dates = pd.DatetimeIndex(sorted(set().union(
        *[set(pd.to_datetime(df['timestamp']).dt.normalize()) for df in candles.values()])))
    windows = build_windows(all_dates, args.windows)
    ma_list = [int(x) for x in args.ma.split(',')]
    capitals = [float(x) for x in args.capital.split(',')]

    print(f"\nGebühr {args.fee:.2%} je Seite | Slippage {args.slippage:.2%} | Mindestorder {args.min_order:.0f} € | "
          f"Anteil je Symbol {args.allocation:.0%} (harte Kappung 20 %, 2 % Risiko) | "
          f"Puffer +{args.entry_buffer:.1%}/-{args.exit_buffer:.1%} | Notstopp {args.max_loss:.0%}")

    last_result: Optional[SimResult] = None
    for capital in capitals:
        cfg = SimConfig(start_capital=capital, fee_pct=args.fee, slippage_pct=args.slippage,
                        min_order_amount=args.min_order, allocation_pct=args.allocation)
        print(f"\n=== Startkapital {capital:,.0f} € ===")
        header = (f"{'SMA':>5} {'Fenster':>8} {'Return':>8} {'B&H':>8} {'MaxDD':>7} {'Sharpe':>7} "
                  f"{'Trades':>6} {'Stops':>5} {'WR':>5} {'Fees€':>7} {'Invest':>7} {'Skip':>5}")
        print(header)
        print('-' * len(header))
        robustness: Dict[int, List[float]] = {}
        for ma in ma_list:
            params = TrendParams(ma_days=ma, entry_buffer_pct=args.entry_buffer,
                                 exit_buffer_pct=args.exit_buffer,
                                 require_rising_ma=not args.no_rising, max_loss_pct=args.max_loss)
            for label, w_start, w_end in windows:
                res = simulate(candles, params, cfg, start=w_start, end=w_end)
                last_result = res
                m = res.metrics
                print(f"{ma:>5} {label:>8} {fmt_pct(m['total_return']):>8} {fmt_pct(res.buy_hold_return):>8} "
                      f"{m['max_dd'] * 100:>6.1f}% {m['sharpe']:>7.2f} {m['n_trades']:>6} {m['stops_hit']:>5} "
                      f"{m['win_rate'] * 100:>4.0f}% {m['fees_paid']:>7.2f} {res.exposure * 100:>6.0f}% "
                      f"{res.skipped_min_order:>5}")
                if label != 'gesamt':
                    robustness.setdefault(ma, []).append(m['total_return'])
            print()

        if robustness:
            print("Robustheit je SMA über die Jahresfenster (positiv / gesamt, schlechtestes Jahr, Summe):")
            for ma, rets in robustness.items():
                pos = sum(1 for r in rets if r > 0)
                print(f"  SMA{ma:<4} {pos}/{len(rets)} positiv | min {fmt_pct(min(rets))} | Summe {fmt_pct(sum(rets))}")

        min_eq = args.min_order / min(args.allocation, cfg.max_position_pct)
        if capital < min_eq:
            print(f"\nHinweis: Mit {capital:,.0f} € und 20 %-Kappung erreicht keine Order die Mindestgröße "
                  f"von {args.min_order:.0f} €. Dafür braucht es mindestens {min_eq:,.0f} € Kapital.")

    if args.trades and last_result is not None:
        print("\nEinzeltrades (letzte Kombination):")
        for t in last_result.trades:
            print(f"  {t['symbol']:<8} {t['entry_date']:%Y-%m-%d} -> {t['exit_date']:%Y-%m-%d} "
                  f"{t['entry_price']:>10.2f} -> {t['exit_price']:>10.2f} | {t['pnl']:>+8.2f} € | {t['reason']}")


if __name__ == '__main__':
    main()
