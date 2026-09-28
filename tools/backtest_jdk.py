#!/usr/bin/env python3
"""
Backtest der JDK-Orderflow-Strategie auf 1h-Kerzen.

Nutzt exakt die Strategie-Klasse des Live-Bots (strategies/jdk_orderflow.py)
und die Parameter aus config/settings.yaml (strategies.jdk). Simuliert unter
realen Spot-Bedingungen: Einstieg zum Open der nächsten Kerze, SL vor TP
innerhalb der Kerze, Break-Even-Stop ab +1R, Gebühren auf beiden Seiten,
Sizing wie main.py (_execute_signal).

Über die Umgebung steuerbar:
    EXCHANGE=binance              # CCXT-Börse für die Historie (Kraken liefert nur 720 Kerzen)
    SYMBOLS=BTC/EUR,ETH/EUR,SOL/EUR
    DAYS=90                       # Fensterlänge
    BACKTEST_END=2026-05-20       # Fensterende (leer = bis jetzt)
    TAKER_FEE=0.0025              # Bitpanda Fusion Stufe 1
    START_CAPITAL=100
    MIN_ORDER_AMOUNT=25           # Trades unter dieser Größe verwerfen (Default 0 = wie Paper-Modus)
    CSV_DIR=pfad/                 # Kerzen aus CSV statt Börse (BTC_EUR.csv: timestamp,open,high,low,close,volume)
    SYNTHETIC=1                   # Zufallsdaten — nur Funktionstest, Ergebnis bedeutungslos
    JDK_MIN_RR=2.5                # beliebiger Parameter aus strategies.jdk überschreiben

Aufruf:
    python tools/backtest_jdk.py
"""
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.jdk_orderflow import JDKOrderflowStrategy  # noqa: E402

EXCHANGE = os.environ.get('EXCHANGE', 'binance')
SYMBOLS = os.environ.get('SYMBOLS', 'BTC/EUR,ETH/EUR,SOL/EUR').split(',')
DAYS = int(os.environ.get('DAYS', 90))
BACKTEST_END = os.environ.get('BACKTEST_END')
TAKER_FEE = float(os.environ.get('TAKER_FEE', 0.0025))
START_CAPITAL = float(os.environ.get('START_CAPITAL', 100.0))
MIN_ORDER_AMOUNT = float(os.environ.get('MIN_ORDER_AMOUNT', 0.0))
# Bitpanda Fusion nimmt Orders erst ab 25 EUR an — der Paper-Modus prüft das nicht.
FUSION_MIN_ORDER = 25.0
CSV_DIR = os.environ.get('CSV_DIR')
SYNTHETIC = os.environ.get('SYNTHETIC') == '1'


def load_config():
    with open(ROOT / 'config' / 'settings.yaml') as f:
        settings = yaml.safe_load(f)
    cfg = dict(settings.get('strategies', {}).get('jdk', {}))
    # JDK_<PARAM>=wert überschreibt einzelne Parameter (Typ wie in der Config)
    for key, value in list(cfg.items()):
        env = os.environ.get(f'JDK_{key.upper()}')
        if env is None:
            continue
        if isinstance(value, bool):
            cfg[key] = env.lower() in ('1', 'true', 'yes')
        elif isinstance(value, (int, float)):
            cfg[key] = type(value)(float(env)) if isinstance(value, int) else float(env)
        else:
            cfg[key] = env
    cfg['fee_pct'] = TAKER_FEE
    return cfg, settings.get('risk', {})


def fetch_candles(symbol: str, days: int, warmup_bars: int) -> pd.DataFrame:
    if CSV_DIR:
        df = pd.read_csv(Path(CSV_DIR) / f"{symbol.replace('/', '_')}.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    if SYNTHETIC:
        rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
        n = days * 24 + warmup_bars
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        wick = np.abs(rng.normal(0, 0.003, n)) * closes
        return pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=n, freq='1h'),
            'open': opens, 'close': closes,
            'high': np.maximum(opens, closes) + wick,
            'low': np.minimum(opens, closes) - wick,
            'volume': rng.lognormal(3, 0.5, n),
        })

    import ccxt
    exchange = getattr(ccxt, EXCHANGE)({'enableRateLimit': True})
    if BACKTEST_END:
        end = int(datetime.strptime(BACKTEST_END, '%Y-%m-%d')
                  .replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        end = exchange.milliseconds()
    since = end - (days * 24 + warmup_bars) * 3600 * 1000
    rows = []
    while since < end:
        batch = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if batch[-1][0] + 1 <= since:
            break
        since = batch[-1][0] + 1
        if len(batch) < 2:
            break
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='ts')
    df = df[df['ts'] <= end]
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop(columns='ts').reset_index(drop=True)


def simulate(cfg: dict, risk_cfg: dict, candles: dict) -> dict:
    strat = JDKOrderflowStrategy(cfg)
    window = strat.lookback_bars
    max_concurrent = risk_cfg.get('max_concurrent_positions', 2)
    max_risk = risk_cfg.get('max_risk_per_trade', 0.02)
    leverage = strat.leverage

    # Gemeinsame Zeitachse über alle Symbole
    timeline = sorted(set().union(*[set(df['timestamp']) for df in candles.values()]))
    index = {sym: {ts: i for i, ts in enumerate(df['timestamp'])} for sym, df in candles.items()}

    balance = START_CAPITAL
    positions, trades, equity_curve, pending = {}, [], [], {}
    below_min_order = 0

    for ts in timeline[window:]:
        # 1) Offene Signale zum Open dieser Kerze einbuchen
        for sym, sig in list(pending.items()):
            i = index[sym].get(ts)
            del pending[sym]
            if i is None or sym in positions or len(positions) >= max_concurrent:
                continue
            row = candles[sym].iloc[i]
            entry = row['open']
            if sig.side == 'long' and not (sig.stop_loss < entry < sig.take_profit):
                continue
            if sig.side == 'short' and not (sig.take_profit < entry < sig.stop_loss):
                continue
            equity = balance + sum(p['margin'] for p in positions.values())
            sl_pct = abs(entry - sig.stop_loss) / entry
            margin = min((equity * max_risk) / sl_pct, equity * 0.20)   # RiskManager.size_from_risk
            margin = max(max(10.0, equity * 0.15), min(equity * 0.25, margin))
            notional = margin * leverage
            if margin > balance or balance < 20 or notional < MIN_ORDER_AMOUNT:
                continue
            balance -= margin
            positions[sym] = {'side': sig.side, 'entry': entry, 'sl': sig.stop_loss,
                              'tp': sig.take_profit, 'margin': margin, 'notional': notional,
                              'setup': sig.setup, 'risk': abs(entry - sig.stop_loss)}
            if notional < FUSION_MIN_ORDER:
                below_min_order += 1

        # 2) Exits dieser Kerze (SL vor TP — konservativ)
        for sym in list(positions):
            i = index[sym].get(ts)
            if i is None:
                continue
            pos, row = positions[sym], candles[sym].iloc[i]
            exit_price, reason = None, None
            if pos['side'] == 'long':
                if row['low'] <= pos['sl']:
                    exit_price, reason = min(pos['sl'], row['open']), 'SL'
                elif row['high'] >= pos['tp']:
                    exit_price, reason = pos['tp'], 'TP'
            else:
                if row['high'] >= pos['sl']:
                    exit_price, reason = max(pos['sl'], row['open']), 'SL'
                elif row['low'] <= pos['tp']:
                    exit_price, reason = pos['tp'], 'TP'
            if exit_price is not None:
                d = 1 if pos['side'] == 'long' else -1
                pnl = pos['notional'] * d * (exit_price - pos['entry']) / pos['entry'] \
                    - pos['notional'] * TAKER_FEE * 2
                balance += pos['margin'] + pnl
                if reason == 'SL' and d * (pos['sl'] - pos['entry']) > 0:
                    reason = 'BE'
                trades.append({'symbol': sym, 'setup': pos['setup'], 'pnl': pnl, 'exit': reason,
                               'r': d * (exit_price - pos['entry']) / pos['risk']})
                del positions[sym]
                continue
            # Break-Even erst ab der nächsten Kerze wirksam
            best = row['high'] if pos['side'] == 'long' else row['low']
            new_sl = strat.breakeven_stop(pos['side'], pos['entry'], pos['sl'], best)
            if new_sl is not None:
                pos['sl'] = new_sl

        # 3) Neue Signale auf der abgeschlossenen Kerze → Einstieg zum nächsten Open
        for sym, df in candles.items():
            i = index[sym].get(ts)
            if i is None or i < window or sym in positions:
                continue
            sig = strat.evaluate(sym, df.iloc[i - window + 1:i + 1])
            if sig is not None:
                pending[sym] = sig

        equity = balance
        for sym, pos in positions.items():
            i = index[sym].get(ts)
            if i is not None:
                price = candles[sym]['close'].iloc[i]
                d = 1 if pos['side'] == 'long' else -1
                equity += pos['margin'] + pos['notional'] * d * (price - pos['entry']) / pos['entry']
            else:
                equity += pos['margin']
        equity_curve.append(equity)

    # Offene Positionen zum letzten Kurs schließen
    for sym, pos in positions.items():
        price = candles[sym]['close'].iloc[-1]
        d = 1 if pos['side'] == 'long' else -1
        pnl = pos['notional'] * d * (price - pos['entry']) / pos['entry'] - pos['notional'] * TAKER_FEE * 2
        balance += pos['margin'] + pnl
        trades.append({'symbol': sym, 'setup': pos['setup'], 'pnl': pnl, 'exit': 'Ende',
                       'r': d * (price - pos['entry']) / pos['risk']})

    eq = pd.Series(equity_curve) if equity_curve else pd.Series([START_CAPITAL])
    peak = eq.cummax()
    wins = [t for t in trades if t['pnl'] > 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    buy_hold = np.mean([df['close'].iloc[-1] / df['close'].iloc[window] - 1 for df in candles.values()])

    return {
        'final': balance,
        'return_pct': (balance - START_CAPITAL) / START_CAPITAL,
        'max_dd': float(((peak - eq) / peak).max()),
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else float('inf'),
        'avg_r': float(np.mean([t['r'] for t in trades])) if trades else 0.0,
        'exits': dict(Counter(t['exit'] for t in trades)),
        'setups': dict(Counter(t['setup'] for t in trades)),
        'by_symbol': dict(Counter(t['symbol'] for t in trades)),
        'buy_hold': float(buy_hold),
        'below_min_order': below_min_order,
    }


def main():
    cfg, risk_cfg = load_config()
    warmup = cfg.get('lookback_bars', 240)
    source = 'CSV' if CSV_DIR else ('SYNTHETISCH' if SYNTHETIC else EXCHANGE)
    print(f"JDK-Backtest | {', '.join(SYMBOLS)} | {DAYS} Tage 1h | Quelle {source} | "
          f"Gebühr {TAKER_FEE:.2%} | Kapital {START_CAPITAL:.0f}")
    if SYNTHETIC:
        print("ACHTUNG: Zufallsdaten — das Ergebnis sagt nichts über die Strategie aus.")

    candles = {}
    for sym in SYMBOLS:
        df = fetch_candles(sym, DAYS, warmup)
        if len(df) <= warmup:
            print(f"  {sym}: zu wenig Kerzen ({len(df)}), übersprungen")
            continue
        candles[sym] = df
        print(f"  {sym}: {len(df)} Kerzen ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")
    if not candles:
        print("FEHLER: keine Daten")
        return 1

    r = simulate(cfg, risk_cfg, candles)
    print()
    print(f"Endkapital:     {r['final']:.2f} ({r['return_pct']:+.1%})   Buy & Hold Ø: {r['buy_hold']:+.1%}")
    print(f"Trades:         {r['n_trades']}   Trefferquote {r['win_rate']:.0%}   "
          f"Profit-Faktor {r['profit_factor']:.2f}   Ø {r['avg_r']:+.2f}R")
    print(f"Max Drawdown:   {r['max_dd']:.1%}")
    print(f"Exits:          {r['exits']}")
    print(f"Setups:         {r['setups']}")
    print(f"Je Symbol:      {r['by_symbol']}")
    if r['below_min_order']:
        print(f"\nHINWEIS: {r['below_min_order']} von {r['n_trades']} Trades lagen unter "
              f"{FUSION_MIN_ORDER:.0f} EUR — auf Bitpanda Fusion wären sie abgelehnt worden.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
