#!/usr/bin/env python3
"""
Kalibrier-Sweep für min_confluence_score nach dem Skalen-Fix (0-1-Skala).

Pass 1: Voller Signalpfad (Regime -> Faktoren -> Aggregator, Schwelle 0) einmal
        über alle Kerzen -> Signal-Raster mit Score/Richtung/TP/SL/Leverage.
Pass 2: Portfolio-Simulation pro Schwellwert über das Raster (Live-Sizing,
        max. 2 Positionen, Confidence-Gate 0.55, 10%-Tagesdrawdown-Pause).
"""
import sys
import json
import time as _time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import ccxt
import yaml
import os
import pickle

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import strategies.confluence_strategy as cs_mod
import strategies.signal_aggregator as agg_mod
from strategies.confluence_strategy import ConfluenceStrategy
from strategies.factors.sentiment import SentimentFactor
from strategies.factors.macro_news import MacroNewsFilter
from strategies.factors.technical import (
    MultiTimeframeTrendFactor, MomentumFactor, VolatilityFilter,
    BreakoutFactor, VolumeConfirmationFactor, MeanReversionFactor,
)

# Ueber die Umgebung steuerbar, um mehrere Marktphasen zu pruefen:
#   DAYS=90  BACKTEST_END=2026-05-01  -> 90 Tage, endend am 01.05.2026
DAYS = int(os.environ.get('DAYS', 30))
BACKTEST_END = os.environ.get('BACKTEST_END')  # ISO-Datum oder leer = bis jetzt
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
WINDOW = 250          # wie live: n=250 fuer 200er-EMA
# Ueber die Umgebung setzbar, um echte Boersen-Gebuehren durchzurechnen:
#   Bitpanda Fusion Stufe 1 = 0.0025, One Trading = 0.0015, Annahme bisher = 0.0006
TAKER_FEE = float(os.environ.get('TAKER_FEE', 0.0006))
START_CAPITAL = float(os.environ.get('START_CAPITAL', 10_000.0))
# Spot-Boersen koennen nicht shorten (Bitpanda Fusion: nur Buy/Sell, kein Hebel).
LONG_ONLY = os.environ.get('LONG_ONLY') == '1'
# Mindestordergroesse der Boerse in Quote-Waehrung (Fusion: 25 EUR bei BTC-EUR).
MIN_ORDER_AMOUNT = float(os.environ.get('MIN_ORDER_AMOUNT', 0.0))
# TP/SL-Floors wie min_tp_pct/min_sl_pct in der settings.yaml
MIN_TP_PCT = float(os.environ.get('MIN_TP_PCT', 0.0035))
MIN_SL_PCT = float(os.environ.get('MIN_SL_PCT', 0.0030))
MAX_CONCURRENT = 2
MARGIN_PCT = 0.20
MAX_DAILY_DD = 0.10
# TP/SL-Profile als ATR-Vielfache, ueber die Umgebung setzbar:
#   TPSL="8:4,10:4"  -> TP 8xATR / SL 4xATR und TP 10xATR / SL 4xATR
# Wichtig ist das Verhaeltnis: liegt der Stop weiter weg als das Ziel (5:7),
# muss die Trefferquote ueber 72% liegen, um die Round-Trip-Gebuehren zu decken.
_tpsl_env = os.environ.get('TPSL')
if _tpsl_env:
    TPSL_PROFILES = [tuple(int(x) for x in pair.split(':')) for pair in _tpsl_env.split(',')]
else:
    TPSL_PROFILES = [(4, 4), (6, 6), (8, 8), (6, 5), (8, 6), (5, 7)]
COOLDOWNS = [0, 6]  # Kerzen Sperre pro Symbol nach Verlust-Exit

agg_mod.print = lambda *a, **k: None
cs_mod.print = lambda *a, **k: None


class HistoricalSentimentFactor(SentimentFactor):
    def __init__(self, fng_by_date, config=None):
        super().__init__(config)
        self.fng_by_date = fng_by_date
        self.current_ts = None

    def _get_fear_and_greed(self):
        if self.current_ts is None:
            return None
        return self.fng_by_date.get(self.current_ts.strftime('%Y-%m-%d'))


def fetch_fng_history():
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=15)
    out = {}
    for row in r.json()["data"]:
        d = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
        out[d.strftime('%Y-%m-%d')] = float(row["value"])
    return out


def _end_ms(exchange):
    """Endzeitpunkt des Backtest-Fensters in ms (Default: jetzt)."""
    if not BACKTEST_END:
        return exchange.milliseconds()
    dt = datetime.strptime(BACKTEST_END, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_candles(exchange, symbol, days):
    end = _end_ms(exchange)
    since = end - days * 24 * 3600 * 1000
    all_c = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, '5m', since=since, limit=1000)
        if not batch:
            break
        all_c.extend(batch)
        if batch[-1][0] == since:
            break
        since = batch[-1][0] + 1
        if len(batch) < 1000 or since >= end:
            break
    df = pd.DataFrame(all_c, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='ts')
    df = df[df['ts'] <= end]          # nichts nach dem Fensterende verwenden
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.reset_index(drop=True)


def build_signal_grid(base_cfg, candles, fng_hist, n):
    """Pass 1: Signale für jede (Kerze, Symbol)-Kombination aufzeichnen."""
    cfg = dict(base_cfg)
    cfg['min_confluence_score'] = 0.0
    strat = ConfluenceStrategy(cfg)
    strat.add_factor(MultiTimeframeTrendFactor())
    strat.add_factor(MomentumFactor())
    strat.add_factor(VolatilityFilter())
    strat.add_factor(BreakoutFactor())
    strat.add_factor(VolumeConfirmationFactor())
    strat.add_factor(MeanReversionFactor())
    strat.add_factor(MacroNewsFilter())
    sent = HistoricalSentimentFactor(fng_hist)
    strat.add_factor(sent)

    grid = {}   # (i, sym) -> signal-dict
    regimes = Counter()
    t0 = _time.time()
    for i in range(WINDOW, n):
        ts = candles[SYMBOLS[0]]['timestamp'].iloc[i]
        sent.current_ts = ts
        for sym in SYMBOLS:
            window = candles[sym].iloc[i - WINDOW:i + 1]
            price = window['close'].iloc[-1]
            signal = strat.analyze(sym, window, price)
            if strat._last_regime:
                regimes[strat._last_regime.name] += 1
            if signal is not None:
                vf = signal.factor_breakdown.get('volatility_filter')
                atr = vf.metadata.get('atr_pct') if vf and vf.metadata else None
                grid[(i, sym)] = {
                    'score': signal.confluence_score, 'conf': signal.confidence,
                    'dir': signal.direction, 'lev': signal.suggested_leverage,
                    'atr': atr,
                }
        if (i - WINDOW) % 2000 == 0:
            print(f"[Pass 1] {i}/{n} | Signale {len(grid)} | {_time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
    return grid, regimes


def simulate(threshold, tp_mult, sl_mult, cooldown, candles, grid, n):
    """Pass 2: Portfolio-Simulation über das Signal-Raster."""
    balance = START_CAPITAL
    positions = {}
    trades = []
    equity_curve = []
    day_start_equity = {}
    trading_paused_day = None
    blocked_until = {}   # symbol -> Kerzen-Index, bis zu dem nach Verlust pausiert wird

    for i in range(WINDOW, n):
        ts = candles[SYMBOLS[0]]['timestamp'].iloc[i]
        day = ts.strftime('%Y-%m-%d')

        equity = balance
        for sym, pos in positions.items():
            price_now = candles[sym]['close'].iloc[i]
            chg = (price_now - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'long' else -1)
            equity += pos['margin'] + pos['margin'] * pos['lev'] * chg
        equity_curve.append(equity)
        day_start_equity.setdefault(day, equity)

        if (day_start_equity[day] - equity) / day_start_equity[day] > MAX_DAILY_DD:
            trading_paused_day = day

        # Exits (SL vor TP, intrabar)
        for sym in list(positions.keys()):
            pos = positions[sym]
            row = candles[sym].iloc[i]
            hit, exit_price = None, None
            if pos['side'] == 'long':
                if row['low'] <= pos['sl']:
                    hit, exit_price = 'SL', pos['sl']
                elif row['high'] >= pos['tp']:
                    hit, exit_price = 'TP', pos['tp']
            else:
                if row['high'] >= pos['sl']:
                    hit, exit_price = 'SL', pos['sl']
                elif row['low'] <= pos['tp']:
                    hit, exit_price = 'TP', pos['tp']
            if hit:
                chg = (exit_price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'long' else -1)
                notional = pos['margin'] * pos['lev']
                pnl = notional * chg - notional * TAKER_FEE * 2
                balance += pos['margin'] + pnl
                trades.append({'symbol': sym, 'side': pos['side'], 'pnl': pnl, 'exit': hit})
                del positions[sym]
                if pnl < 0 and cooldown > 0:
                    blocked_until[sym] = i + cooldown

        # Entries
        if trading_paused_day == day:
            continue
        for sym in SYMBOLS:
            if sym in positions or len(positions) >= MAX_CONCURRENT:
                continue
            if blocked_until.get(sym, -1) > i:
                continue
            sig = grid.get((i, sym))
            if sig is None or sig['score'] < threshold:
                continue
            if LONG_ONLY and sig['dir'] != 'long':
                continue          # Spot-Boerse: Shorts nicht ausfuehrbar
            price = candles[sym]['close'].iloc[i]
            # TP/SL wie im Aggregator: ATR-verankert mit Fee-Floors
            atr = sig['atr'] or 0.0
            tp_pct = max(tp_mult * atr, MIN_TP_PCT)
            sl_pct = max(sl_mult * atr, MIN_SL_PCT)
            d = 1 if sig['dir'] == 'long' else -1
            tp = price * (1 + d * tp_pct)
            sl = price * (1 - d * sl_pct)
            margin = equity * MARGIN_PCT
            margin = max(max(10.0, equity * 0.15), min(equity * 0.25, margin))
            if margin > balance or balance < 20:
                continue
            # Notional muss die Mindestordergroesse der Boerse erreichen
            if MIN_ORDER_AMOUNT and margin * sig['lev'] < MIN_ORDER_AMOUNT:
                continue
            balance -= margin
            positions[sym] = {'side': sig['dir'], 'entry': price, 'margin': margin,
                              'lev': sig['lev'], 'tp': tp, 'sl': sl}

    for sym, pos in positions.items():
        price = candles[sym]['close'].iloc[n - 1]
        chg = (price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'long' else -1)
        notional = pos['margin'] * pos['lev']
        pnl = notional * chg - notional * TAKER_FEE * 2
        balance += pos['margin'] + pnl
        trades.append({'symbol': sym, 'side': pos['side'], 'pnl': pnl, 'exit': 'EOD'})

    eq = pd.Series(equity_curve)
    returns = eq.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(288 * 365) if len(returns) > 1 and returns.std() > 0 else 0.0
    peak = eq.cummax()
    max_dd = float(((peak - eq) / peak).max()) if len(eq) else 0.0
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    return {
        'threshold': threshold, 'tp_mult': tp_mult, 'sl_mult': sl_mult, 'cooldown': cooldown,
        'final': balance,
        'return_pct': (balance - START_CAPITAL) / START_CAPITAL,
        'sharpe': float(sharpe),
        'max_dd': max_dd,
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else float('inf'),
        'exits': dict(Counter(t['exit'] for t in trades)),
        'by_symbol': dict(Counter(t['symbol'] for t in trades)),
        'longs': sum(1 for t in trades if t['side'] == 'long'),
        'shorts': sum(1 for t in trades if t['side'] == 'short'),
    }


def main():
    with open('config/settings.yaml') as f:
        settings = yaml.safe_load(f)
    base_cfg = settings['strategies']['confluence']

    print("Lade Fear&Greed-Historie...", flush=True)
    fng = fetch_fng_history()
    print(f"Lade {DAYS} Tage 5m-Kerzen via Binance...", flush=True)
    ex = ccxt.binance({'enableRateLimit': True})
    candles = {sym: fetch_candles(ex, sym, DAYS) for sym in SYMBOLS}
    n = min(len(df) for df in candles.values())
    for sym, df in candles.items():
        print(f"  {sym}: {len(df)} Kerzen", flush=True)

    bh = [(candles[s]['close'].iloc[-1] - candles[s]['close'].iloc[WINDOW]) / candles[s]['close'].iloc[WINDOW]
          for s in SYMBOLS]
    buy_hold = sum(bh) / len(bh)

    cache_path = os.environ.get('GRID_CACHE_PATH', '/tmp/confluence_grid.pkl')
    if os.environ.get('GRID_CACHE') == '1' and os.path.exists(cache_path):
        print("Pass 1: lade Grid aus Cache...", flush=True)
        with open(cache_path, 'rb') as fh:
            grid, regimes, candles, n = pickle.load(fh)
    else:
        print("Pass 1: Signal-Raster aufbauen...", flush=True)
        grid, regimes = build_signal_grid(base_cfg, candles, fng, n)
        with open(cache_path, 'wb') as fh:
            pickle.dump((grid, regimes, candles, n), fh)
    scores = pd.Series([g['score'] for g in grid.values()])
    print(f"Signale mit Richtung: {len(grid)} von {(n - WINDOW) * len(SYMBOLS)} Bewertungen", flush=True)
    print(f"Score-Verteilung: mean={scores.mean():.3f} p50={scores.quantile(.5):.3f} "
          f"p75={scores.quantile(.75):.3f} p90={scores.quantile(.9):.3f} max={scores.max():.3f}", flush=True)
    print(f"Regime-Verteilung: {dict(regimes)}", flush=True)

    # Schwellen aus der beobachteten Verteilung (Perzentile) statt fixer Liste
    ths = sorted({round(float(scores.quantile(q)), 3) for q in (0.8, 0.9, 0.95, 0.97)})
    print(f"Getestete Schwellen (Perzentile der Verteilung): {ths}", flush=True)
    results = [simulate(th, tp, sl, cd, candles, grid, n)
               for th in ths for (tp, sl) in TPSL_PROFILES for cd in COOLDOWNS]

    print("\n" + "=" * 80)
    print(f"SCHWELLWERT-SWEEP  |  {DAYS} Tage 5m  |  Buy&Hold (Ø 3 Coins): {buy_hold:+.1%}")
    print("=" * 80)
    print(f"{'Schwelle':>8} {'TPxATR':>7} {'SLxATR':>7} {'CD':>3} {'Return':>8} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'MaxDD':>7} {'Sharpe':>8} {'L/S':>9}")
    for r in sorted(results, key=lambda x: (-x['win_rate'])):
        print(f"{r['threshold']:>8} {r['tp_mult']:>7} {r['sl_mult']:>7} {r['cooldown']:>3} {r['return_pct']:>7.1%} {r['n_trades']:>7} {r['win_rate']:>7.1%} "
              f"{r['profit_factor']:>6.2f} {r['max_dd']:>6.1%} {r['sharpe']:>8.2f} "
              f"{r['longs']:>4}/{r['shorts']}")

    with open('/tmp/confluence_sweep_result.json', 'w') as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
