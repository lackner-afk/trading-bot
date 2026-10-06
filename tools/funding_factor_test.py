#!/usr/bin/env python3
"""
Prueft, ob die Funding Rate den Ausgang eines Trades vorhersagt.

These: Hohe positive Funding Rates heissen, dass Long-Positionen fuer ihre
Position zahlen — ein Zeichen ueberhitzter Positionierung, das Ruecksetzern
vorausgeht. Waere das so, muesste ein LONG-Einstieg bei hoher Funding Rate
schlechter laufen als bei niedriger: also ein negativer IC.

Wichtig zur Methodik: Funding wird alle 8 h fixiert, ueber 90 Tage sind das 270
Werte je Symbol. Wuerde man pro 5m-Kerze testen, haette man 45.000 stark
autokorrelierte Beobachtungen und dadurch viel zu optimistische p-Werte. Darum
genau **eine Beobachtung je Funding-Periode**.

Aufruf:
    DAYS=90 BACKTEST_END=2026-05-20 TPSL="12:5" python3 tools/funding_factor_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import ccxt
from scipy import stats

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import tools.backtest_confluence as bt
from tools.factor_analysis import outcome_for

PERP = {'BTC/USDT': 'BTC/USDT:USDT', 'ETH/USDT': 'ETH/USDT:USDT', 'SOL/USDT': 'SOL/USDT:USDT'}


def atr_pct_at(df, i, period=14):
    """ATR relativ zum Schlusskurs an Kerze i."""
    lo = max(0, i - period)
    w = df.iloc[lo:i + 1]
    if len(w) < 2:
        return None
    prev = w['close'].shift()
    tr = pd.concat([w['high'] - w['low'],
                    (w['high'] - prev).abs(),
                    (w['low'] - prev).abs()], axis=1).max(axis=1)
    atr = tr.mean()
    close = w['close'].iloc[-1]
    return float(atr / close) if close > 0 and not pd.isna(atr) else None


def main():
    tp_mult, sl_mult = bt.TPSL_PROFILES[0]
    spot = ccxt.binance({'enableRateLimit': True})
    fut = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

    end = bt._end_ms(spot)
    since = end - bt.DAYS * 24 * 3600 * 1000

    rows = []
    for sym in bt.SYMBOLS:
        df = bt.fetch_candles(spot, sym, bt.DAYS)
        ts = df['ts'].values

        hist = fut.fetch_funding_rate_history(PERP[sym], since=since, limit=1000)
        hist = [h for h in hist if h['timestamp'] <= end]
        print(f"  {sym}: {len(df)} Kerzen, {len(hist)} Funding-Perioden", flush=True)

        for h in hist:
            # Kerze, die zum Zeitpunkt der Funding-Fixierung offen war
            i = int(np.searchsorted(ts, h['timestamp'], side='right')) - 1
            if i < bt.WINDOW or i >= len(df) - 1:
                continue
            atr = atr_pct_at(df, i)
            if not atr:
                continue
            tp_pct = max(tp_mult * atr, 0.010)
            sl_pct = max(sl_mult * atr, 0.0030)
            # Immer LONG bewerten: dann heisst negativer IC "hohe Funding = schlecht fuer Long"
            win = outcome_for(df, i, 'long', tp_pct, sl_pct)
            if win is None:
                continue
            rows.append({'symbol': sym, 'rate': float(h['fundingRate']), 'win': win})

    if len(rows) < 100:
        sys.exit(f"Zu wenige Beobachtungen ({len(rows)}).")

    d = pd.DataFrame(rows)
    ic, p = stats.spearmanr(d['rate'], d['win'])
    print(f"\nBeobachtungen: {len(d)} (je eine pro Funding-Periode und Symbol)")
    print(f"Funding Rate: Median {d['rate'].median():+.5f}, "
          f"p10 {d['rate'].quantile(.1):+.5f}, p90 {d['rate'].quantile(.9):+.5f}")
    print(f"Trefferquote LONG gesamt: {d['win'].mean():.1%}\n")

    print(f"IC (Funding vs. Long-Erfolg): {ic:+.3f}   p={p:.2e}")
    if p < 0.01 and ic <= -0.03:
        print("  -> Contrarian bestaetigt: hohe Funding Rate schadet Longs")
    elif p < 0.01 and ic >= 0.03:
        print("  -> umgekehrt: hohe Funding Rate begleitet erfolgreiche Longs")
    else:
        print("  -> kein belastbarer Zusammenhang")

    d['q'] = pd.qcut(d['rate'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    g = d.groupby('q', observed=True).agg(n=('win', 'size'), wr=('win', 'mean'),
                                          rate=('rate', 'median'))
    print("\nQuintil  n    Funding(Median)  Trefferquote LONG")
    for q, r in g.iterrows():
        print(f"  Q{q}   {int(r['n']):4d}   {r['rate']:+.5f}        {r['wr']:.1%}")


if __name__ == '__main__':
    main()
