#!/usr/bin/env python3
"""
Misst, welcher Confluence-Faktor tatsaechlich Vorhersagekraft hat.

Fuer jedes Signal wird der reine Ausgang bestimmt: laeuft der Kurs ab dieser
Kerze zuerst ins Take-Profit oder ins Stop-Loss? Ohne Portfolio-Effekte (keine
Positionsgrenze, kein Cooldown) — es geht allein um die Signalguete.

Danach werden die Signale je Faktor in Quintile nach dessen Score sortiert. Ein
Faktor mit Vorhersagekraft zeigt einen Gradienten: hohe Scores -> hoehere
Trefferquote. Ein flacher Verlauf heisst, der Faktor traegt nichts bei.

Aufruf (Umgebung wie backtest_confluence.py):
    TPSL="12:5" DAYS=90 GRID_CACHE=1 GRID_CACHE_PATH=/tmp/fa_w1.pkl \
        python3 tools/factor_analysis.py
"""
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import ccxt
from scipy import stats

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import tools.backtest_confluence as bt


# Wie viele Kerzen ein Signal Zeit bekommt, TP oder SL zu erreichen.
# 288 x 5m = 24 h; bei 1h-Kerzen sind 120 Bars = 5 Tage sinnvoller, weil
# derselbe ATR-Vielfache-Abstand dort laenger braucht.
HORIZON_BARS = int(os.environ.get('HORIZON_BARS', 288))


def outcome_for(candles, i, direction, tp_pct, sl_pct, horizon=None):
    """
    Ausgang eines Signals: 1 = TP zuerst, 0 = SL zuerst, None = keins innerhalb
    des Horizonts (288 Kerzen a 5m = 24 h).

    Wird in derselben Kerze sowohl TP als auch SL beruehrt, zaehlt der Ausgang
    konservativ als Verlust — welcher zuerst kam, ist aus OHLC nicht ableitbar.
    """
    entry = candles['close'].iloc[i]
    d = 1 if direction == 'long' else -1
    tp = entry * (1 + d * tp_pct)
    sl = entry * (1 - d * sl_pct)

    horizon = HORIZON_BARS if horizon is None else horizon
    hi = candles['high'].values
    lo = candles['low'].values
    end = min(i + 1 + horizon, len(candles))
    for j in range(i + 1, end):
        if direction == 'long':
            hit_sl = lo[j] <= sl
            hit_tp = hi[j] >= tp
        else:
            hit_sl = hi[j] >= sl
            hit_tp = lo[j] <= tp
        if hit_sl:
            return 0            # konservativ: Verlust schlaegt Gewinn in derselben Kerze
        if hit_tp:
            return 1
    return None


def quintile_table(rows, factor):
    """Trefferquote je Score-Quintil eines Faktors."""
    vals = [(r['factors'][factor][0], r['win']) for r in rows if factor in r['factors']]
    if len(vals) < 100:
        return None
    df = pd.DataFrame(vals, columns=['score', 'win'])
    # Viele Faktoren haben stark gebundene Werte -> rank statt qcut auf Rohwerten
    df['q'] = pd.qcut(df['score'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    g = df.groupby('q', observed=True).agg(n=('win', 'size'), wr=('win', 'mean'),
                                           lo=('score', 'min'), hi=('score', 'max'))
    return g


def main():
    tp_mult, sl_mult = bt.TPSL_PROFILES[0]
    print(f"Faktor-Analyse | {bt.DAYS} Tage {bt.TIMEFRAME} | TP {tp_mult}xATR / SL {sl_mult}xATR "
          f"| Horizont {HORIZON_BARS} Kerzen | Ende {bt.BACKTEST_END or 'jetzt'}", flush=True)

    fng = bt.fetch_fng_history()
    ex = ccxt.binance({'enableRateLimit': True})
    candles = {}
    for sym in bt.SYMBOLS:
        candles[sym] = bt.fetch_candles(ex, sym, bt.DAYS)
        print(f"  {sym}: {len(candles[sym])} Kerzen", flush=True)
    n = min(len(df) for df in candles.values())

    cache = os.environ.get('GRID_CACHE_PATH', '/tmp/factor_grid.pkl')
    if os.environ.get('GRID_CACHE') == '1' and os.path.exists(cache):
        print("Grid aus Cache...", flush=True)
        with open(cache, 'rb') as fh:
            grid, _ = pickle.load(fh)
    else:
        print("Grid aufbauen...", flush=True)
        cfg = bt.yaml.safe_load(open('config/settings.yaml'))['strategies']['confluence']
        grid, regimes = bt.build_signal_grid(cfg, candles, fng, n)
        with open(cache, 'wb') as fh:
            pickle.dump((grid, regimes), fh)

    # Ausgang je Signal bestimmen
    rows = []
    for (i, sym), sig in grid.items():
        if not sig.get('factors') or not sig.get('atr'):
            continue
        tp_pct = max(tp_mult * sig['atr'], 0.010)
        sl_pct = max(sl_mult * sig['atr'], 0.0030)
        win = outcome_for(candles[sym], i, sig['dir'], tp_pct, sl_pct)
        if win is None:
            continue
        factors = dict(sig['factors'])
        factors['** CONFLUENCE (gesamt) **'] = (sig['score'], sig['dir'])
        rows.append({'win': win, 'score': sig['score'], 'dir': sig['dir'],
                     'factors': factors})

    if not rows:
        print("Keine auswertbaren Signale."); return

    base = np.mean([r['win'] for r in rows])
    print(f"\nAuswertbare Signale: {len(rows)} | Trefferquote gesamt: {base:.1%}")
    print(f"(Break-even bei TP {tp_mult}/SL {sl_mult} plus 0,5 % Gebuehren: ca. "
          f"{(sl_mult + 1.7) / (tp_mult + sl_mult):.0%})\n")

    print("=" * 78)
    print("Faktor-Guete: IC = Rangkorrelation zwischen Score und Ausgang")
    print("=" * 78)

    results = []
    for factor in sorted({f for r in rows for f in r['factors']}):
        vals = [(r['factors'][factor][0], r['win']) for r in rows if factor in r['factors']]
        if len(vals) < 200:
            continue
        scores = np.array([v[0] for v in vals], dtype=float)
        wins = np.array([v[1] for v in vals], dtype=float)
        if np.allclose(scores, scores[0]):
            results.append((0.0, 1.0, factor, None, "konstant"))
            continue
        ic, pval = stats.spearmanr(scores, wins)
        g = quintile_table(rows, factor)
        results.append((ic, pval, factor, g, None))

    for ic, pval, factor, g, note in sorted(results, key=lambda x: -abs(x[0])):
        if note == "konstant":
            print(f"\n{factor:26s} liefert einen konstanten Score — keine Aussage moeglich")
            continue
        # Signifikanz: nur bei p < 0.01 ist der Zusammenhang belastbar
        if pval >= 0.01:
            verdict = "kein belastbarer Zusammenhang"
        elif ic >= 0.03:
            verdict = "traegt bei"
        elif ic <= -0.03:
            verdict = "KONTRAPRODUKTIV — Score invers zum Erfolg"
        else:
            verdict = "signifikant, aber zu schwach zum Tragen"
        bar = " ".join(f"{w:.0%}" for w in g['wr']) if g is not None else "-"
        print(f"\n{factor:26s} IC {ic:+.3f}  p={pval:.1e}   [{verdict}]")
        print(f"{'':26s} Trefferquote Q1->Q5: {bar}")

    print("\n" + "=" * 78)
    print("IC ist die Spearman-Korrelation zwischen Faktor-Score und Ausgang (0/1).")
    print("Positiv = hoher Score sagt Gewinner voraus. In der Praxis gilt ein IC ab")
    print("etwa 0,03 als brauchbar, ab 0,05 als gut. p < 0,01 = statistisch belastbar.")
    print("Q1->Q5 zeigt zusaetzlich, ob der Zusammenhang gleichmaessig verlaeuft.")


if __name__ == '__main__':
    main()
