#!/usr/bin/env python3
"""
Null-Modell: schlagen die Signale ueberhaupt den Muenzwurf?

Vergleicht die Trefferquote der Confluence-Signale mit zufaelligen Einstiegen —
gleiche Kerzen, gleiche TP/SL-Regel, gleicher Horizont, gleiche ATR-Verteilung,
aber Zeitpunkt und Richtung ausgewuerfelt.

Das ist die Referenz, gegen die sich jede Strategie-Idee zuerst messen lassen
sollte. Eine Trefferquote von 30 % klingt schlecht und eine von 60 % gut — was
sie wirklich wert sind, zeigt erst der Abstand zum Zufall bei demselben
Chance-Risiko-Verhaeltnis.

Aufruf (Umgebung wie backtest_confluence.py; das Grid muss vorliegen):
    TIMEFRAME=5m DAYS=90 BACKTEST_END=2026-05-20 TPSL="12:5" \
        GRID_CACHE_PATH=/tmp/fa_w2.pkl python3 tools/null_model.py
"""
import os
import sys
import pickle

import numpy as np
import ccxt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import tools.backtest_confluence as bt
from tools.factor_analysis import outcome_for

# Fester Startwert, damit ein Lauf reproduzierbar bleibt
SEED = int(os.environ.get('SEED', 20260825))
ROUNDS = int(os.environ.get('ROUNDS', 3))


def main():
    tp_mult, sl_mult = bt.TPSL_PROFILES[0]
    rng = np.random.default_rng(SEED)

    ex = ccxt.binance({'enableRateLimit': True})
    candles = {s: bt.fetch_candles(ex, s, bt.DAYS) for s in bt.SYMBOLS}
    n = min(len(df) for df in candles.values())

    cache = os.environ.get('GRID_CACHE_PATH', '/tmp/confluence_grid.pkl')
    if not os.path.exists(cache):
        sys.exit(f"Grid fehlt: {cache} — erst factor_analysis.py oder "
                 f"backtest_confluence.py mit diesem Pfad laufen lassen.")
    with open(cache, 'rb') as fh:
        grid, _ = pickle.load(fh)

    def tp_sl(atr):
        return max(tp_mult * atr, 0.010), max(sl_mult * atr, 0.0030)

    # --- echte Signale ---
    real, atrs = [], []
    for (i, sym), sig in grid.items():
        if not sig.get('atr'):
            continue
        tp_pct, sl_pct = tp_sl(sig['atr'])
        w = outcome_for(candles[sym], i, sig['dir'], tp_pct, sl_pct)
        if w is not None:
            real.append(w)
            atrs.append(sig['atr'])

    if not real:
        sys.exit("Keine auswertbaren Signale im Grid.")

    # --- Zufall: gleiche Anzahl und ATR-Verteilung, Zeitpunkt und Richtung gewuerfelt ---
    syms = list(candles)
    rounds = []
    for _ in range(ROUNDS):
        rand = []
        for atr in atrs:
            sym = syms[rng.integers(len(syms))]
            i = int(rng.integers(bt.WINDOW, n - 1))
            direction = 'long' if rng.random() < 0.5 else 'short'
            tp_pct, sl_pct = tp_sl(atr)
            w = outcome_for(candles[sym], i, direction, tp_pct, sl_pct)
            if w is not None:
                rand.append(w)
        rounds.append(np.mean(rand))

    r = float(np.mean(real))
    z = float(np.mean(rounds))
    # Standardfehler der Differenz zweier Anteile (Zufallsseite ueber alle Runden)
    n_rand = len(rand) * ROUNDS
    se = (r * (1 - r) / len(real) + z * (1 - z) / n_rand) ** 0.5
    sigma = (r - z) / se if se > 0 else 0.0

    print(f"{bt.TIMEFRAME} | {bt.DAYS} Tage | Ende {bt.BACKTEST_END or 'jetzt'} "
          f"| TP {tp_mult}x / SL {sl_mult}xATR")
    print(f"  Confluence-Signale: {r:6.1%}   (n={len(real)})")
    print(f"  Zufallseinstiege:   {z:6.1%}   ({ROUNDS} Runden, je n={len(rand)})")
    print(f"  Differenz:          {r - z:+6.1%}   ({sigma:+.1f} Standardfehler)")
    if sigma > 2:
        verdict = "besser als Zufall"
    elif sigma < -2:
        verdict = "SCHLECHTER als Zufall"
    else:
        verdict = "nicht von Zufall unterscheidbar"
    print(f"  -> {verdict}")
    # Wieviel fehlt bis zur Gewinnschwelle?
    need = (sl_mult + 1.7) / (tp_mult + sl_mult)
    print(f"  Break-even bei diesen Multiplikatoren inkl. 0,5 % Gebuehren: {need:.0%} "
          f"-> es fehlen {need - r:+.1%}")


if __name__ == '__main__':
    main()
