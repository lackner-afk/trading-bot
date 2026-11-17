"""
Belastungsprobe für die Tagestrend-Regeln: Ist das Backtest-Ergebnis mehr als
Glück beim Zeitpunkt? Rechnet mit demselben Simulator wie tools/backtest_daily.py
(gleiche Gebühr, Mindestorder, Positionsgröße).

  1. Zufalls-Baseline: dieselben Halte- und Pausenphasen je Coin (gleiche Zahl
     Trades, gleiche Zeit im Markt), nur in zufälliger Reihenfolge.
  2. Parameterfläche: Donchian-Einstieg N, Ausstieg N/2 — steht 20/10 auf einem
     Plateau oder auf einer Spitze?
  3. Zeit-Split: erste gegen zweite Hälfte, jede frisch mit Startkapital.
  4. Walk-forward: N auf den 730 Tagen davor wählen, 365 Tage handeln.

Umgebung wie backtest_daily.py, dazu:
  VARIANT=don20          (don20 | don55 | vote3 | sma50 … — Regel für 1. und 3.)
  RUNS=400               (Zufallsläufe)
  SEED=42
  BINANCE_PUBLIC_API=https://data-api.binance.vision/api/v3
                         (anderer Marktdaten-Endpoint, falls api.binance.com
                          am Standort gesperrt ist)

Beispiel (Live-Konfiguration Panda-9):
  POSITION_FRACTION=0.2 MAX_POSITIONS=2 python tools/robustness_daily.py
"""
import os
import sys

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import backtest_daily as bt  # noqa: E402

VARIANT = os.environ.get('VARIANT', 'don20')
RUNS = int(os.environ.get('RUNS', 400))
SEED = int(os.environ.get('SEED', 42))
ENTRY_DAYS = [5, 10, 15, 20, 30, 40, 55, 80, 100]

_base_signals = bt.signals


def signals(df: pd.DataFrame, variant: str) -> pd.Series:
    """Wie backtest_daily.signals, dazu 'dN' = Donchian N / N/2 und fertige Serien."""
    if isinstance(variant, dict):
        return variant[id(df)]
    if variant.startswith('d') and variant[1:].isdigit():
        n = int(variant[1:])
        return bt.donchian_hold(df['high'], df['low'], df['close'], n, max(n // 2, 2))
    return _base_signals(df, variant)


bt.signals = signals


def spells(sig: pd.Series) -> list:
    """Zerlegt eine Halte-Serie in Phasen: [(halten?, Länge in Tagen), …]."""
    values = sig.to_numpy().astype(bool)
    out, i = [], 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        out.append((bool(values[i]), j - i))
        i = j
    return out


def shuffled(sig: pd.Series, rng) -> pd.Series:
    """Gleiche Phasen, zufällige Reihenfolge — Zahl der Trades und Zeit im Markt bleiben."""
    phases = spells(sig)
    ins = [n for hold, n in phases if hold]
    outs = [n for hold, n in phases if not hold]
    rng.shuffle(ins)
    rng.shuffle(outs)
    first_in = bool(rng.integers(2)) if len(ins) == len(outs) else len(ins) > len(outs)
    a, b = (ins, outs) if first_in else (outs, ins)
    values = []
    for k in range(max(len(a), len(b))):
        if k < len(a):
            values += [first_in] * a[k]
        if k < len(b):
            values += [not first_in] * b[k]
    return pd.Series(values[:len(sig)], index=sig.index)


def random_baseline(data: dict, start, end, variant: str, runs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real = bt.simulate(data, variant, start, end)
    base = {s: signals(data[s], variant) for s in bt.SYMBOLS}
    totals, sharpes, dds = [], [], []
    for _ in range(runs):
        fake = {
            id(data[s]): shuffled(base[s].loc[start:end], rng).reindex(base[s].index, fill_value=False)
            for s in bt.SYMBOLS
        }
        r = bt.simulate(data, fake, start, end)
        totals.append(r['total'])
        sharpes.append(r['sharpe'])
        dds.append(r['max_dd'])
    totals, sharpes, dds = np.array(totals), np.array(sharpes), np.array(dds)
    return {
        'real': real,
        'median_total': float(np.median(totals)),
        'q05': float(np.quantile(totals, 0.05)),
        'q95': float(np.quantile(totals, 0.95)),
        'median_sharpe': float(np.median(sharpes)),
        'median_dd': float(np.median(dds)),
        # Anteil der Zufallsläufe, die mindestens so gut sind wie die Regel
        'p_total': float(np.mean(totals >= real['total'])),
        'p_sharpe': float(np.mean(sharpes >= real['sharpe'])),
        'p_dd': float(np.mean(dds <= real['max_dd'])),
    }


def walk_forward(data: dict, start, end, train_days: int = 730, test_days: int = 365) -> list:
    rows = []
    t0 = start + pd.Timedelta(days=train_days)
    while t0 + pd.Timedelta(days=90) < end:
        t1 = min(t0 + pd.Timedelta(days=test_days), end)
        best = max(ENTRY_DAYS, key=lambda n: bt.simulate(
            data, f'd{n}', t0 - pd.Timedelta(days=train_days), t0)['sharpe'])
        rows.append({
            'from': t0, 'to': t1, 'chosen': best,
            'wf': bt.simulate(data, f'd{best}', t0, t1)['total'],
            'fix20': bt.simulate(data, 'd20', t0, t1)['total'],
            'hold': bt.simulate(data, 'hold', t0, t1)['total'],
        })
        t0 = t1
    return rows


def _fmt(r: dict) -> str:
    return f"{r['total']:+7.0%}  Sharpe {r['sharpe']:5.2f}  Rückgang {r['max_dd']:4.0%}"


def main():
    ex = ccxt.binance({'enableRateLimit': True})
    api = os.environ.get('BINANCE_PUBLIC_API')
    if api:
        ex.urls['api']['public'] = api
        # nur Spot-Märkte laden — der Ersatz-Endpoint kennt keine Futures
        ex.options['fetchMarkets'] = ['spot']
        ex.options['fetchCurrencies'] = False
    # ccxt übergeht Proxy-Variablen der Umgebung — hinter einem Proxy ausdrücklich setzen
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if proxy:
        ex.https_proxy = proxy
    print(f"Lade Tageskerzen ab {bt.SINCE} …", flush=True)
    data = {s: bt.fetch_daily(ex, s) for s in bt.SYMBOLS}
    common = data[bt.SYMBOLS[0]].index
    for s in bt.SYMBOLS[1:]:
        common = common.intersection(data[s].index)
    data = {s: df.loc[common] for s, df in data.items()}
    start, end = common[bt.WARMUP], common[-1]
    print(f"{', '.join(bt.SYMBOLS)} · {start.date()} bis {end.date()} · Gebühr {bt.TAKER_FEE:.2%} · "
          f"Start {bt.START_CAPITAL:.0f} € · je Coin {bt.POSITION_FRACTION:.0%} · höchstens {bt.MAX_POSITIONS}\n")

    print(f"1. Zufalls-Baseline für {VARIANT} ({RUNS} Läufe, gleiche Phasenlängen)")
    nb = random_baseline(data, start, end, VARIANT, RUNS, SEED)
    print(f"   Regel:  {_fmt(nb['real'])}  ({nb['real']['trades']} Käufe+Verkäufe, "
          f"Gebühren {nb['real']['fees']:.1f} €)")
    print(f"   Zufall: Median {nb['median_total']:+.0%} (5–95 %: {nb['q05']:+.0%} bis {nb['q95']:+.0%}), "
          f"Sharpe {nb['median_sharpe']:.2f}, Rückgang {nb['median_dd']:.0%}")
    print(f"   Anteil Zufallsläufe mindestens so gut: Rendite {nb['p_total']:.1%}, "
          f"Sharpe {nb['p_sharpe']:.1%}, Rückgang {nb['p_dd']:.1%}")
    print("   (unter ~5 % = kaum durch Zufall erklärbar; 30 % = nicht von Zufall unterscheidbar)\n")

    print("2. Parameterfläche Donchian N / N/2")
    for n in ENTRY_DAYS:
        print(f"   N={n:3d}: {_fmt(bt.simulate(data, f'd{n}', start, end))}")
    print(f"   Halten: {_fmt(bt.simulate(data, 'hold', start, end))}\n")

    mid = common[bt.WARMUP + (len(common) - bt.WARMUP) // 2]
    print(f"3. Zeit-Split bei {mid.date()} (jede Hälfte frisch mit Startkapital)")
    for v in ['hold', VARIANT]:
        print(f"   {v:6s} 1. Hälfte {_fmt(bt.simulate(data, v, start, mid))} | "
              f"2. Hälfte {_fmt(bt.simulate(data, v, mid, end))}")

    print("\n4. Walk-forward (N auf 730 Tagen gewählt, 365 Tage gehandelt)")
    rows = walk_forward(data, start, end)
    chain = {'wf': 1.0, 'fix20': 1.0, 'hold': 1.0}
    for r in rows:
        for k in chain:
            chain[k] *= 1 + r[k]
        print(f"   {r['from'].date()} – {r['to'].date()}  N={r['chosen']:3d}: gewählt {r['wf']:+6.1%} | "
              f"fest 20 {r['fix20']:+6.1%} | Halten {r['hold']:+6.1%}")
    if rows:
        print(f"   verkettet: gewählt {chain['wf'] - 1:+.0%} | fest 20 {chain['fix20'] - 1:+.0%} | "
              f"Halten {chain['hold'] - 1:+.0%}")


if __name__ == '__main__':
    sys.exit(main())
