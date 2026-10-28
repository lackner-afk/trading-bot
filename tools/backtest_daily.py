"""
Tageskerzen-Backtest: einfacher Trendfilter gegen Halten (Test T1 aus
docs/KRYPTO_BOTS_IM_VERGLEICH.md).

Je Coin ein Drittel des Kapitals. Entscheidung am Tagesschluss, Ausführung
zum Eröffnungskurs des Folgetags (kein Blick in die Zukunft). Gebühr auf
jeden Kauf und Verkauf, Mindestordergröße wie bei Fusion.

Die Varianten sind vorab festgelegt und werden ALLE berichtet — nicht die
beste heraussuchen:
  hold        Drittel je Coin kaufen und halten
  sma50/100/200   halten, solange Schlusskurs > gleitender Durchschnitt
  mom30/90    halten, solange die Coin über 30/90 Tage gestiegen ist
  vote3       halten, solange mindestens 3 der 5 Regeln oben halten wollen
              (am 14.10.2026 nachträglich ergänzt, nach T1 — Kandidat für den
              Live-Betrieb, weil keine Einzelregel überall die beste war)
  don20/don55 Donchian-Ausbruch: Kauf über dem 20-/55-Tage-Hoch, Verkauf unter
              dem 10-/20-Tage-Tief (Turtle-Systeme 1 und 2; am 28.10.2026 vor
              dem Lauf festgelegt, Kandidat für den Umbau von Panda-9)

Umgebung:
  TAKER_FEE=0.0025  START_CAPITAL=100  MIN_ORDER_AMOUNT=30
  POSITION_FRACTION=0.2  (Anteil des Kapitals je Coin; leer = gleichmäßig
                          aufteilen. Live gilt die harte 20-%-Grenze des
                          RiskManagers.)
  MAX_POSITIONS=1   (höchstens so viele Coins gleichzeitig; leer = alle.
                     Wollen mehr Coins rein, gewinnt die Reihenfolge in SYMBOLS)
  SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
  SINCE=2018-01-01  (Startdatum der Datenabfrage; Bewertung beginnt, sobald
                     alle Coins 200 Tage Vorlauf haben)
"""
import importlib.util
import os
import sys
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

# Dieselben Regeln wie der Live-Bot — sonst prüft der Backtest etwas anderes.
# Direkt aus der Datei laden: `import strategies.daily_trend` zöge über
# strategies/__init__.py das ganze Paket samt ML-Teil mit.
_spec = importlib.util.spec_from_file_location(
    'daily_trend', os.path.join(os.path.dirname(__file__), '..', 'strategies', 'daily_trend.py'))
_daily_trend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_daily_trend)
MIN_VOTES, rule_signal, vote_count = _daily_trend.MIN_VOTES, _daily_trend.rule_signal, _daily_trend.vote_count
donchian_hold = _daily_trend.donchian_hold

# Über die Umgebung änderbar, z. B. SYMBOLS=BTC/USDT,ETH/USDT — wichtig gegen
# Auswahl-Verzerrung: SOL stieg 2021 auf das ~18-Fache und trägt viel vom Ergebnis.
SYMBOLS = os.environ.get('SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT').split(',')
TAKER_FEE = float(os.environ.get('TAKER_FEE', 0.0025))
START_CAPITAL = float(os.environ.get('START_CAPITAL', 100.0))
MIN_ORDER_AMOUNT = float(os.environ.get('MIN_ORDER_AMOUNT', 30.0))
SINCE = os.environ.get('SINCE', '2018-01-01')
POSITION_FRACTION = float(os.environ.get('POSITION_FRACTION') or 1 / len(SYMBOLS))
MAX_POSITIONS = int(os.environ.get('MAX_POSITIONS') or len(SYMBOLS))
WARMUP = 200

VARIANTS = ['hold', 'sma50', 'sma100', 'sma200', 'mom30', 'mom90', 'vote3', 'don20', 'don55']
DONCHIAN = {'don20': (20, 10), 'don55': (55, 20)}


def fetch_daily(exchange, symbol):
    since = int(datetime.strptime(SINCE, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='ts', keep='last')
    df['date'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)
    # Die heutige Tageskerze ist noch nicht abgeschlossen
    df = df[df['date'] < pd.Timestamp.utcnow().tz_localize(None).normalize()]
    return df.set_index('date')[['open', 'high', 'low', 'close']]


def signals(df: pd.DataFrame, variant: str) -> pd.Series:
    """True = am Tagesschluss halten wollen."""
    close = df['close']
    if variant in DONCHIAN:
        return donchian_hold(df['high'], df['low'], close, *DONCHIAN[variant])
    if variant == 'hold':
        return pd.Series(True, index=close.index)
    if variant == 'vote3':
        return vote_count(close) >= MIN_VOTES
    return rule_signal(close, variant)


def simulate(data: dict, variant: str, start, end):
    """Tagesweise Simulation von start bis end (inklusive)."""
    dates = [d for d in data[SYMBOLS[0]].index if start <= d <= end]
    sig = {s: signals(data[s], variant) for s in SYMBOLS}
    cash = START_CAPITAL
    units = {s: 0.0 for s in SYMBOLS}
    fees = 0.0
    trades = 0
    curve = []
    for k, d in enumerate(dates):
        # Ausführung zum Eröffnungskurs, Entscheidung vom Vortag
        if k > 0:
            prev = dates[k - 1]
            equity_open = cash + sum(units[s] * data[s].at[d, 'open'] for s in SYMBOLS)
            for s in SYMBOLS:
                want = bool(sig[s].get(prev, False))
                price = data[s].at[d, 'open']
                if not want and units[s] > 0:
                    value = units[s] * price
                    fee = value * TAKER_FEE
                    cash += value - fee
                    fees += fee
                    trades += 1
                    units[s] = 0.0
                elif want and units[s] == 0:
                    if sum(1 for u in units.values() if u > 0) >= MAX_POSITIONS:
                        continue
                    budget = equity_open * POSITION_FRACTION
                    # Wie live (main.py _execute_signal): zu kleine Order auf die
                    # Mindestgröße anheben, solange sie höchstens das halbe Kapital
                    # bindet — sonst bliebe ein geschrumpftes Konto für immer in Cash.
                    if budget < MIN_ORDER_AMOUNT and MIN_ORDER_AMOUNT <= equity_open * 0.5:
                        budget = MIN_ORDER_AMOUNT
                    budget = min(cash, budget)
                    if budget >= MIN_ORDER_AMOUNT:
                        fee = budget * TAKER_FEE
                        units[s] = (budget - fee) / price
                        cash -= budget
                        fees += fee
                        trades += 1
        equity = cash + sum(units[s] * data[s].at[d, 'close'] for s in SYMBOLS)
        curve.append((d, equity))
    eq = pd.Series(dict(curve))
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / START_CAPITAL - 1
    return {
        'variant': variant,
        'final': eq.iloc[-1],
        'total': total,
        'cagr': (eq.iloc[-1] / START_CAPITAL) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0,
        'max_dd': float(((eq.cummax() - eq) / eq.cummax()).max()),
        'sharpe': float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0,
        'trades': trades,
        'fees': fees,
        'curve': eq,
    }


def main():
    ex = ccxt.binance({'enableRateLimit': True})
    print(f"Lade Tageskerzen ab {SINCE} …", flush=True)
    data = {s: fetch_daily(ex, s) for s in SYMBOLS}
    common = data[SYMBOLS[0]].index
    for s in SYMBOLS[1:]:
        common = common.intersection(data[s].index)
    data = {s: df.loc[common] for s, df in data.items()}
    start, end = common[WARMUP], common[-1]
    print(f"Bewertung {start.date()} bis {end.date()} ({len(common) - WARMUP} Tage), "
          f"Gebühr {TAKER_FEE:.2%}, Start {START_CAPITAL:.0f} €, Mindestorder {MIN_ORDER_AMOUNT:.0f} €, "
          f"je Coin {POSITION_FRACTION:.0%}, höchstens {MAX_POSITIONS} Positionen\n")

    full = {v: simulate(data, v, start, end) for v in VARIANTS}
    print(f"{'Variante':9s} {'Ende €':>9s} {'gesamt':>9s} {'p.a.':>7s} {'MaxDD':>6s} {'Sharpe':>6s} {'Trades':>6s} {'Gebühr €':>9s}")
    for v, r in full.items():
        print(f"{v:9s} {r['final']:9.0f} {r['total']:+9.0%} {r['cagr']:+7.1%} {r['max_dd']:6.0%} "
              f"{r['sharpe']:6.2f} {r['trades']:6d} {r['fees']:9.1f}")

    # Jahr für Jahr: jede Variante startet am 1.1. neu mit START_CAPITAL
    print("\nRendite je Kalenderjahr (jedes Jahr frisch mit Startkapital):")
    years = sorted({d.year for d in common if d >= start})
    print(f"{'Variante':9s} " + " ".join(f"{y:>7d}" for y in years))
    for v in VARIANTS:
        row = []
        for y in years:
            ys = max(start, pd.Timestamp(f'{y}-01-01'))
            ye = min(end, pd.Timestamp(f'{y}-12-31'))
            row.append(simulate(data, v, ys, ye)['total'])
        print(f"{v:9s} " + " ".join(f"{x:+7.0%}" for x in row))

    # Dieselben vier 90-Tage-Fenster wie beim Confluence-Backtest
    print("\n90-Tage-Fenster wie im Confluence-Backtest (Signale mit vollem Vorlauf):")
    ends = [pd.Timestamp('2026-02-20'), pd.Timestamp('2026-05-20'), pd.Timestamp('2026-08-20'), end]
    print(f"{'Variante':9s} " + " ".join(f"{str(e.date()):>11s}" for e in ends))
    for v in VARIANTS:
        row = [simulate(data, v, e - pd.Timedelta(days=90), e)['total'] for e in ends]
        print(f"{v:9s} " + " ".join(f"{x:+11.1%}" for x in row))

    out = os.environ.get('CURVE_CSV')
    if out:
        pd.DataFrame({v: r['curve'] for v, r in full.items()}).to_csv(out)
        print(f"\nEquity-Kurven: {out}")


if __name__ == '__main__':
    sys.exit(main())
