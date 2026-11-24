"""
Long/Short-Backtest (24.11.2026): handelt eine Strategie mehr und übersteht sie
den Abwärtsmarkt besser, wenn sie auch shorten darf? Grundlage für Swing-1.

Die Varianten sind vorab festgelegt und werden ALLE berichtet:
  1d Donchian 20/10 L/S   Panda-9-Regel, zusätzlich short unter dem 20-Tage-Tief
  4h Donchian 20/10 L/S   dasselbe auf 4h-Kerzen
  4h Donchian 55/20 L/S   langsamerer Kanal auf 4h
  4h EMA 20/50            immer drin: long über, short unter der Kreuzung (live: Swing-1)
  1h EMA 20/50            dasselbe auf 1h
  1d Halten               Vergleich

Signal am Kerzenschluss, Ausführung zum Eröffnungskurs der nächsten Kerze.
FRAC je Coin (Standard 20 %), 0,25 % Gebühr je Seite, Hebel 1.

Grenzen (bewusst einfach):
  - Kein Rebalancing: jede Coin hat ihren eigenen Topf. Ab 2021 dominiert SOL
    (Anstieg aufs Vielfache, dann −96 %) — die Spalten 365/90 Tage sind belastbarer.
  - Keine Leihkosten für Shorts (Spot-Börsen können gar nicht shorten).
  - Binance-USDT-Kerzen, live rechnet der Bot mit Kraken-EUR.

Umgebung: TAKER_FEE=0.0025  FRAC=0.2  SINCE=2021-01-01  SYMBOLS=BTC/USDT,...
"""
import importlib.util
import os

import ccxt
import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    'daily_trend', os.path.join(os.path.dirname(__file__), '..', 'strategies', 'daily_trend.py'))
_daily_trend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_daily_trend)
ema_direction = _daily_trend.ema_direction

FEE = float(os.environ.get('TAKER_FEE', 0.0025))
FRAC = float(os.environ.get('FRAC', 0.2))
SINCE = os.environ.get('SINCE', '2021-01-01')
SYMBOLS = os.environ.get('SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT').split(',')
WARMUP_DAYS = 60

exchange = ccxt.binance({'enableRateLimit': True})


def load(symbol: str, timeframe: str) -> pd.DataFrame:
    # Achtung: parse8601 braucht eine volle Zeitangabe, sonst None (= nur die letzten 1000 Kerzen)
    since = exchange.parse8601(f'{SINCE}T00:00:00Z')
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('ts')
    df.index = pd.to_datetime(df.ts, unit='ms')
    return df


def donchian_ls(df: pd.DataFrame, n_in: int, n_out: int) -> pd.Series:
    """+1 über dem n_in-Hoch bis unter das n_out-Tief, −1 spiegelbildlich, sonst 0."""
    hi_in = df.h.rolling(n_in).max().shift(1)
    lo_in = df.l.rolling(n_in).min().shift(1)
    hi_out = df.h.rolling(n_out).max().shift(1)
    lo_out = df.l.rolling(n_out).min().shift(1)
    out, p = [], 0
    for c, hi, lo, ho, lx in zip(df.c, hi_in, lo_in, hi_out, lo_out):
        if p == 1 and c < lx:
            p = 0
        elif p == -1 and c > ho:
            p = 0
        if p == 0:
            p = 1 if c > hi else -1 if c < lo else 0
        out.append(p)
    return pd.Series(out, index=df.index)


VARIANTS = {
    '1d Donchian 20/10 L/S': ('1d', lambda d: donchian_ls(d, 20, 10)),
    '4h Donchian 20/10 L/S': ('4h', lambda d: donchian_ls(d, 20, 10)),
    '4h Donchian 55/20 L/S': ('4h', lambda d: donchian_ls(d, 55, 20)),
    '4h EMA 20/50 immer drin': ('4h', lambda d: ema_direction(d.c, 20, 50)),
    '1h EMA 20/50 immer drin': ('1h', lambda d: ema_direction(d.c, 20, 50)),
    '1d Halten (nur long)': ('1d', lambda d: pd.Series(1, index=d.index)),
}


def simulate(frames, fn, start):
    total, trades = None, 0
    for df in frames:
        df = df[df.index >= start - pd.Timedelta(days=WARMUP_DAYS)]
        sig = fn(df)[df.index >= start]
        df = df[df.index >= start]
        pos = sig.shift(1).fillna(0)                      # Ausführung nächste Kerze
        ret = df.o.pct_change().shift(-1).fillna(0)       # Eröffnung → Eröffnung
        change = pos.diff().abs().fillna(abs(pos.iloc[0]))
        trades += int((change > 0).sum())
        equity = (1 + pos * ret - change * FEE).cumprod()
        total = equity * FRAC if total is None else total + equity * FRAC
    total = total + (1 - FRAC * len(frames))
    days = max((total.index[-1] - total.index[0]).days, 1)
    return total.iloc[-1] - 1, (total / total.cummax() - 1).min(), trades / days * 30


def main():
    data = {tf: [load(s, tf) for s in SYMBOLS] for tf in ('1d', '4h', '1h')}
    first = max(d.index[0] for frames in data.values() for d in frames) + pd.Timedelta(days=WARMUP_DAYS)
    now = pd.Timestamp.now()
    periods = [(f'ab {first:%m/%Y}', first),
               ('letzte 365 T', now - pd.Timedelta(days=365)),
               ('letzte 90 T', now - pd.Timedelta(days=90))]
    print(f"Gebühr {FEE:.2%} je Seite · {FRAC:.0%} je Coin · {', '.join(SYMBOLS)}")
    for name, (tf, fn) in VARIANTS.items():
        parts = [f"{p}: {r * 100:+.0f} % / Rückgang {dd * 100:.0f} % / {n:.1f} Trades/Monat"
                 for p, start in periods for r, dd, n in [simulate(data[tf], fn, start)]]
        print(f"{name:26} | " + ' | '.join(parts), flush=True)


if __name__ == '__main__':
    main()
