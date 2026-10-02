#!/usr/bin/env python3
"""
Backtest der Order-Flow-Strategie über mehrere unabhängige Marktphasen.

Datenquelle: Binance-Klines. Sie enthalten das Taker-Buy-Volumen (Spalte 9),
also echten Aggressor-Order-Flow je Kerze — nicht nur eine Schätzung aus der
Kerzenfarbe. CCXT verwirft diese Spalte, daher direkter REST-Aufruf.

Bewertet wird bewusst NICHT der beste Lauf eines Fensters, sondern ob eine
Konfiguration in ALLEN Fenstern positiv ist (Lehre aus docs/BACKTEST_BEFUNDE.md).

Ohne Lookahead: Signal auf Schlusskurs von Kerze i, Einstieg zum Open von i+1.
Intrabar wird der Stop vor dem Ziel geprüft (konservativ).

Steuerung über die Umgebung (Defaults = reales Fusion-Konto):
    TIMEFRAME=1h              # 5m, 15m, 1h, 4h
    DAYS=90                   # Länge eines Fensters
    WINDOWS=3                 # Anzahl aufeinanderfolgender Fenster rückwärts
    BACKTEST_END=2026-10-01   # Ende des jüngsten Fensters (leer = jetzt)
    SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
    TAKER_FEE=0.0025          # Bitpanda Fusion Stufe 1
    START_CAPITAL=100
    MIN_ORDER_AMOUNT=25
    POSITION_PCT=0.30         # Anteil der Equity je Position (Spot, kein Hebel)
    MAX_CONCURRENT=2
    LONG_ONLY=1               # Spot-Börse: keine Shorts
    MAX_HOLD=48               # Zeit-Stop in Kerzen
    MIN_TP_PCT=0.010          # TP-Floor über den Round-Trip-Kosten
    LOOKBACKS=6,12,24         # Grid: Imbalance-Fenster in Kerzen
    Z_THRESHOLDS=1.5,2.0,2.5  # Grid: Z-Score-Schwelle
    MODES=trend,absorption    # Grid: Setups (jeweils einzeln getestet)
    TPSL=3:2,4:2,6:3          # Grid: TP/SL als ATR-Vielfache
    BINANCE_URL=https://api.binance.com
    DATA_DIR=/tmp/orderflow_data   # Kerzen-Cache
    OUTPUT=/tmp/orderflow_sweep.json
    NULL_RUNS=5               # Null-Test: Läufe mit zeitlich verschobenem Order Flow
"""
import itertools
import json
import os
import pickle
import sys
import time as _time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.factors.order_flow import DEFAULT_PARAMS, compute_order_flow


def _env_list(name: str, default: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


TIMEFRAME = os.environ.get("TIMEFRAME", "1h")
DAYS = int(os.environ.get("DAYS", 90))
WINDOWS = int(os.environ.get("WINDOWS", 3))
BACKTEST_END = os.environ.get("BACKTEST_END")
SYMBOLS = [s.replace("/", "") for s in _env_list("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")]
TAKER_FEE = float(os.environ.get("TAKER_FEE", 0.0025))
START_CAPITAL = float(os.environ.get("START_CAPITAL", 100.0))
MIN_ORDER_AMOUNT = float(os.environ.get("MIN_ORDER_AMOUNT", 25.0))
POSITION_PCT = float(os.environ.get("POSITION_PCT", 0.30))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 2))
LONG_ONLY = os.environ.get("LONG_ONLY", "1") == "1"
MAX_HOLD = int(os.environ.get("MAX_HOLD", 48))
MIN_TP_PCT = float(os.environ.get("MIN_TP_PCT", 0.010))
MAX_DAILY_DD = 0.10
LOOKBACKS = [int(x) for x in _env_list("LOOKBACKS", "6,12,24")]
Z_THRESHOLDS = [float(x) for x in _env_list("Z_THRESHOLDS", "1.5,2.0,2.5")]
MODES = _env_list("MODES", "trend,absorption")
TPSL_PROFILES = [tuple(float(x) for x in pair.split(":")) for pair in _env_list("TPSL", "3:2,4:2,6:3")]
BINANCE_URL = os.environ.get("BINANCE_URL", "https://api.binance.com").rstrip("/")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/orderflow_data"))
OUTPUT = os.environ.get("OUTPUT", "/tmp/orderflow_sweep.json")
NULL_RUNS = int(os.environ.get("NULL_RUNS", 5))

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
# Vorlauf, damit Z-Score und Trend-EMA ab dem ersten Fenster-Bar eingeschwungen sind
WARMUP_BARS = DEFAULT_PARAMS["z_window"] + DEFAULT_PARAMS["trend_ema"] + max(LOOKBACKS)


def _end_ms() -> int:
    if not BACKTEST_END:
        return int(_time.time() * 1000)
    dt = datetime.strptime(BACKTEST_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance-Klines inkl. Taker-Buy-Volumen, mit Datei-Cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{symbol}_{TIMEFRAME}_{start_ms}_{end_ms}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    rows = []
    since = start_ms
    while since < end_ms:
        for attempt in range(4):
            try:
                r = requests.get(f"{BINANCE_URL}/api/v3/klines", params={
                    "symbol": symbol, "interval": TIMEFRAME,
                    "startTime": since, "endTime": end_ms, "limit": 1000,
                }, timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                print(f"  {symbol}: Abruf fehlgeschlagen ({e}), neuer Versuch...", file=sys.stderr)
                _time.sleep(2 ** (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + TF_MS[TIMEFRAME]
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote", "ignore",
    ])
    df = df[["ts", "open", "high", "low", "close", "volume", "taker_buy_volume"]].astype(float)
    df["ts"] = df["ts"].astype("int64")
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    # Nur abgeschlossene Kerzen: die laufende Kerze hätte unvollständiges Volumen
    df = df[df["ts"] + TF_MS[TIMEFRAME] <= end_ms].reset_index(drop=True)
    df.to_pickle(cache)
    return df


def load_window(start_ms: int, end_ms: int) -> Dict[str, pd.DataFrame]:
    """Kerzen aller Symbole inkl. Vorlauf, auf gemeinsame Zeitstempel ausgerichtet."""
    fetch_start = start_ms - WARMUP_BARS * TF_MS[TIMEFRAME]
    raw = {s: fetch_klines(s, fetch_start, end_ms).set_index("ts") for s in SYMBOLS}
    common = sorted(set.intersection(*(set(df.index) for df in raw.values())))
    return {s: df.loc[common].reset_index() for s, df in raw.items()}


def simulate(candles: Dict[str, pd.DataFrame], flows: Dict[str, pd.DataFrame],
             start_idx: int, tp_mult: float, sl_mult: float) -> Dict:
    """Portfolio-Simulation (Spot, kein Hebel) ab start_idx."""
    n = len(next(iter(candles.values())))
    arr = {s: {c: candles[s][c].to_numpy() for c in ("ts", "open", "high", "low", "close")}
           for s in SYMBOLS}
    sig_long = {s: flows[s]["signal_long"].to_numpy(dtype=bool) for s in SYMBOLS}
    sig_short = {s: flows[s]["signal_short"].to_numpy(dtype=bool) for s in SYMBOLS}
    atr = {s: flows[s]["atr_pct"].to_numpy() for s in SYMBOLS}

    cash = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    equity_curve: List[float] = []
    day_start: Dict[str, float] = {}
    paused_day = None

    def close_position(sym: str, price: float, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        gross = pos["qty"] * (price - pos["entry"]) * pos["dir"]
        fee = pos["qty"] * price * TAKER_FEE
        cash += pos["notional"] + gross - fee
        pnl = gross - fee - pos["entry_fee"]
        trades.append({"symbol": sym, "dir": pos["dir"], "pnl": pnl,
                       "ret": pnl / pos["notional"], "exit": reason,
                       "bars": pos["bars"]})

    for i in range(start_idx, n):
        day = datetime.fromtimestamp(arr[SYMBOLS[0]]["ts"][i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        # 1) Einstiege zum Open, auf Basis des Signals der Vorkerze
        equity_open = cash + sum(
            p["notional"] + p["qty"] * (arr[s]["open"][i] - p["entry"]) * p["dir"]
            for s, p in positions.items())
        day_start.setdefault(day, equity_open)
        if paused_day != day:
            for sym in SYMBOLS:
                if sym in positions or len(positions) >= MAX_CONCURRENT:
                    continue
                j = i - 1
                if sig_long[sym][j]:
                    d = 1
                elif sig_short[sym][j] and not LONG_ONLY:
                    d = -1
                else:
                    continue
                a = atr[sym][j]
                if not np.isfinite(a) or a <= 0:
                    continue
                notional = min(equity_open * POSITION_PCT, cash)
                entry_fee = notional * TAKER_FEE
                if notional < MIN_ORDER_AMOUNT or notional + entry_fee > cash:
                    continue
                entry = arr[sym]["open"][i]
                tp_pct = max(tp_mult * a, MIN_TP_PCT)
                sl_pct = sl_mult * a
                cash -= notional + entry_fee
                positions[sym] = {
                    "dir": d, "entry": entry, "qty": notional / entry,
                    "notional": notional, "entry_fee": entry_fee, "bars": 0,
                    "tp": entry * (1 + d * tp_pct), "sl": entry * (1 - d * sl_pct),
                }

        # 2) Ausstiege innerhalb der Kerze (Stop vor Ziel), dann Zeit-Stop
        for sym in list(positions):
            pos = positions[sym]
            hi, lo, cl = arr[sym]["high"][i], arr[sym]["low"][i], arr[sym]["close"][i]
            if pos["dir"] == 1:
                if lo <= pos["sl"]:
                    close_position(sym, pos["sl"], "SL")
                    continue
                if hi >= pos["tp"]:
                    close_position(sym, pos["tp"], "TP")
                    continue
            else:
                if hi >= pos["sl"]:
                    close_position(sym, pos["sl"], "SL")
                    continue
                if lo <= pos["tp"]:
                    close_position(sym, pos["tp"], "TP")
                    continue
            pos["bars"] += 1
            if pos["bars"] >= MAX_HOLD:
                close_position(sym, cl, "TIME")

        equity = cash + sum(
            p["notional"] + p["qty"] * (arr[s]["close"][i] - p["entry"]) * p["dir"]
            for s, p in positions.items())
        equity_curve.append(equity)
        if (day_start[day] - equity) / day_start[day] > MAX_DAILY_DD:
            paused_day = day

    for sym in list(positions):
        close_position(sym, arr[sym]["close"][n - 1], "EOD")

    eq = np.array(equity_curve) if equity_curve else np.array([START_CAPITAL])
    peak = np.maximum.accumulate(eq)
    wins = [t for t in trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    return {
        "return_pct": (cash - START_CAPITAL) / START_CAPITAL,
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "avg_trade_ret": float(np.mean([t["ret"] for t in trades])) if trades else 0.0,
        "max_dd": float(((peak - eq) / peak).max()),
        "exits": dict(Counter(t["exit"] for t in trades)),
    }


def rotate_flow(candles: Dict[str, pd.DataFrame], rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    """
    Null-Modell: Taker-Anteil je Symbol zirkulär um einen Zufallsversatz
    verschieben. Autokorrelation und Verteilung des Flusses bleiben erhalten,
    nur der zeitliche Bezug zum Preis ist zerstört.
    """
    out = {}
    for sym, df in candles.items():
        df = df.copy()
        ratio = (df["taker_buy_volume"] / df["volume"].replace(0.0, np.nan)).fillna(0.5).to_numpy()
        shift = int(rng.integers(len(ratio) // 4, 3 * len(ratio) // 4))
        df["taker_buy_volume"] = np.roll(ratio, shift) * df["volume"].to_numpy()
        out[sym] = df
    return out


def run_grid(window_data: List[Tuple[Dict[str, pd.DataFrame], int]], configs: List[Tuple]) -> List[Dict]:
    """Alle Konfigurationen über alle Fenster simulieren."""
    results: Dict[Tuple, List[Dict]] = {c: [] for c in configs}
    for candles, start_idx in window_data:
        flow_cache = {}
        for lb, zt, mode, (tp, sl) in configs:
            key = (lb, zt, mode)
            if key not in flow_cache:
                params = {"lookback": lb, "z_threshold": zt, "modes": (mode,)}
                flow_cache[key] = {s: compute_order_flow(df, params) for s, df in candles.items()}
            results[(lb, zt, mode, (tp, sl))].append(simulate(candles, flow_cache[key], start_idx, tp, sl))

    rows = []
    for (lb, zt, mode, (tp, sl)), res in results.items():
        rets = [r["return_pct"] for r in res]
        rows.append({
            "lookback": lb, "z": zt, "mode": mode, "tp": tp, "sl": sl,
            "returns": rets, "sum": sum(rets), "worst": min(rets),
            "positive_windows": sum(r > 0 for r in rets),
            "trades": [r["n_trades"] for r in res],
            "win_rates": [r["win_rate"] for r in res],
            "avg_trade_ret": [r["avg_trade_ret"] for r in res],
            "profit_factors": [r["profit_factor"] for r in res],
            "max_dds": [r["max_dd"] for r in res],
            "exits": [r["exits"] for r in res],
        })
    # Sortiert nach dem schlechtesten Fenster — Robustheit vor Spitzenrendite
    rows.sort(key=lambda r: (-r["positive_windows"], -r["worst"]))
    return rows


def main() -> None:
    end = _end_ms()
    span = DAYS * 86_400_000
    windows = [(end - (k + 1) * span, end - k * span) for k in range(WINDOWS)]

    def fmt(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"Order-Flow-Backtest | {TIMEFRAME} | {WINDOWS}x{DAYS} Tage | Fee {TAKER_FEE:.2%} | "
          f"{'Long-only' if LONG_ONLY else 'Long/Short'} | Kapital {START_CAPITAL:.0f}", flush=True)

    configs = list(itertools.product(LOOKBACKS, Z_THRESHOLDS, MODES, TPSL_PROFILES))
    window_data = []
    buy_hold = []
    for w_start, w_end in windows:
        print(f"\nFenster {fmt(w_start)} → {fmt(w_end)}: lade Kerzen...", flush=True)
        candles = load_window(w_start, w_end)
        ts = candles[SYMBOLS[0]]["ts"].to_numpy()
        # start_idx >= 1 nötig, weil der Einstieg das Signal der Vorkerze liest
        start_idx = max(int(np.searchsorted(ts, w_start)), 1)
        bh = np.mean([df["close"].iloc[-1] / df["open"].iloc[start_idx] - 1 for df in candles.values()])
        buy_hold.append(float(bh))
        print(f"  {len(ts) - start_idx} Kerzen im Fenster (+{start_idx} Vorlauf) | Buy&Hold {bh:+.1%}", flush=True)
        window_data.append((candles, start_idx))

    rows = run_grid(window_data, configs)

    labels = [f"{fmt(s)[2:]}" for s, _ in windows]
    print("\n" + "=" * 110)
    print("Buy&Hold je Fenster: " + "  ".join(f"{l}: {b:+.1%}" for l, b in zip(labels, buy_hold)))
    print("=" * 110)
    head = " ".join(f"{l:>9}" for l in labels)
    print(f"{'LB':>3} {'Z':>4} {'Modus':>10} {'TP:SL':>7} {head} {'Summe':>8} {'Trades':>12} {'WinRate':>16} {'Ø/Trade':>8}")
    for r in rows[:25]:
        rets = " ".join(f"{x:>+9.1%}" for x in r["returns"])
        trades = "/".join(str(t) for t in r["trades"])
        wr = "/".join(f"{w:.0%}" for w in r["win_rates"])
        avg = np.mean(r["avg_trade_ret"])
        print(f"{r['lookback']:>3} {r['z']:>4} {r['mode']:>10} {r['tp']:>3g}:{r['sl']:<3g} {rets} "
              f"{r['sum']:>+8.1%} {trades:>12} {wr:>16} {avg:>+8.2%}")

    robust = sum(r["positive_windows"] == WINDOWS for r in rows)
    best_worst = rows[0]["worst"]
    print(f"\nIn ALLEN {WINDOWS} Fenstern positiv: {robust} von {len(rows)} Konfigurationen "
          f"| bestes schlechtestes Fenster: {best_worst:+.1%}")

    # Null-Test: dasselbe Grid mit zeitlich verschobenem Order Flow.
    # Verglichen wird, was man durch Auswahl der besten Konfiguration erreicht —
    # schafft der Zufall das Gleiche, ist das echte Ergebnis Selektions-Glück.
    null_stats = []
    if NULL_RUNS > 0:
        print(f"\nNull-Test: {NULL_RUNS} Läufe mit verschobenem Order Flow...", flush=True)
        rng = np.random.default_rng(42)
        for k in range(NULL_RUNS):
            shuffled = [(rotate_flow(c, rng), si) for c, si in window_data]
            null_rows = run_grid(shuffled, configs)
            null_stats.append({
                "robust": sum(r["positive_windows"] == WINDOWS for r in null_rows),
                "best_worst": null_rows[0]["worst"],
            })
            print(f"  Lauf {k + 1}: {null_stats[-1]['robust']} robust | "
                  f"bestes schlechtestes Fenster {null_stats[-1]['best_worst']:+.1%}", flush=True)
        null_robust = [s["robust"] for s in null_stats]
        null_best = [s["best_worst"] for s in null_stats]
        print(f"\nEcht:   {robust} robust | bestes schlechtestes Fenster {best_worst:+.1%}")
        print(f"Zufall: Ø {np.mean(null_robust):.1f} robust (max {max(null_robust)}) | "
              f"bestes schlechtestes Fenster max {max(null_best):+.1%}")
        if robust > max(null_robust) and best_worst > max(null_best) and best_worst > 0:
            print("→ Der Order Flow schlägt den Zufall. Nächster Schritt: Paper-Test, kein Echtgeld.")
        else:
            print("→ Kein Beleg für eine Kante: Der Zufall erreicht Ähnliches. Kein Echtgeld auf dieser Basis.")

    with open(OUTPUT, "w") as f:
        json.dump({"buy_hold": buy_hold, "windows": [(fmt(s), fmt(e)) for s, e in windows],
                   "null_test": null_stats, "results": rows}, f, indent=1, default=str)
    print(f"Details: {OUTPUT}")


if __name__ == "__main__":
    main()
