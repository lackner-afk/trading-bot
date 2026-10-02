"""
Order-Flow-Faktor auf Basis echter Aggressor-Daten (Taker-Buy/-Sell-Volumen).

Wichtig: Die bisherige Spalte `volume_delta` in den Feeds ist KEIN Order Flow —
sie gibt nur dem Kerzenvolumen das Vorzeichen der Kerzenfarbe. Dieser Faktor
braucht die Spalte `taker_buy_volume` (Volumen, das von aggressiven Käufern
per Market-Order ausgeführt wurde). Binance-Klines liefern sie direkt mit,
alternativ lässt sie sich aus Einzel-Trades (Feld `side`) aggregieren.

Zwei Setups, jeweils spiegelbildlich für Short:

1. trend — aggressive Käufer dominieren deutlich (Imbalance-Z-Score hoch),
   der Preis bestätigt die Bewegung und liegt über der Trend-EMA.
   Idee: Echter Kaufdruck trägt die Bewegung weiter.

2. absorption — aggressive Verkäufer dominieren deutlich, der Preis fällt
   aber kaum (gemessen in ATR). Idee: Passive Limit-Käufer schlucken den
   Verkaufsdruck; sind die Verkäufer erschöpft, dreht der Markt.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import Factor, FactorResult


DEFAULT_PARAMS: Dict = {
    "lookback": 12,          # Kerzen für die Imbalance (rollierende Summe)
    "z_window": 200,         # Kerzen für die Normierung der Imbalance
    "z_threshold": 2.0,      # ab diesem Z-Score gilt der Fluss als "deutlich"
    "atr_period": 14,
    "trend_ema": 100,
    "absorption_max_move_atr": 0.5,  # Preis darf sich max. so weit gegen den Fluss bewegen
    "modes": ("trend", "absorption"),
}


def compute_order_flow(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Berechnet Order-Flow-Kennzahlen und Signale für jede Kerze (vektorisiert).

    Wird vom Live-Faktor (letzte Zeile) und vom Backtest (alle Zeilen)
    gleichermaßen genutzt, damit beide exakt dieselbe Logik fahren.
    Jede Zeile nutzt nur Daten bis einschließlich dieser Kerze.

    Erwartet Spalten: high, low, close, volume, taker_buy_volume.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    out = pd.DataFrame(index=df.index)

    volume = df["volume"].astype(float)
    buy = df["taker_buy_volume"].astype(float).clip(lower=0.0)
    buy = np.minimum(buy, volume)
    delta = 2.0 * buy - volume                     # Käufer- minus Verkäufer-Volumen
    out["delta"] = delta
    out["cvd"] = delta.cumsum()

    n = p["lookback"]
    vol_sum = volume.rolling(n).sum()
    imbalance = delta.rolling(n).sum() / vol_sum.replace(0.0, np.nan)   # -1 .. +1
    out["imbalance"] = imbalance

    # Jeder Markt hat einen strukturellen Taker-Bias — erst die Abweichung
    # davon ist Information. Daher Z-Score statt Rohwert.
    zw = p["z_window"]
    imb_mean = imbalance.rolling(zw, min_periods=zw // 2).mean()
    imb_std = imbalance.rolling(zw, min_periods=zw // 2).std()
    out["imbalance_z"] = (imbalance - imb_mean) / imb_std.replace(0.0, np.nan)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift()
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_pct = true_range.rolling(p["atr_period"]).mean() / close
    out["atr_pct"] = atr_pct

    # Preisbewegung über das Imbalance-Fenster, in ATR-Einheiten
    move_atr = (close / close.shift(n) - 1.0) / atr_pct.replace(0.0, np.nan)
    out["move_atr"] = move_atr

    trend_ema = close.ewm(span=p["trend_ema"], adjust=False).mean()
    uptrend = close > trend_ema
    out["uptrend"] = uptrend

    z = out["imbalance_z"]
    zt = p["z_threshold"]
    max_move = p["absorption_max_move_atr"]
    modes = set(p["modes"])

    trend_long = (z >= zt) & (move_atr > 0) & uptrend
    trend_short = (z <= -zt) & (move_atr < 0) & ~uptrend
    absorb_long = (z <= -zt) & (move_atr >= -max_move)
    absorb_short = (z >= zt) & (move_atr <= max_move)

    false = pd.Series(False, index=df.index)
    out["trend_long"] = trend_long if "trend" in modes else false
    out["trend_short"] = trend_short if "trend" in modes else false
    out["absorb_long"] = absorb_long if "absorption" in modes else false
    out["absorb_short"] = absorb_short if "absorption" in modes else false
    out["signal_long"] = (out["trend_long"] | out["absorb_long"]).fillna(False)
    out["signal_short"] = (out["trend_short"] | out["absorb_short"]).fillna(False)
    return out


class OrderFlowFactor(Factor):
    """
    Richtungsfaktor aus echtem Order Flow (Taker-Imbalance).

    Liefert None, wenn die Kerzen keine Spalte `taker_buy_volume` haben —
    dann gibt es schlicht keinen Order Flow, und der Faktor soll nicht raten.
    """

    name = "order_flow"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.params = {**DEFAULT_PARAMS, **{k: v for k, v in self.config.items()
                                            if k in DEFAULT_PARAMS}}

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or "taker_buy_volume" not in candles.columns:
            return None
        min_len = max(self.params["z_window"] // 2, self.params["trend_ema"] // 2) + self.params["lookback"]
        if len(candles) < min_len:
            return None

        flow = compute_order_flow(candles, self.params)
        last = flow.iloc[-1]
        z = last["imbalance_z"]
        if pd.isna(z) or pd.isna(last["move_atr"]):
            return None

        meta = {
            "imbalance": float(last["imbalance"]),
            "imbalance_z": float(z),
            "move_atr": float(last["move_atr"]),
            "atr_pct": float(last["atr_pct"]),
        }
        # Score wächst mit der Stärke des Flusses über der Schwelle (0.70 - 1.0)
        strength = min((abs(z) - self.params["z_threshold"]) / 2.0, 1.0)
        score = 0.70 + 0.30 * max(strength, 0.0)

        if last["trend_long"]:
            direction, setup = "long", "Kaufdruck mit Preisbestätigung"
        elif last["absorb_long"]:
            direction, setup = "long", "Verkaufsdruck absorbiert"
        elif last["trend_short"]:
            direction, setup = "short", "Verkaufsdruck mit Preisbestätigung"
        elif last["absorb_short"]:
            direction, setup = "short", "Kaufdruck absorbiert"
        else:
            return FactorResult(
                name=self.name, score=0.5, confidence=0.5, direction=None,
                reason=f"Order Flow neutral (Z={z:+.2f})", metadata=meta,
            )

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.8,
            direction=direction,
            reason=f"{setup} (Imbalance-Z={z:+.2f}, Bewegung {last['move_atr']:+.2f} ATR)",
            metadata=meta,
        )
