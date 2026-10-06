"""
Regime Detector (Strengthened for Phase 1)

Detects the current market regime so the bot can:
- Adjust factor weights
- Choose appropriate assets
- Change risk parameters
- Decide which strategy styles are favored
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np


@dataclass
class MarketRegime:
    name: str                    # "trending", "ranging", "high_vol_event", "low_vol_chop", "event_driven"
    confidence: float            # 0.0 - 1.0
    description: str
    characteristics: Dict[str, float] = None   # e.g. {"volatility": 0.8, "trend_strength": 0.9}

    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}


class RegimeDetector:
    """
    More advanced regime detection.

    Current regimes:
    - trending (strong directional moves)
    - ranging (mean-reverting, choppy)
    - high_vol_event (news, liquidation cascades, macro events)
    - low_vol_chop (very quiet, dangerous for momentum)
    - event_driven (around major releases like CPI)

    ACHTUNG (geprüft 06.10.2026): Diese Erkennung misst den Markt NICHT richtig.
    Ihre Schwellen (0,16 / 0,85) liegen 45–60× über der berechneten Stunden-Vola,
    und vol_ratio enthält den Faktor √(12/30) = 0,63 — Ergebnis: 91–94 %
    low_vol_chop, nie trending/high_vol_event. Sie bleibt trotzdem die Grundlage
    der Handelsentscheidungen, weil die Strategie auf dieses Etikett eingestellt
    ist: Mit der korrekten Erkennung (MarketConditions unten) war der Backtest in
    allen vier 90-Tage-Fenstern schlechter. Was der Markt tatsächlich tut, zeigt
    MarketConditions — nur im Dashboard, ohne Einfluss auf Trades.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.vol_lookback = self.config.get("vol_lookback", 30)
        self.trend_lookback = self.config.get("trend_lookback", 50)

    def detect(self, symbol: str, candles: pd.DataFrame) -> MarketRegime:
        if candles is None or len(candles) < 40:
            return MarketRegime("unknown", 0.0, "Insufficient data")

        df = candles.copy()
        returns = df['close'].pct_change().dropna()

        # 1. Volatility measures
        short_vol = returns.tail(12).std() * np.sqrt(12)
        medium_vol = returns.tail(self.vol_lookback).std() * np.sqrt(self.vol_lookback)
        vol_ratio = short_vol / medium_vol if medium_vol > 0 else 1.0

        # 2. Trend strength
        ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        price = df['close'].iloc[-1]

        trend_alignment = 0.0
        if ema20 > ema50 and price > ema20:
            trend_alignment = 0.85
        elif ema20 < ema50 and price < ema20:
            trend_alignment = 0.85

        # 3. Structure
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        structure_score = 0.7 if (recent_highs.is_monotonic_increasing or recent_lows.is_monotonic_decreasing) else 0.3

        # 4. Trend strength (more granular)
        trend_strength = min(trend_alignment * structure_score * 1.2, 1.0)

        # === Regime Classification ===

        if short_vol > 0.85 or vol_ratio > 1.7:
            return MarketRegime(
                "high_vol_event",
                confidence=0.88,
                description="High volatility / event regime",
                characteristics={
                    "volatility_level": round(short_vol, 3),
                    "trend_strength": round(trend_strength, 2),
                    "vol_ratio": round(vol_ratio, 2)
                }
            )

        if short_vol < 0.16 and vol_ratio < 0.85:
            return MarketRegime(
                "low_vol_chop",
                confidence=0.78,
                description="Low volatility choppy / ranging market",
                characteristics={
                    "volatility_level": round(short_vol, 3),
                    "trend_strength": round(trend_strength, 2)
                }
            )

        if trend_strength >= 0.65:
            direction = "bullish" if ema20 > ema50 else "bearish"
            return MarketRegime(
                "trending",
                confidence=0.82,
                description=f"Strong {direction} trend",
                characteristics={
                    "volatility_level": round(short_vol, 3),
                    "trend_strength": round(trend_strength, 2),
                    "direction": direction
                }
            )

        return MarketRegime(
            "ranging",
            confidence=0.68,
            description="Range-bound / mean-reverting conditions",
            characteristics={
                "volatility_level": round(short_vol, 3),
                "trend_strength": round(trend_strength, 2)
            }
        )

    def get_preferred_style(self, regime: MarketRegime) -> List[str]:
        """Returns which strategy styles are favored in the current regime."""
        if regime.name == "trending":
            return ["momentum", "breakout", "trend_following"]
        elif regime.name == "ranging":
            return ["mean_reversion", "range_trading", "extremes"]
        elif regime.name == "high_vol_event":
            return ["cautious_momentum", "news_reaction", "reduced_size"]
        elif regime.name == "low_vol_chop":
            return ["avoid_momentum", "mean_reversion", "wait_for_setup"]
        else:
            return ["balanced"]


class MarketConditions:
    """
    Korrekte Marktmessung — NUR zur Anzeige (Dashboard), ohne Einfluss auf Trades.

    Vergleicht die Schwankung der letzten Stunde mit der des eigenen Fensters
    (ohne Wurzel-Versatz) und misst die Richtung über die Kaufman-Effizienz.
    Liefert dieselben Regime-Namen wie RegimeDetector, damit beide vergleichbar sind.
    """

    # Schwellen aus der Verteilung über 90 Tage BTC/ETH/SOL-EUR auf 5m (Stand
    # 06.10.2026, siehe docs/BACKTEST_BEFUNDE.md „Regime-Erkennung“):
    #   vol_ratio  = Std. der letzten 12 Renditen / Std. des ganzen Fensters
    #                p5 0,36 · p20 0,56 · p50 0,81 · p95 1,89
    #   efficiency = |Kursänderung| / Summe der Einzelbewegungen über 48 Kerzen
    #                p50 0,12 · p80 0,21 · p90 0,28
    # Vorher: absolute Schwellen (0,16 / 0,85) auf einer Stunden-Vola von ~0,003
    # und ein Verhältnis mit eingebautem Faktor √(12/30) = 0,63 — ergab in 30
    # Tagen 91–94 % low_vol_chop und nie trending/high_vol_event.
    HIGH_VOL_RATIO = 1.9
    LOW_VOL_RATIO = 0.56
    TREND_EFFICIENCY = 0.28
    CHOP_EFFICIENCY = 0.21

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.short_window = self.config.get("short_window", 12)        # 1 h auf 5m
        self.efficiency_window = self.config.get("efficiency_window", 48)  # 4 h auf 5m
        self.high_vol_ratio = self.config.get("high_vol_ratio", self.HIGH_VOL_RATIO)
        self.low_vol_ratio = self.config.get("low_vol_ratio", self.LOW_VOL_RATIO)
        self.trend_efficiency = self.config.get("trend_efficiency", self.TREND_EFFICIENCY)
        self.chop_efficiency = self.config.get("chop_efficiency", self.CHOP_EFFICIENCY)

    def detect(self, symbol: str, candles: pd.DataFrame) -> MarketRegime:
        if candles is None or len(candles) < max(60, self.efficiency_window + 2):
            return MarketRegime("unknown", 0.0, "Insufficient data")

        close = candles['close']
        returns = close.pct_change().dropna()

        # 1. Volatilität relativ zum eigenen Fenster — beide Seiten als Standardabweichung
        #    pro Kerze, also ohne Wurzel-Faktoren, die das Verhältnis verschieben.
        short_std = returns.tail(self.short_window).std()
        base_std = returns.std()
        vol_ratio = short_std / base_std if base_std > 0 else 1.0
        # Wie hoch die aktuelle Schwankung im Fenster liegt (0 = ruhigste, 1 = wildeste
        # Stunde). Der Aggregator erwartet volatility_level auf dieser 0–1-Skala.
        rolling_std = returns.rolling(self.short_window).std().dropna()
        vol_level = float((rolling_std <= short_std).mean()) if len(rolling_std) else 0.5

        # 2. Richtungs-Effizienz (Kaufman): 1 = gerade Linie, ~0 = reines Hin und Her
        recent = close.tail(self.efficiency_window + 1)
        path = recent.diff().abs().sum()
        efficiency = float(abs(recent.iloc[-1] - recent.iloc[0]) / path) if path > 0 else 0.0

        # 3. Trendrichtung über die Durchschnitte
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = close.iloc[-1]
        aligned_up = ema20 > ema50 and price > ema20
        aligned_down = ema20 < ema50 and price < ema20
        trend_strength = min(efficiency / 0.4, 1.0)

        chars = {
            "volatility_level": round(vol_level, 3),
            "vol_ratio": round(float(vol_ratio), 2),
            "efficiency": round(efficiency, 3),
            "trend_strength": round(trend_strength, 2),
        }

        # === Klassifikation (Reihenfolge = Vorrang) ===
        if vol_ratio >= self.high_vol_ratio:
            return MarketRegime("high_vol_event", 0.85,
                                "Schwankung deutlich über dem Normalwert", chars)

        if efficiency >= self.trend_efficiency and (aligned_up or aligned_down):
            direction = "bullish" if aligned_up else "bearish"
            chars["direction"] = direction
            return MarketRegime("trending", 0.8, f"Klarer {direction} Trend", chars)

        if vol_ratio <= self.low_vol_ratio and efficiency < self.chop_efficiency:
            return MarketRegime("low_vol_chop", 0.75,
                                "Ungewöhnlich ruhig, ohne Richtung", chars)

        return MarketRegime("ranging", 0.65, "Normale Schwankung in einer Spanne", chars)
