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
