"""
Sentiment Factor (Phase 2)

Provides sentiment-based conviction to the multi-factor system.

Current implementation (starting point):
- Integrates Alternative.me Fear & Greed Index (free, no API key needed)
- Can be extended later with LunarCrush, Santiment, on-chain sentiment, etc.

The factor acts differently depending on the regime:
- In "ranging" or "low_vol_chop": Extreme fear/greed are strong contrarian signals
- In "trending": Sentiment should mostly confirm the trend (not fight it)
- In "high_vol_event": Sentiment is de-weighted (too noisy)
"""

import requests
from typing import Optional, Dict
import pandas as pd
from datetime import datetime, timedelta

from .base import Factor, FactorResult


class SentimentFactor(Factor):
    """
    Sentiment scoring factor.

    Uses Fear & Greed Index as primary source for now.
    Score is normalized to 0-1 and interpreted depending on regime.
    """

    name = "sentiment"

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # How often we refresh the Fear & Greed data (in seconds)
        self.cache_seconds = self.config.get("cache_seconds", 3600)  # 1 hour default
        self._last_fetch = None
        self._cached_value = None   # 0-100 (0 = Extreme Fear, 100 = Extreme Greed)

        # Weights for different regimes (can be tuned)
        self.regime_weights = self.config.get("regime_weights", {
            "ranging": 0.9,
            "low_vol_chop": 1.0,
            "trending": 0.6,
            "high_vol_event": 0.3,
            "event_driven": 0.4,
            "unknown": 0.7
        })

    def _fetch_fear_and_greed(self) -> Optional[float]:
        """Fetch current Fear & Greed Index from alternative.me (free API)."""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                value = float(data["data"][0]["value"])
                self._cached_value = value
                self._last_fetch = datetime.now()
                return value
        except Exception as e:
            print(f"[SentimentFactor] Failed to fetch Fear & Greed: {e}")

        return None

    def _get_fear_and_greed(self) -> Optional[float]:
        """Get cached or fresh Fear & Greed value."""
        now = datetime.now()

        if (self._cached_value is None or
            self._last_fetch is None or
            (now - self._last_fetch).total_seconds() > self.cache_seconds):

            return self._fetch_fear_and_greed()

        return self._cached_value

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        regime = kwargs.get("regime", "unknown")
        fng_value = self._get_fear_and_greed()

        if fng_value is None:
            return None

        # Normalize Fear & Greed to 0-1 (0 = Extreme Fear, 1 = Extreme Greed)
        normalized = fng_value / 100.0

        # Get regime-specific weight
        regime_weight = self.regime_weights.get(regime, 0.7)

        # Interpretation logic
        if fng_value < 25:           # Extreme Fear
            if regime in ["ranging", "low_vol_chop"]:
                direction = "long"
                score = 0.9 * regime_weight
                reason = f"Extreme Fear ({fng_value}) → strong contrarian long signal"
            else:
                direction = "long"
                score = 0.65 * regime_weight
                reason = f"Extreme Fear ({fng_value}) → cautious long bias"

        elif fng_value > 75:         # Extreme Greed
            if regime in ["ranging", "low_vol_chop"]:
                direction = "short"
                score = 0.9 * regime_weight
                reason = f"Extreme Greed ({fng_value}) → strong contrarian short signal"
            else:
                direction = None
                score = 0.4
                reason = f"Extreme Greed ({fng_value}) → reduce conviction"

        elif fng_value < 45:         # Fear
            direction = "long"
            score = 0.65 * regime_weight
            reason = f"Fear zone ({fng_value}) → mild bullish sentiment"

        elif fng_value > 55:         # Greed
            direction = "short" if regime in ["ranging", "low_vol_chop"] else None
            score = 0.55 * regime_weight
            reason = f"Greed zone ({fng_value})"

        else:                        # Neutral
            direction = None
            score = 0.5
            reason = f"Neutral sentiment ({fng_value})"

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.75,   # Sentiment is useful but noisy
            direction=direction,
            reason=reason,
            metadata={
                "fear_and_greed": fng_value,
                "regime": regime,
                "regime_weight": regime_weight
            }
        )
