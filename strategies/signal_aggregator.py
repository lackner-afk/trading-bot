"""
Signal Aggregator - The new central brain of the trading bot.

Combines multiple FactorResults into a single high-quality TradeSignal
using a confluence scoring system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .factors.base import FactorResult


@dataclass
class TradeSignal:
    """Final aggregated trading signal."""
    symbol: str
    direction: str                    # "long" or "short"
    confidence: float                 # 0.0 - 1.0 overall conviction
    confluence_score: float           # 0.0 - 10.0 (or higher) total score
    suggested_leverage: float
    take_profit: float
    stop_loss: float
    reason: str
    factor_breakdown: Dict[str, FactorResult] = field(default_factory=dict)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalAggregator:
    """
    Combines multiple factors into one trading decision using a confluence system.

    This is the central component that will eventually replace the old
    MomentumStrategy / CryptoScalper logic.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_confluence = self.config.get("min_confluence_score", 5.8)
        self.base_leverage = self.config.get("base_leverage", 8)
        self.min_technical_factors = self.config.get("min_technical_factors", 2)

        # Base weights (will be adjusted by regime)
        self.base_weights = self.config.get("factor_weights", {
            "technical": 0.58,
            "sentiment": 0.25,
            "macro_news": 0.17
        })
        self.weights = self.base_weights.copy()

    def aggregate(self,
                  symbol: str,
                  current_price: float,
                  factor_results: List[FactorResult],
                  regime: Optional[str] = None,
                  regime_characteristics: Dict = None,
                  macro_risk_multiplier: float = 1.0) -> Optional[TradeSignal]:

        if not factor_results:
            return None

        # Apply regime-adjusted weights if regime information is provided
        if regime:
            adjusted_weights = self.get_regime_adjusted_weights(regime, regime_characteristics)
            self.weights = adjusted_weights

        # Categorize factors
        tech_results = [f for f in factor_results if any(x in f.name for x in ["trend", "momentum", "breakout", "volume", "volatility", "technical"])]
        sentiment_results = [f for f in factor_results if "sentiment" in f.name or f.name == "sentiment"]
        macro_results = [f for f in factor_results if any(x in f.name for x in ["macro", "news", "cpi", "event", "macro_news_filter"])]

        tech_score = self._average_score(tech_results) * self.weights["technical"]
        sent_score = self._average_score(sentiment_results) * self.weights["sentiment"]
        macro_score = self._average_score(macro_results) * self.weights["macro_news"]

        total_score = tech_score + sent_score + macro_score

        # Direction voting (weighted)
        long_score = sum(f.score for f in factor_results if f.direction == "long")
        short_score = sum(f.score for f in factor_results if f.direction == "short")

        if long_score <= short_score or total_score < self.min_confluence:
            if short_score > long_score and total_score >= self.min_confluence:
                direction = "short"
            else:
                return None
        else:
            direction = "long"

        # Require at least X technical factors to have decent conviction
        if len(tech_results) < self.min_technical_factors:
            return None

        confidence = min(total_score / 9.5, 1.0)

        # Dynamic leverage based on confluence + regime + macro events
        leverage = self._calculate_leverage(confidence, regime) * macro_risk_multiplier

        # Simple but reasonable TP/SL (will be improved with ATR later)
        tp_pct = 0.016 + (confidence * 0.012)
        sl_pct = 0.008 + (confidence * 0.005)

        if direction == "long":
            take_profit = current_price * (1 + tp_pct)
            stop_loss = current_price * (1 - sl_pct)
        else:
            take_profit = current_price * (1 - tp_pct)
            stop_loss = current_price * (1 + sl_pct)

        reason = (
            f"Confluence {total_score:.1f}/10 | "
            f"Tech {tech_score:.1f} | Sent {sent_score:.1f} | Macro {macro_score:.1f}"
        )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            confluence_score=total_score,
            suggested_leverage=leverage,
            take_profit=take_profit,
            stop_loss=stop_loss,
            reason=reason,
            factor_breakdown={f.name: f for f in factor_results}
        )

    def _average_score(self, results: List[FactorResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def get_regime_adjusted_weights(self, regime_name: str, characteristics: Dict = None) -> Dict[str, float]:
        """
        Returns factor weights adjusted for the current regime.
        This is the core of Phase 4 adaptive weighting.
        """
        weights = self.base_weights.copy()
        chars = characteristics or {}

        vol_level = chars.get("volatility_level", 0.4)
        trend_strength = chars.get("trend_strength", 0.5)

        if regime_name == "trending":
            # Favor technical trend/momentum heavily
            weights["technical"] = 0.72
            weights["sentiment"] = 0.18
            weights["macro_news"] = 0.10

        elif regime_name == "ranging":
            # Sentiment and mean-reversion (via technical) become more important
            weights["technical"] = 0.48
            weights["sentiment"] = 0.35
            weights["macro_news"] = 0.17

        elif regime_name == "high_vol_event":
            # Reduce overall risk, rely more on macro and strong technical confirmation
            weights["technical"] = 0.50
            weights["sentiment"] = 0.15
            weights["macro_news"] = 0.35

        elif regime_name == "low_vol_chop":
            # Be very selective — higher bar for technical, use sentiment for extremes
            weights["technical"] = 0.45
            weights["sentiment"] = 0.40
            weights["macro_news"] = 0.15

        elif regime_name == "event_driven":
            weights["technical"] = 0.40
            weights["sentiment"] = 0.20
            weights["macro_news"] = 0.40

        # Slight volatility adjustment
        if vol_level > 0.7:
            weights["technical"] *= 0.9
            weights["macro_news"] *= 1.15
        elif vol_level < 0.2:
            weights["technical"] *= 0.85
            weights["sentiment"] *= 1.2

        # Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

    def set_weights_for_regime(self, regime_name: str, characteristics: Dict = None):
        """Apply regime-adjusted weights to the aggregator."""
        self.weights = self.get_regime_adjusted_weights(regime_name, characteristics)

    def _calculate_leverage(self, confidence: float, regime: Optional[str], macro_risk_multiplier: float = 1.0) -> float:
        base = self.base_leverage * (0.6 + confidence * 0.7)

        if regime == "high_vol_event":
            base *= 0.55
        elif regime == "low_vol_chop":
            base *= 0.45
        elif regime == "trending":
            base *= 1.2
        elif regime == "event_driven":
            base *= 0.65

        # Apply macro event risk reduction
        base *= macro_risk_multiplier

        return max(2.0, min(base, 18.0))
