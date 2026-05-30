"""
Asset Selector / Dynamic Universe Manager

Responsible for choosing which assets the bot should focus on,
based on current regime, liquidity, volatility, and narrative strength.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class AssetCandidate:
    symbol: str
    score: float                    # Overall attractiveness 0-1
    liquidity_score: float
    volatility_score: float
    regime_fit: float
    reason: str


class AssetSelector:
    """
    Selects the best assets to trade right now.

    This is a key component for making the bot more adaptive and profitable.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Base universe (can be expanded later)
        self.base_universe = self.config.get("base_universe", [
            "BTC_EUR", "ETH_EUR", "SOL_EUR", "XRP_EUR", "BNB_EUR"
        ])

        self.min_liquidity_score = self.config.get("min_liquidity_score", 0.4)  # lowered in aggressive test mode

    def select_assets(self, regime: str, candles_dict: Dict[str, pd.DataFrame],
                      max_assets: int = 5) -> List[AssetCandidate]:
        """
        Returns the best assets to trade given the current regime and market data.
        """
        candidates: List[AssetCandidate] = []

        for symbol, candles in candles_dict.items():
            if symbol not in self.base_universe or candles is None or len(candles) < 30:
                continue

            # Simple heuristics for now (will be improved)
            recent_volume = candles['volume'].tail(20).mean()
            avg_volume = candles['volume'].mean()
            liquidity_score = min(recent_volume / (avg_volume + 1e-9), 1.5)

            returns = candles['close'].pct_change().dropna()
            volatility = returns.tail(30).std()

            # Volatility scoring: sweet spot for momentum (not too low, not insane)
            # Aggressive test mode (A): be much more permissive in low_vol_chop
            if volatility < 0.008:
                vol_score = 0.65   # was 0.3 - allow trading in quiet markets
            elif volatility > 0.06:
                vol_score = 0.5
            else:
                vol_score = 0.9

            # Regime fit
            if regime == "trending":
                regime_fit = 0.9 if symbol in ["BTC_EUR", "ETH_EUR", "SOL_EUR"] else 0.6
            elif regime == "ranging":
                regime_fit = 0.85 if symbol in ["BNB_EUR", "XRP_EUR"] else 0.5
            elif regime == "high_vol_event":
                regime_fit = 0.7 if symbol in ["BTC_EUR", "ETH_EUR"] else 0.4   # Safer assets
            else:
                regime_fit = 0.6

            total_score = (liquidity_score * 0.35 + vol_score * 0.35 + regime_fit * 0.3)

            reason = f"Liquidity {liquidity_score:.2f} | Vol {volatility:.2%} | Regime fit {regime_fit:.2f}"

            candidates.append(AssetCandidate(
                symbol=symbol,
                score=total_score,
                liquidity_score=liquidity_score,
                volatility_score=vol_score,
                regime_fit=regime_fit,
                reason=reason
            ))

        # Sort and filter
        candidates.sort(key=lambda x: x.score, reverse=True)
        filtered = [c for c in candidates if c.liquidity_score >= self.min_liquidity_score]

        return filtered[:max_assets]

    def get_current_universe(self, regime: str) -> List[str]:
        """Returns the list of symbols we should actually monitor/trade right now."""
        candidates = self.select_assets(regime, {})  # Will be called with real data later
        return [c.symbol for c in candidates]
