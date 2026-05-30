"""
ConfluenceStrategy - The new main strategy class using the multi-factor system.

This class combines:
- RegimeDetector
- AssetSelector / UniverseManager
- Multiple Factors (technical + macro + later sentiment)
- SignalAggregator

It is designed to eventually replace the old MomentumStrategy / CryptoScalper.
"""

from typing import Dict, List, Optional
import pandas as pd

from .regime_detector import RegimeDetector
from .asset_selector import AssetSelector
from .universe_manager import UniverseManager
from .signal_aggregator import SignalAggregator, TradeSignal
from .factors.base import Factor
from .crypto_scalper import ScalperSignal, SignalType  # for compatibility during transition


class ConfluenceStrategy:
    """
    Modern multi-factor strategy using confluence scoring.

    Usage example:
        strategy = ConfluenceStrategy(config=...)
        strategy.add_factor(MultiTimeframeTrendFactor())
        strategy.add_factor(BreakoutFactor())
        ...

        signal = strategy.analyze(symbol, candles, current_price)
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.regime_detector = RegimeDetector(self.config.get("regime", {}))
        self.asset_selector = AssetSelector(self.config.get("asset_selector", {}))
        self.universe_manager = UniverseManager(self.config.get("universe", {}))
        self.aggregator = SignalAggregator(self.config.get("aggregator", {}))

        self.factors: List[Factor] = []

        # Current regime cache (updated when analyze is called)
        self._last_regime = None
        self._last_universe = None

    @classmethod
    def create_default(cls, config: Dict = None):
        """
        Factory method that creates a ConfluenceStrategy with a strong
        default set of factors. This is the recommended way to get started.
        """
        strategy = cls(config)
        strategy.add_common_factors(include_sentiment=True, include_macro=True)
        return strategy

    def add_factor(self, factor: Factor):
        """Add a factor to the strategy."""
        self.factors.append(factor)

    def set_factors(self, factors: List[Factor]):
        """Replace all factors."""
        self.factors = factors

    def add_common_factors(self, include_sentiment: bool = True, include_macro: bool = True):
        """
        Convenience method to quickly add a solid set of default factors.
        This is the recommended way to get a strong starting configuration.
        """
        from .factors.technical import (
            MultiTimeframeTrendFactor,
            MomentumFactor,
            VolatilityFilter,
            BreakoutFactor,
            VolumeConfirmationFactor,
        )

        self.add_factor(MultiTimeframeTrendFactor())
        self.add_factor(MomentumFactor())
        self.add_factor(VolatilityFilter())
        self.add_factor(BreakoutFactor())
        self.add_factor(VolumeConfirmationFactor())

        if include_macro:
            from .factors.macro_news import MacroNewsFilter
            self.add_factor(MacroNewsFilter())

        if include_sentiment:
            from .factors.sentiment import SentimentFactor
            self.add_factor(SentimentFactor())

    def analyze(self, symbol: str, candles: pd.DataFrame,
                current_price: float) -> Optional[TradeSignal]:
        """
        Main entry point. Analyzes the market using all registered factors
        + regime + asset selection logic.
        """
        if not self.factors:
            return None

        # 1. Detect current regime
        regime = self.regime_detector.detect(symbol, candles)
        self._last_regime = regime

        # 2. Get factor results
        factor_results = []
        for factor in self.factors:
            if not factor.is_enabled():
                continue
            try:
                result = factor.calculate(symbol, candles, current_price, regime=regime.name)
                if result:
                    factor_results.append(result)
            except Exception as e:
                # Fail gracefully per factor
                print(f"[ConfluenceStrategy] Factor {factor.name} failed: {e}")

        if not factor_results:
            return None

        # === Phase 6 Debug Logging: Show exactly what each factor scored ===
        if factor_results:
            factor_debug = []
            for fr in sorted(factor_results, key=lambda x: x.score, reverse=True):
                factor_debug.append(
                    f"{fr.name}: score={fr.score:.2f} dir={fr.direction or '-'} | {fr.reason[:60]}"
                )
            print(f"[CONFLUENCE FACTORS] {symbol} | Regime={regime.name} | " + " || ".join(factor_debug))

        # 3. Get macro risk adjustment
        macro_filter = next((f for f in self.factors if f.name == "macro_news_filter"), None)
        macro_risk = macro_filter.get_risk_multiplier() if macro_filter else 1.0

        # 4. Aggregate into final signal with regime-aware weights
        signal = self.aggregator.aggregate(
            symbol=symbol,
            current_price=current_price,
            factor_results=factor_results,
            regime=regime.name,
            regime_characteristics=regime.characteristics,
            macro_risk_multiplier=macro_risk
        )

        return signal

    def get_current_regime(self):
        return self._last_regime

    def get_recommended_assets(self, candles_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Returns recommended assets for the current regime."""
        return self.universe_manager.get_current_universe(candles_dict)

    def get_macro_risk_multiplier(self) -> float:
        """Returns the current macro risk multiplier (1.0 = normal, <1.0 = reduced risk)."""
        macro_filter = next(
            (f for f in self.factors if f.name == "macro_news_filter"),
            None
        )
        if macro_filter:
            return macro_filter.get_risk_multiplier()
        return 1.0

    def analyze_legacy(self, symbol: str, candles: pd.DataFrame,
                       current_price: float) -> Optional[ScalperSignal]:
        """
        Compatibility method that returns the old ScalperSignal format.
        This allows us to gradually migrate the loops in main.py.

        Phase 6 enhancement: attaches _confluence_data (factor_breakdown, regime, etc.)
        so that rich attribution is available even when using the legacy wrapper.
        """
        signal = self.analyze(symbol, candles, current_price)

        if signal is None:
            return None

        signal_type = SignalType.LONG if signal.direction == "long" else SignalType.SHORT

        legacy = ScalperSignal(
            signal_type=signal_type,
            symbol=signal.symbol,
            price=current_price,
            confidence=signal.confidence,
            reason=signal.reason,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            suggested_leverage=int(signal.suggested_leverage),
            atr_value=0.0,  # Can be improved later
            timestamp=signal.timestamp
        )

        # Phase 6: Carry rich confluence attribution data on the legacy signal
        # Access via getattr(signal, '_confluence_data', None)
        legacy._confluence_data = {
            "factor_breakdown": signal.factor_breakdown,
            "confluence_score": signal.confluence_score,
            "regime": getattr(self, '_last_regime', None),
            "macro_risk_multiplier": self.get_macro_risk_multiplier(),
        }

        return legacy
