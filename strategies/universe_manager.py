"""
Universe Manager

Combines RegimeDetector + AssetSelector to give the bot
a clean, ready-to-use set of assets + recommended style for the current regime.
"""

from typing import List, Dict
from .regime_detector import RegimeDetector, MarketRegime
from .asset_selector import AssetSelector, AssetCandidate


class UniverseManager:
    """
    High-level component that tells the bot:
    - What regime we are in
    - Which assets to focus on right now
    - Which strategy styles are preferred
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.regime_detector = RegimeDetector(config.get("regime", {}))
        self.asset_selector = AssetSelector(config.get("asset_selector", {}))

    def get_current_universe(self, candles_dict: Dict[str, "pd.DataFrame"]) -> Dict:
        """
        Main method to call from the main loop or aggregator.

        Returns a dict with:
        - regime
        - preferred_styles
        - selected_assets (list of AssetCandidate)
        """
        # Pick one major symbol to determine overall regime (BTC is usually best)
        regime_symbol = "BTC_EUR"
        if regime_symbol not in candles_dict:
            regime_symbol = list(candles_dict.keys())[0]

        regime = self.regime_detector.detect(regime_symbol, candles_dict.get(regime_symbol))

        selected = self.asset_selector.select_assets(
            regime=regime.name,
            candles_dict=candles_dict,
            max_assets=self.config.get("max_assets", 5)
        )

        preferred_styles = self.regime_detector.get_preferred_style(regime)

        return {
            "regime": regime,
            "preferred_styles": preferred_styles,
            "selected_assets": selected,
            "symbols": [a.symbol for a in selected]
        }
