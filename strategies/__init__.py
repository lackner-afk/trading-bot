"""
Strategies package - Old strategies (legacy) + New 2026 Multi-Factor System

New architecture lives under:
- strategies/factors/
- strategies/signal_aggregator.py
- strategies/regime_detector.py
- strategies/asset_selector.py
"""

# Legacy strategies (will be phased out)
from .crypto_scalper import CryptoScalper
from .momentum import MomentumStrategy
from .ml_predictor import MLPredictor

# New 2026 Multi-Factor System
from .regime_detector import RegimeDetector, MarketRegime
from .asset_selector import AssetSelector, AssetCandidate
from .signal_aggregator import SignalAggregator, TradeSignal
from .factors.technical import (
    MultiTimeframeTrendFactor,
    MomentumFactor,
    VolatilityFilter,
    BreakoutFactor,
    VolumeConfirmationFactor,
)
from .factors.macro_news import MacroNewsFilter
from .factors.economic_calendar import EconomicCalendar, EconomicEvent, EventImpact
from .factors.sentiment import SentimentFactor
from .confluence_strategy import ConfluenceStrategy
from .crypto_scalper import ScalperSignal, SignalType  # legacy types during transition

__all__ = [
    # Legacy
    'CryptoScalper', 'MomentumStrategy', 'MLPredictor',
    # New System
    'RegimeDetector', 'MarketRegime',
    'AssetSelector', 'AssetCandidate',
    'SignalAggregator', 'TradeSignal'
]
