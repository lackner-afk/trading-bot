"""
New Multi-Factor Strategy System (2026 Overhaul)

This package contains independent factor modules that contribute to trading decisions.
Factors are combined in the SignalAggregator using a confluence scoring system.
"""

from .base import Factor, FactorResult
from .technical import (
    MultiTimeframeTrendFactor,
    MomentumFactor,
    VolatilityFilter,
    BreakoutFactor,
    VolumeConfirmationFactor,
)
from .macro_news import MacroNewsFilter
from .economic_calendar import EconomicCalendar, EconomicEvent, EventImpact
from .sentiment import SentimentFactor

__all__ = [
    "Factor",
    "FactorResult",
    "MultiTimeframeTrendFactor",
    "MomentumFactor",
    "VolatilityFilter",
]