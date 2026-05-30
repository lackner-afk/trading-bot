"""
Base classes for the new multi-factor strategy system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class FactorResult:
    """
    Result returned by a single factor.
    """
    name: str
    score: float                  # 0.0 to 1.0 (or -1.0 to 1.0 for directional)
    confidence: float             # How confident we are in this factor (0-1)
    direction: Optional[str] = None   # "long", "short", or None
    reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Factor(ABC):
    """
    Abstract base class for all trading factors.

    A Factor analyzes market data (and optionally external data) and returns
    a score that contributes to the overall trading decision.
    """

    name: str = "base_factor"

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    @abstractmethod
    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:
        """
        Calculate the factor's contribution.

        Args:
            symbol: Trading pair (e.g. "BTC_EUR")
            candles: DataFrame with OHLCV + indicators
            current_price: Latest price
            **kwargs: Additional context (e.g. sentiment data, macro events)

        Returns:
            FactorResult or None if the factor cannot produce a signal
        """
        pass

    def is_enabled(self) -> bool:
        return self.enabled
