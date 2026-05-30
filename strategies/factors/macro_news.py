"""
Advanced Macro / News Filter (Phase 3)

This version integrates with the EconomicCalendar for proper event awareness.
It can:
- Automatically detect when we are in a high-impact event window
- Apply risk reduction (lower score + suggested position size multiplier)
- React to "surprises" once actual results are fed in
"""

from typing import Optional, Dict
from datetime import datetime

import pandas as pd

from .base import Factor, FactorResult
from .economic_calendar import EconomicCalendar, EconomicEvent


class MacroNewsFilter(Factor):
    """
    Phase 3 version of the macro/news filter.

    It now uses an EconomicCalendar instance for real event management.
    """

    name = "macro_news_filter"

    def __init__(self, config: Dict = None, calendar: Optional[EconomicCalendar] = None):
        super().__init__(config)

        self.calendar = calendar or EconomicCalendar()

        # How aggressive we are with risk reduction around events
        self.event_risk_multiplier = self.config.get("event_risk_multiplier", 0.4)
        self.hours_before = self.config.get("hours_before", 4)
        self.hours_after = self.config.get("hours_after", 2)

    def set_calendar(self, calendar: EconomicCalendar):
        """Inject or replace the economic calendar."""
        self.calendar = calendar

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        now = datetime.now()

        active_event: Optional[EconomicEvent] = self.calendar.is_in_event_window(
            self.hours_before, self.hours_after
        )

        if active_event:
            # More nuanced scoring based on timing and surprise
            event_time = active_event.time
            minutes_to_event = (event_time - now).total_seconds() / 60

            base_score = self.event_risk_multiplier

            # Stronger reduction right before the release (highest uncertainty)
            if minutes_to_event > 0 and minutes_to_event < 90:
                base_score *= 0.7  # even more conservative pre-release

            # After the release, behavior depends on surprise
            surprise = active_event.surprise
            if surprise is not None:
                abs_surprise = abs(surprise)
                if abs_surprise > 1.0:  # Very large surprise
                    base_score *= 0.6
                    reason = f"Post {active_event.name} - Very large surprise ({surprise:.1%}) → strong risk reduction"
                elif abs_surprise > 0.4:
                    base_score *= 0.75
                    reason = f"Post {active_event.name} - Significant surprise ({surprise:.1%})"
                else:
                    reason = f"Post {active_event.name} - In-line result"
            else:
                reason = f"High-impact event window: {active_event.name}"

            return FactorResult(
                name=self.name,
                score=max(0.25, base_score),  # never go below 25% conviction
                confidence=0.9,
                direction=None,
                reason=reason,
                metadata={
                    "in_event_window": True,
                    "active_event": active_event.name,
                    "minutes_to_event": round(minutes_to_event),
                    "surprise": surprise,
                    "risk_multiplier": base_score
                }
            )

        # No event window
        return FactorResult(
            name=self.name,
            score=1.0,
            confidence=0.65,
            direction=None,
            reason="No major macro event window",
            metadata={"in_event_window": False}
        )

    def get_risk_multiplier(self) -> float:
        """
        Returns the current suggested position size multiplier (0.0 - 1.0).
        This should be multiplied with normal position sizes.
        """
        return self.calendar.get_current_risk_multiplier(
            self.hours_before, self.hours_after
        )
