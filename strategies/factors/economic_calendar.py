"""
Economic Calendar Helper (Phase 3 foundation)

This module provides infrastructure for handling major macro events
like CPI, FOMC, Non-Farm Payrolls, etc.

It supports:
- Defining known high-impact events
- Adding upcoming events with forecast/previous values
- Feeding actual results and calculating "surprise"
- Querying whether we are currently in a high-risk event window
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum


class EventImpact(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EconomicEvent:
    name: str
    time: datetime
    impact: EventImpact = EventImpact.HIGH
    currency: str = "USD"           # USD, EUR, etc.
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    surprise: Optional[float] = None   # (actual - forecast) / |forecast| or similar

    def calculate_surprise(self) -> Optional[float]:
        if self.actual is not None and self.forecast is not None and self.forecast != 0:
            self.surprise = (self.actual - self.forecast) / abs(self.forecast)
        return self.surprise


class EconomicCalendar:
    """
    Manages known and upcoming high-impact economic events.

    This is the foundation for smart macro-aware trading.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.events: List[EconomicEvent] = []

        # Default list of major recurring high-impact events (US-focused for now)
        self.major_event_names = {
            "CPI", "Core CPI", "FOMC", "Fed Rate Decision", "FOMC Press Conference",
            "Non-Farm Payrolls", "NFP", "Unemployment Rate", "Average Hourly Earnings",
            "GDP", "Core GDP", "Retail Sales", "Core Retail Sales",
            "PPI", "Core PPI", "ISM Manufacturing", "ISM Services",
            "Consumer Confidence", "Michigan Consumer Sentiment"
        }

    def add_event(self, event: EconomicEvent):
        """Add an upcoming or known event."""
        self.events.append(event)
        # Keep sorted by time
        self.events.sort(key=lambda e: e.time)

    def add_upcoming_event(self, name: str, time: datetime,
                           impact: EventImpact = EventImpact.HIGH,
                           currency: str = "USD",
                           forecast: Optional[float] = None,
                           previous: Optional[float] = None):
        """Convenience method to add an event with consensus numbers."""
        event = EconomicEvent(
            name=name,
            time=time,
            impact=impact,
            currency=currency,
            forecast=forecast,
            previous=previous
        )
        self.add_event(event)

    def feed_actual_result(self, name: str, actual: float, time: Optional[datetime] = None):
        """
        Feed the actual result for an event (usually after release).
        Calculates the surprise automatically.
        """
        for event in self.events:
            if event.name.lower() == name.lower():
                if time is None or abs((event.time - time).total_seconds()) < 3600:
                    event.actual = actual
                    event.calculate_surprise()
                    return True
        return False

    def get_upcoming_events(self, hours_ahead: int = 48) -> List[EconomicEvent]:
        """Return events in the next X hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)
        return [e for e in self.events if now <= e.time <= cutoff and e.impact == EventImpact.HIGH]

    def is_in_event_window(self, hours_before: int = 4, hours_after: int = 2) -> Optional[EconomicEvent]:
        """
        Check if we are currently inside the risk window of a high-impact event.
        Returns the event if yes, otherwise None.
        """
        now = datetime.now()
        for event in self.events:
            if event.impact != EventImpact.HIGH:
                continue

            window_start = event.time - timedelta(hours=hours_before)
            window_end = event.time + timedelta(hours=hours_after)

            if window_start <= now <= window_end:
                return event
        return None

    def get_current_risk_multiplier(self, hours_before: int = 4, hours_after: int = 2) -> float:
        """
        Returns a suggested position size multiplier based on proximity to events.
        1.0 = normal risk
        < 1.0 = reduce risk
        """
        event = self.is_in_event_window(hours_before, hours_after)
        if event:
            # Stronger reduction right before and right after the release
            return 0.4
        return 1.0

    def get_next_major_event(self) -> Optional[EconomicEvent]:
        """Returns the next high-impact event (if any)."""
        now = datetime.now()
        upcoming = [e for e in self.events if e.time >= now and e.impact == EventImpact.HIGH]
        return upcoming[0] if upcoming else None
