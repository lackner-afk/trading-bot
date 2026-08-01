"""
Exit-Logik für Positionen der ConfluenceStrategy.

Vorher liefen Confluence-Exits über `self.momentum` — eine deaktivierte
Strategie. Zwei Dinge waren daran kaputt:

1. `MomentumStrategy.highest_prices` wird nur im Konstruktor initialisiert
   und nirgends im Repo je befüllt. `strategy.highest_prices.get(symbol)`
   lieferte damit immer None, der Trailing-Stop war toter Code.
2. Confluence-Signale tragen `atr_value=0.0`, weshalb auch alle
   ATR-basierten Pfade der Momentum-Exit-Logik leerliefen.

Dieser Manager pflegt das Hoch/Tief seit Entry selbst und holt die ATR aus
dem Feed statt aus einem Strategie-Cache, der nie gefüllt wird.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExitState:
    """Laufendes Tracking einer offenen Position."""
    entry_price: float
    entry_time: datetime
    highest: float
    lowest: float
    trailing_armed: bool = False


@dataclass
class ConfluenceExitManager:
    """
    Prüft Stop-Loss, Take-Profit, Trailing-Stop und optionalen Zeit-Stop.

    Die Preisspur wird über `update_price()` aus dem Main-Loop gefüttert —
    genau der Schritt, der in der alten Momentum-Exit-Logik fehlte.
    """

    # Trailing-Stop als ATR-Vielfaches; ohne ATR greift der Prozent-Fallback
    trailing_atr_multiplier: float = 1.2
    trailing_pct_fallback: float = 0.005
    # Trailing wird erst scharf, wenn die Position im Gewinn liegt
    trailing_arm_profit_pct: float = 0.005
    # Maximale Haltedauer in Stunden (0 = aus)
    max_hold_hours: float = 0.0

    states: Dict[str, ExitState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict = None) -> "ConfluenceExitManager":
        cfg = (config or {}).get("exit", {}) or {}
        return cls(
            trailing_atr_multiplier=float(cfg.get("trailing_atr_multiplier", 1.2)),
            trailing_pct_fallback=float(cfg.get("trailing_pct_fallback", 0.005)),
            trailing_arm_profit_pct=float(cfg.get("trailing_arm_profit_pct", 0.005)),
            max_hold_hours=float(cfg.get("max_hold_hours", 0.0)),
        )

    # ----- Tracking -------------------------------------------------

    def register(self, symbol: str, entry_price: float,
                 entry_time: Optional[datetime] = None):
        """Beginnt das Tracking für eine neu geöffnete Position."""
        self.states[symbol] = ExitState(
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(),
            highest=entry_price,
            lowest=entry_price,
        )

    def forget(self, symbol: str):
        """Beendet das Tracking (Position geschlossen)."""
        self.states.pop(symbol, None)

    def update_price(self, symbol: str, price: float):
        """Aktualisiert Hoch/Tief seit Entry. Muss bei jedem Preistick laufen."""
        state = self.states.get(symbol)
        if state is None:
            return
        if price > state.highest:
            state.highest = price
        if price < state.lowest:
            state.lowest = price

    def ensure_tracked(self, symbol: str, entry_price: float,
                       entry_time: Optional[datetime] = None):
        """
        Legt fehlendes Tracking nachträglich an — etwa für Positionen, die
        beim Neustart aus der DB geladen wurden.
        """
        if symbol not in self.states:
            self.register(symbol, entry_price, entry_time)

    # ----- Entscheidung ---------------------------------------------

    def check_exit(self, symbol: str, entry_price: float, current_price: float,
                   side: str, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   atr: Optional[float] = None,
                   entry_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Returns (should_exit, reason).

        Reihenfolge ist bewusst: harte Stops zuerst, danach Trailing, zuletzt
        der Zeit-Stop.
        """
        if current_price <= 0 or entry_price <= 0:
            return False, ""

        self.ensure_tracked(symbol, entry_price, entry_time)
        self.update_price(symbol, current_price)
        state = self.states[symbol]

        is_long = side == "long"

        # 1. Stop-Loss
        if stop_loss and stop_loss > 0:
            if is_long and current_price <= stop_loss:
                return True, f"Stop-Loss ({current_price:.2f} <= {stop_loss:.2f})"
            if not is_long and current_price >= stop_loss:
                return True, f"Stop-Loss ({current_price:.2f} >= {stop_loss:.2f})"

        # 2. Take-Profit
        if take_profit and take_profit > 0:
            if is_long and current_price >= take_profit:
                return True, f"Take-Profit ({current_price:.2f} >= {take_profit:.2f})"
            if not is_long and current_price <= take_profit:
                return True, f"Take-Profit ({current_price:.2f} <= {take_profit:.2f})"

        # 3. Trailing-Stop — erst scharf, wenn genug Gewinn aufgelaufen ist
        profit_pct = ((current_price - entry_price) / entry_price) if is_long \
            else ((entry_price - current_price) / entry_price)

        if profit_pct >= self.trailing_arm_profit_pct:
            state.trailing_armed = True

        if state.trailing_armed:
            if atr and atr > 0:
                distance = atr * self.trailing_atr_multiplier
            else:
                distance = current_price * self.trailing_pct_fallback

            if is_long:
                trigger = state.highest - distance
                if current_price <= trigger:
                    return True, (
                        f"Trailing-Stop (Hoch {state.highest:.2f} - {distance:.2f} "
                        f"= {trigger:.2f})"
                    )
            else:
                trigger = state.lowest + distance
                if current_price >= trigger:
                    return True, (
                        f"Trailing-Stop (Tief {state.lowest:.2f} + {distance:.2f} "
                        f"= {trigger:.2f})"
                    )

        # 4. Zeit-Stop
        if self.max_hold_hours > 0:
            held_hours = (datetime.now() - state.entry_time).total_seconds() / 3600.0
            if held_hours >= self.max_hold_hours:
                return True, f"Zeit-Stop ({held_hours:.1f}h >= {self.max_hold_hours}h)"

        return False, ""
