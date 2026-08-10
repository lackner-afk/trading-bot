"""
Marktbeschränkungen des Ziel-Venues.

Hintergrund: Bitpanda Fusion ist derzeit Spot-only — kein Leverage, kein
Short-Selling (Margin ist bei Bitpanda als "coming soon" angekündigt). Die
ConfluenceStrategy erzeugt aber von Haus aus Leverage 2–18x und SHORT-Signale.

Würde der Bot die Testphase mit Leverage und Shorts fahren, beschriebe die
gesammelte Performance eine Strategie, die live gar nicht ausführbar ist —
die Zahlen wären für eine Go-Live-Entscheidung wertlos. Deshalb gelten die
Beschränkungen schon im Paper-Modus.

Durchgesetzt wird das an vier unabhängigen Stellen (defense in depth), damit
kein einzelner Bug einen Short oder gehebelte Order durchlässt:
  1. SignalAggregator  — erzeugt gar kein Short-Entry, Leverage fix 1.0
  2. main.py           — routet Short-Signale als Exit für offene Longs
  3. RiskManager       — blockt jeden Trade mit Leverage > 1
  4. Portfolio         — weigert sich, eine Short-Position zu buchen
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MarketConstraints:
    """Was auf dem Ziel-Venue tatsächlich handelbar ist."""

    spot_only: bool = True
    allow_short: bool = False
    max_leverage: float = 1.0

    # Was mit einem SHORT-Signal passiert, wenn Shorts nicht erlaubt sind:
    #   "exit_only" — schließt eine offene Long-Position auf dem Symbol
    #                 (die sinnvolle Spot-Semantik: "verkauf, was du hast")
    #   "ignore"    — Signal wird verworfen
    short_signal_policy: str = "exit_only"

    # Mindest-Ordervolumen des Venues. Bei 100 EUR Kapital und 20 %
    # Positionsgröße liegt eine Order bei ~20 EUR — das ist nah genug an
    # typischen Mindestgrößen, um es explizit zu prüfen.
    min_order_notional_eur: float = 10.0

    @classmethod
    def from_config(cls, config: Dict = None) -> "MarketConstraints":
        """
        Baut die Constraints aus dem `trading:`-Block der settings.yaml.

        Fehlt der Block, gelten die Spot-Defaults — der sichere Fall.
        """
        cfg = (config or {}).get("trading", {}) or {}
        return cls(
            spot_only=bool(cfg.get("spot_only", True)),
            allow_short=bool(cfg.get("allow_short", False)),
            max_leverage=float(cfg.get("max_leverage", 1.0)),
            short_signal_policy=str(cfg.get("short_signal_policy", "exit_only")),
            min_order_notional_eur=float(cfg.get("min_order_notional_eur", 10.0)),
        )

    def clamp_leverage(self, leverage: float) -> float:
        """Begrenzt Leverage auf das, was das Venue hergibt."""
        if self.spot_only:
            return 1.0
        return max(1.0, min(float(leverage), self.max_leverage))

    def describe(self) -> str:
        """Einzeiler fürs Startup-Log."""
        if self.spot_only:
            return (
                f"SPOT-Modus: long-only, Leverage 1.0, "
                f"Short-Signale → {self.short_signal_policy}"
            )
        return f"Margin-Modus: max Leverage {self.max_leverage}x, Shorts {'an' if self.allow_short else 'aus'}"
