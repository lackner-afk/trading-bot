"""
Reconciliation für Spot-Venues (Bitpanda Fusion).

Der springende Punkt: auf einem Spot-Venue gibt es kein eigenes
Positions-Konzept. Der **Base-Asset-Bestand ist die Position** — wer 0.0004 BTC
hält, ist genau in diesem Umfang long BTC_EUR. Damit lässt sich der Abgleich,
den `core/reconciliation.py::_reconcile_positions` bislang nur als Warnung
formulieren konnte, tatsächlich durchführen.

Zwei Driftrichtungen, beide gefährlich:

- **Verwaiste lokale Position**: der Bot glaubt, BTC zu halten, das Konto sagt
  nein. Er würde einen Verkauf versuchen, der scheitert, und beim Kill-Switch
  landen.
- **Unbekannter Bestand**: auf dem Konto liegt BTC, das der Bot nicht kennt.
  Meist manuell gekauft — der Bot darf es nicht anfassen, muss es aber melden,
  weil es sein Exposure verfälscht.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.symbols import base_asset, quote_asset

logger = logging.getLogger(__name__)


@dataclass
class SpotDrift:
    """Eine Abweichung zwischen lokalem Zustand und Konto."""
    symbol: str
    kind: str            # 'orphaned_local' | 'unknown_holding' | 'size_mismatch'
    local_amount: float
    exchange_amount: float
    detail: str = ""


@dataclass
class SpotReconciliationReport:
    balance_synced: bool = False
    local_quote_balance: float = 0.0
    exchange_quote_balance: float = 0.0
    quote_difference: float = 0.0
    drifts: List[SpotDrift] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = False

    @property
    def has_blocking_drift(self) -> bool:
        """
        Verwaiste lokale Positionen blockieren den Start.

        Ein unbekannter Bestand auf dem Konto ist unschön, aber der Bot kann
        damit sicher weiterlaufen — er fasst ihn nicht an. Eine lokale
        Position ohne Deckung dagegen führt garantiert zu einem
        fehlschlagenden Exit.
        """
        return any(d.kind == 'orphaned_local' for d in self.drifts)


class SpotReconciler:
    """Gleicht Portfolio-Zustand mit den Kontobeständen des Venues ab."""

    # Staubgrenze: Restbeträge unterhalb dieses Gegenwerts sind kein Bestand
    DUST_THRESHOLD_QUOTE = 1.0
    # Toleranz beim Mengenvergleich (Rundung, Teilausführungen, Gebühren in Base)
    SIZE_TOLERANCE = 0.02

    def __init__(self, portfolio, engine, quote_currency: str = "EUR"):
        self.portfolio = portfolio
        self.engine = engine
        self.quote_currency = quote_currency.upper()
        self.logger = logging.getLogger('SpotReconciler')

    async def reconcile(self, prices: Dict[str, float] = None) -> SpotReconciliationReport:
        """
        Führt den Abgleich durch.

        `prices` dient nur dazu, Bestände in Quote-Gegenwert umzurechnen —
        ohne Preise wird die Staubgrenze übersprungen.
        """
        report = SpotReconciliationReport()
        prices = prices or {}

        try:
            balances = await self.engine.fetch_balance()
        except Exception as e:
            report.errors.append(f"Kontostand nicht abrufbar: {e}")
            self._log(report)
            return report

        if not isinstance(balances, dict):
            report.errors.append(f"Unerwartetes Balance-Format: {type(balances).__name__}")
            self._log(report)
            return report

        self._sync_quote_balance(balances, report)
        self._compare_positions(balances, prices, report)

        report.success = not report.errors and not report.has_blocking_drift
        self._log(report)
        return report

    # ----- Teilschritte ---------------------------------------------

    def _sync_quote_balance(self, balances: Dict[str, float],
                            report: SpotReconciliationReport):
        """Übernimmt den echten Quote-Bestand als Wahrheit."""
        exchange_quote = float(balances.get(self.quote_currency, 0.0))
        local_quote = float(self.portfolio.balance)

        report.exchange_quote_balance = exchange_quote
        report.local_quote_balance = local_quote
        report.quote_difference = exchange_quote - local_quote

        if abs(report.quote_difference) > 1.0:
            report.warnings.append(
                f"{self.quote_currency}-Bestand weicht ab: lokal {local_quote:.2f}, "
                f"Konto {exchange_quote:.2f} (Differenz {report.quote_difference:+.2f})"
            )
            # Das Konto ist die Wahrheit — der Bot rechnet sonst mit Geld,
            # das er nicht hat, oder lässt Kapital ungenutzt.
            self.portfolio.balance = exchange_quote
            self.portfolio._update_equity()
            self.portfolio._save_state()
            report.actions_taken.append(
                f"Lokale Balance auf Kontostand gesetzt ({exchange_quote:.2f} {self.quote_currency})"
            )

        report.balance_synced = True

    def _compare_positions(self, balances: Dict[str, float], prices: Dict[str, float],
                           report: SpotReconciliationReport):
        """Vergleicht lokale Positionen mit den Base-Asset-Beständen."""
        local_bases: Dict[str, str] = {}

        for symbol, position in self.portfolio.positions.items():
            try:
                base = base_asset(symbol)
            except ValueError:
                report.warnings.append(f"Symbol {symbol} nicht interpretierbar")
                continue

            local_bases[base] = symbol
            held = float(balances.get(base, 0.0))
            price = prices.get(symbol) or position.entry_price
            expected = position.size / price if price > 0 else 0.0

            if held <= 0 or (expected > 0 and held < expected * (1 - self.SIZE_TOLERANCE)):
                report.drifts.append(SpotDrift(
                    symbol=symbol, kind='orphaned_local',
                    local_amount=expected, exchange_amount=held,
                    detail=(f"Lokale Position erwartet {expected:.8f} {base}, "
                            f"Konto haelt {held:.8f}"),
                ))
                report.errors.append(
                    f"Verwaiste lokale Position {symbol}: erwartet {expected:.8f} {base}, "
                    f"vorhanden {held:.8f}. Ein Exit wuerde fehlschlagen."
                )
            elif expected > 0 and held > expected * (1 + self.SIZE_TOLERANCE):
                report.drifts.append(SpotDrift(
                    symbol=symbol, kind='size_mismatch',
                    local_amount=expected, exchange_amount=held,
                    detail=f"Konto haelt mehr {base} als die lokale Position ausweist",
                ))
                report.warnings.append(
                    f"{symbol}: Konto haelt {held:.8f} {base}, lokale Position nur "
                    f"{expected:.8f} - moeglicherweise manuell zugekauft"
                )

        # Bestände ohne lokale Entsprechung
        for asset, amount in balances.items():
            asset = asset.upper()
            if asset == self.quote_currency or amount <= 0 or asset in local_bases:
                continue

            symbol = f"{asset}_{self.quote_currency}"
            price = prices.get(symbol, 0.0)
            value = amount * price if price else None

            if value is not None and value < self.DUST_THRESHOLD_QUOTE:
                continue    # Staub, kein Bestand

            report.drifts.append(SpotDrift(
                symbol=symbol, kind='unknown_holding',
                local_amount=0.0, exchange_amount=float(amount),
                detail=(f"{amount:.8f} {asset} auf dem Konto, dem Bot unbekannt"
                        + (f" (~{value:.2f} {self.quote_currency})" if value else "")),
            ))
            report.warnings.append(
                f"Unbekannter Bestand: {amount:.8f} {asset} - der Bot fasst ihn nicht an, "
                f"er verfaelscht aber die Exposure-Rechnung"
            )

        if not report.drifts:
            report.actions_taken.append("Positionen stimmen mit den Kontobestaenden ueberein")

    def _log(self, report: SpotReconciliationReport):
        self.logger.info("=" * 60)
        self.logger.info("Spot-Reconciliation")
        self.logger.info("=" * 60)
        self.logger.info(
            f"{self.quote_currency}: lokal {report.local_quote_balance:.2f} | "
            f"Konto {report.exchange_quote_balance:.2f} "
            f"(Differenz {report.quote_difference:+.2f})"
        )
        for action in report.actions_taken:
            self.logger.info(f"  OK   {action}")
        for warning in report.warnings:
            self.logger.warning(f"  WARN {warning}")
        for error in report.errors:
            self.logger.critical(f"  FEHLER {error}")
        self.logger.info(
            f"Ergebnis: {'OK' if report.success else 'NICHT BESTANDEN'}"
        )
        self.logger.info("=" * 60)


async def run_spot_reconciliation(portfolio, engine, prices: Dict[str, float] = None,
                                  quote_currency: str = "EUR") -> SpotReconciliationReport:
    """Bequemer Einstiegspunkt für main.py."""
    return await SpotReconciler(portfolio, engine, quote_currency).reconcile(prices)
