"""
Reconciliation Module für Live-Trading

Ziel: Nach Bot-Start (oder Crash) den lokalen Portfolio-Zustand mit dem echten
Stand auf der Exchange (One Trading) abgleichen.

Wichtige Szenarien:
- Bot war offline → Orders wurden auf Exchange gefüllt oder storniert
- Manuelle Trades auf der Exchange
- Partial fills
- Balance-Differenzen (Gebühren, Airdrops, etc.)

Verwendung (später in main.py):
    from core.reconciliation import Reconciler
    reconciler = Reconciler(portfolio, live_order_engine)
    report = await reconciler.reconcile_on_startup()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .portfolio import Portfolio, Position
from .live_order_engine import LiveOrderEngine
from .order_engine import Order, OrderStatus


@dataclass
class ReconciliationReport:
    """Ergebnis eines Reconciliation-Laufs"""
    timestamp: datetime
    balance_synced: bool = False
    local_balance: float = 0.0
    exchange_balance: float = 0.0
    balance_difference: float = 0.0

    open_orders_checked: int = 0
    orders_filled_while_offline: int = 0
    orders_cancelled_while_offline: int = 0
    orphaned_local_positions: int = 0   # Lokale Positionen ohne Gegenstück auf Exchange
    new_positions_from_exchange: int = 0

    actions_taken: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    success: bool = True


class Reconciler:
    """
    Führt Abgleich zwischen lokalem Portfolio und Exchange durch.
    """

    def __init__(self, portfolio: Portfolio, live_engine: LiveOrderEngine):
        self.portfolio = portfolio
        self.live_engine = live_engine
        self.logger = logging.getLogger('Reconciler')

    async def reconcile_on_startup(self) -> ReconciliationReport:
        """
        Wird beim Start des Bots im Live-Modus aufgerufen.
        Sollte früh im start() Flow passieren (nach Feed-Start, vor Strategie-Loops).
        """
        report = ReconciliationReport(timestamp=datetime.now())

        self.logger.info("=" * 60)
        self.logger.info("Starte Reconciliation mit One Trading...")
        self.logger.info("=" * 60)

        try:
            # 1. Balance Sync
            await self._sync_balance(report)

            # 2. Offene Orders abgleichen
            await self._reconcile_open_orders(report)

            # 3. Lokale Positionen vs. reale offene Orders / Balances prüfen
            self._reconcile_positions(report)

            report.success = len(report.errors) == 0

        except Exception as e:
            self.logger.error(f"Reconciliation fehlgeschlagen: {e}", exc_info=True)
            report.errors.append(str(e))
            report.success = False

        self._log_report(report)
        return report

    async def _sync_balance(self, report: ReconciliationReport):
        """Synchronisiert den Kontostand mit der Exchange."""
        report.local_balance = self.portfolio.balance

        try:
            exchange_bal = await self.live_engine.fetch_balance()
            if not exchange_bal:
                report.warnings.append("Konnte Balance nicht von Exchange abrufen")
                return

            # One Trading / CCXT gibt meist 'total' oder 'free' zurück.
            # Wir nehmen fürs Erste den EUR-Bestand (base_currency).
            # TODO: Besser machen, sobald wir wissen, welche Währung primär ist.
            total_eur = 0.0
            if 'EUR' in exchange_bal.get('total', {}):
                total_eur = float(exchange_bal['total']['EUR'])
            elif 'free' in exchange_bal and 'EUR' in exchange_bal['free']:
                total_eur = float(exchange_bal['free']['EUR'])

            report.exchange_balance = total_eur
            report.balance_difference = total_eur - report.local_balance

            # Für Live: Wir vertrauen erstmal der Exchange als Source of Truth für Balance
            # (später evtl. mit Delta-Tracking)
            if abs(report.balance_difference) > 1.0:  # > 1 EUR Toleranz
                self.portfolio.balance = total_eur
                self.portfolio._update_equity()
                self.portfolio._save_state()
                report.balance_synced = True
                report.actions_taken.append(
                    f"Balance angepasst: {report.local_balance:.2f} → {total_eur:.2f} EUR "
                    f"(Diff: {report.balance_difference:+.2f})"
                )
                self.logger.warning(f"Balance-Differenz erkannt und korrigiert: {report.balance_difference:+.2f} EUR")
            else:
                report.balance_synced = True
                report.actions_taken.append("Balance innerhalb Toleranz — kein Sync nötig")

        except Exception as e:
            report.errors.append(f"Balance Sync Fehler: {e}")

    async def _reconcile_open_orders(self, report: ReconciliationReport):
        """
        Holt offene Orders von der Exchange und gleicht sie mit lokalen Pending Orders ab.
        Da der aktuelle LiveOrderEngine noch kein internes pending_orders hat (im Gegensatz zur Paper Engine),
        fokussieren wir uns hier auf das Erkennen von fills während der Offline-Zeit.
        """
        try:
            # Aktuell hat LiveOrderEngine noch keine gute "pending" Verwaltung.
            # Wir holen alle offenen Orders und loggen sie.
            # Spätere Erweiterung: Vergleich mit lokalen offenen Orders aus Portfolio/Engine.

            # Fürs Erste: Wir können fetch_open_orders erweitern oder direkt nutzen.
            # Hier ein Platzhalter, der später mit echter Logik gefüllt wird.

            # Beispiel: Wenn wir in Zukunft lokale pending orders tracken würden:
            # local_pending = self.live_engine.get_pending_orders()  # falls implementiert

            open_orders = await self.live_engine.fetch_open_orders()
            report.open_orders_checked = len(open_orders)

            if open_orders:
                self.logger.info(f"{len(open_orders)} offene Order(s) auf Exchange gefunden.")
                for o in open_orders[:5]:  # Nur erste 5 loggen
                    self.logger.info(f"  - {o.get('id')}: {o.get('side')} {o.get('symbol')} @ {o.get('price')} (status: {o.get('status')})")

            # TODO Phase 4/5: Wenn lokale Orders existieren → prüfen ob sie auf Exchange noch open sind.
            # Falls nicht mehr open → prüfen ob gefüllt (über fetch_my_trades oder order history) und Trade/Position updaten.

            report.actions_taken.append(f"{len(open_orders)} offene Orders auf Exchange geprüft")

        except Exception as e:
            report.errors.append(f"Open Orders Reconciliation Fehler: {e}")

    def _reconcile_positions(self, report: ReconciliationReport):
        """
        Vergleicht lokale simulierte Positionen mit dem realen Stand auf der Exchange.

        Auf One Trading (Spot) gibt es keine "Leverage Positions" wie im Bot.
        Die Reconciliation hier dient vor allem dazu:
        - Warnungen auszugeben bei lokalen Positionen ohne reales Gegenstück
        - Später: echte Spot-Balances als "Positionen" zu interpretieren
        """
        local_positions = list(self.portfolio.positions.keys())

        if not local_positions:
            report.actions_taken.append("Keine lokalen Positionen vorhanden")
            return

        # Aktuell haben wir noch keine gute Möglichkeit, "real positions" von One Trading zu holen
        # (da es Spot ist). Wir warnen daher erstmal nur.

        for symbol in local_positions:
            pos = self.portfolio.positions[symbol]
            report.warnings.append(
                f"Lokale Position {symbol} existiert noch (Entry @ {pos.entry_price}, Size {pos.size}). "
                "Im reinen Spot-Modus muss dies manuell oder über Balance-Sync gehandhabt werden."
            )
            report.orphaned_local_positions += 1

        if report.orphaned_local_positions > 0:
            report.actions_taken.append(
                f"{report.orphaned_local_positions} lokale Position(en) ohne direkten Exchange-Gegenpart gefunden"
            )

    def _log_report(self, report: ReconciliationReport):
        self.logger.info("=" * 60)
        self.logger.info("Reconciliation Report")
        self.logger.info(f"Zeit: {report.timestamp}")
        self.logger.info(f"Erfolg: {report.success}")
        self.logger.info(f"Balance Sync: {report.balance_synced} | Diff: {report.balance_difference:+.2f} EUR")
        self.logger.info(f"Offene Orders geprüft: {report.open_orders_checked}")
        self.logger.info(f"Actions: {len(report.actions_taken)}")
        self.logger.info(f"Warnings: {len(report.warnings)}")
        self.logger.info(f"Errors: {len(report.errors)}")
        self.logger.info("=" * 60)

        for action in report.actions_taken:
            self.logger.info(f"  ✓ {action}")
        for warning in report.warnings:
            self.logger.warning(f"  ⚠ {warning}")
        for err in report.errors:
            self.logger.error(f"  ✗ {err}")

        self.logger.info("=" * 60)


# Hilfsfunktion für zukünftige Nutzung in main.py
async def run_startup_reconciliation(portfolio: Portfolio, live_engine: LiveOrderEngine) -> ReconciliationReport:
    """Convenience-Funktion für den Start-Flow."""
    reconciler = Reconciler(portfolio, live_engine)
    return await reconciler.reconcile_on_startup()
