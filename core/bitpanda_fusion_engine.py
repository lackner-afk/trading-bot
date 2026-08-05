"""
Order-Execution auf Bitpanda Fusion.

Warum eigener REST-Client statt CCXT: CCXT unterstützt Fusion nicht
(Issue #25354, offen seit Feb 2025, kein PR). Der Fusion-MCP könnte
ebenfalls ausführen, für einen Dauerläufer-Bot ist die REST-API aber der
direktere Weg — kein Zusatzprozess, deterministisches Fehlerverhalten,
volle Kontrolle über Retries und Idempotenz.

Fusion ist Spot: kein Hebel, keine Shorts. Ein SELL setzt voraus, dass der
Base-Asset-Bestand vorhanden ist; die Spot-Beschränkungen erzwingt der Bot
bereits an vier Stellen oberhalb (siehe core/market_constraints.py).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from core.execution_base import BaseOrderEngine
from core.fusion_client import (
    DEAD_STATES,
    FILLED_STATES,
    ORDER_TYPE_STOP_MARKET,
    FusionAPIError,
    FusionClient,
    PairInfo,
)
from core.order_engine import ExecutionResult, Order, OrderStatus, OrderType
from data.symbols import to_venue

VENUE = "fusion"


class BitpandaFusionOrderEngine(BaseOrderEngine):
    """
    Echte Ausführung auf Bitpanda Fusion.

    Shadow Mode ist Default: die Engine loggt exakt, was sie tun würde,
    schickt aber keine Order los. Bewusst so herum — die URL-Pfade der API
    sind aus dem CLI abgeleitet und nicht gegen die (nicht abrufbare) Doku
    verifiziert. Erst `preflight()` grün, dann `shadow_mode: false`.
    """

    def __init__(self, api_key: str, config: Dict = None,
                 client: FusionClient = None, constraints=None):
        self.config = config or {}
        self.logger = logging.getLogger("BitpandaFusionEngine")

        from core.market_constraints import MarketConstraints
        self.constraints = constraints or MarketConstraints()

        self.client = client or FusionClient(
            api_key=api_key,
            host=self.config.get("host"),
            config=self.config,
        )

        self.shadow_mode = self.config.get("shadow_mode", True)
        if self.shadow_mode:
            self.logger.warning("!!! SHADOW MODE AKTIV - es werden KEINE echten Orders platziert !!!")

        self.fees = {
            "crypto_maker": self.config.get("crypto_maker", 0.0004),
            "crypto_taker": self.config.get("crypto_taker", 0.0006),
        }

        self.on_fill: Optional[callable] = None
        self.pending_orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._preflight_ok = False

        # symbol -> Venue-Order-ID des Schutz-Stops. Ohne dieses Tracking
        # bliebe der Stop nach einem lokalen Exit beim Venue liegen und
        # würde später erneut verkaufen — eine Geisterorder.
        self.protective_stops: Dict[str, str] = {}

    # ----- Vorbereitung ---------------------------------------------

    async def preflight(self) -> Dict:
        """
        Prüft Auth und Endpunkte gegen die echte API.

        Muss vor dem ersten scharfen Order-Versand grün sein.
        """
        report = await self.client.preflight()
        self._preflight_ok = bool(report.get("ok"))

        await self._check_fee_tier(report)

        if self._preflight_ok:
            self.logger.info("Fusion-Preflight OK")
        else:
            for err in report.get("errors", []):
                self.logger.critical(f"Fusion-Preflight: {err}")
        return report

    async def _check_fee_tier(self, report: Dict):
        """
        Gleicht die konfigurierte Gebühr gegen die echte Stufe des Kontos ab.

        Eine zu optimistische Annahme verfälscht Backtest, Paper-Ergebnisse
        und das Profitabilitäts-Gate gleichermaßen — und zwar in die
        gefährliche Richtung: die Strategie sieht besser aus als sie ist.
        """
        try:
            actual = await self.client.get_fee_tier()
        except FusionAPIError as e:
            report.setdefault("checks", {})["fee_tier"] = {"ok": False, "detail": str(e)}
            self.logger.warning(f"Gebuehrenstufe nicht abrufbar: {e}")
            return

        if actual is None:
            self.logger.warning(
                "Gebuehrenstufe nicht aus der Account-Antwort lesbar - "
                "konfigurierten Wert manuell gegen 'bp-fusion account info' pruefen"
            )
            return

        configured = self.fees["crypto_taker"]
        report.setdefault("checks", {})["fee_tier"] = {
            "ok": True, "detail": f"{actual:.4%} (Config: {configured:.4%})"
        }

        # Nur wenn die Realität teurer ist, ist es gefaehrlich
        if actual > configured * 1.05:
            self.logger.critical(
                f"GEBUEHREN ZU NIEDRIG KONFIGURIERT: Konto zahlt {actual:.3%}, "
                f"settings.yaml rechnet mit {configured:.3%}. Alle bisherigen "
                f"Backtest- und Paper-Ergebnisse sind zu optimistisch. "
                f"fees.crypto_maker/crypto_taker auf {actual:.4f} setzen."
            )
            report.setdefault("errors", []).append(
                f"Gebuehrenstufe weicht ab: real {actual:.3%} vs. Config {configured:.3%}"
            )
            report["ok"] = False
            self._preflight_ok = False
        elif actual < configured * 0.95:
            self.logger.info(
                f"Gebuehrenstufe guenstiger als konfiguriert: {actual:.3%} statt "
                f"{configured:.3%} - Ergebnisse sind konservativ gerechnet."
            )

    async def _pair_info(self, symbol: str) -> Optional[PairInfo]:
        """Handelsregeln für ein Bot-Symbol (BTC_EUR -> BTC-EUR)."""
        venue_symbol = to_venue(symbol, VENUE)
        try:
            pairs = await self.client.get_pairs()
        except FusionAPIError as e:
            self.logger.error(f"Handelspaare nicht abrufbar: {e}")
            return None
        return pairs.get(venue_symbol)

    def _next_client_order_id(self, prefix: str) -> str:
        """
        Eindeutige Client-Order-ID.

        Dient als Idempotenz-Anker: taucht dieselbe ID zweimal auf, soll das
        Venue die zweite Order verwerfen statt eine Doppelposition zu öffnen.
        """
        self._order_counter += 1
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}-{stamp}-{self._order_counter:04d}"

    def _estimate_fees(self, notional: float, taker: bool = True) -> float:
        rate = self.fees["crypto_taker"] if taker else self.fees["crypto_maker"]
        return abs(notional) * rate

    # ----- Ausführung -----------------------------------------------

    async def execute_market_order(self, symbol: str, side: str, size: float,
                                   current_price: float, leverage: float = 1.0,
                                   strategy: str = "",
                                   market_type: str = "crypto") -> ExecutionResult:
        """
        Market-Order. `size` ist EUR-Notional.

        Beim Kauf wird `amount` (Quote-Volumen) geschickt — Fusion unterstützt
        das direkt und es vermeidet den Rundungsfehler, der beim Umweg über
        die Base-Menge entsteht. Beim Verkauf muss die Base-Menge übergeben
        werden, weil nur die tatsächlich im Bestand liegt.
        """
        started = datetime.now()
        client_order_id = self._next_client_order_id("BP-MKT")
        order = Order(
            id=client_order_id, symbol=symbol, side=side,
            order_type=OrderType.MARKET, size=size, leverage=leverage,
            strategy=strategy, market_type=market_type, timestamp=started,
        )

        if self.constraints.spot_only and leverage > 1.0:
            return self._failure(order, f"Spot-Venue: Leverage {leverage}x nicht handelbar", started)

        if current_price <= 0:
            return self._failure(order, "Kein gueltiger Referenzpreis", started)

        info = await self._pair_info(symbol)
        amount = size / current_price

        if info is not None:
            amount = info.round_amount(amount)
            problem = info.validate(amount, size)
            if problem:
                return self._failure(order, problem, started)
        else:
            self.logger.warning(
                f"Keine Handelsregeln fuer {symbol} - Order ohne Praezisionspruefung"
            )

        venue_symbol = to_venue(symbol, VENUE)

        if self.shadow_mode:
            return self._shadow_fill(order, venue_symbol, amount, current_price, started)

        try:
            if side == "buy":
                response = await self.client.create_order(
                    pair=venue_symbol, side=side, order_type="market",
                    amount=round(size, 2), client_order_id=client_order_id,
                )
            else:
                response = await self.client.create_order(
                    pair=venue_symbol, side=side, order_type="market",
                    quantity=amount, client_order_id=client_order_id,
                )
        except FusionAPIError as e:
            self.logger.error(f"[FUSION] Market-Order fehlgeschlagen: {e}")
            return self._failure(order, str(e), started)

        return await self._result_from_response(order, response, current_price, started)

    async def execute_limit_order(self, symbol: str, side: str, size: float,
                                  limit_price: float, leverage: float = 1.0,
                                  strategy: str = "",
                                  market_type: str = "crypto") -> Order:
        """Limit-Order. Wird in pending_orders getrackt, bis sie füllt."""
        client_order_id = self._next_client_order_id("BP-LIM")
        order = Order(
            id=client_order_id, symbol=symbol, side=side,
            order_type=OrderType.LIMIT, size=size, price=limit_price,
            leverage=leverage, strategy=strategy, market_type=market_type,
        )

        if limit_price <= 0:
            order.status = OrderStatus.REJECTED
            return order

        info = await self._pair_info(symbol)
        amount = size / limit_price
        price = limit_price

        if info is not None:
            amount = info.round_amount(amount)
            price = info.round_price(limit_price)
            problem = info.validate(amount, size)
            if problem:
                self.logger.warning(f"[FUSION] Limit-Order abgelehnt: {problem}")
                order.status = OrderStatus.REJECTED
                return order

        venue_symbol = to_venue(symbol, VENUE)

        if self.shadow_mode:
            self.logger.warning(
                f"[SHADOW] Wuerde Limit-Order platzieren: {side} {amount} {venue_symbol} @ {price}"
            )
            self.pending_orders[order.id] = order
            return order

        try:
            response = await self.client.create_order(
                pair=venue_symbol, side=side, order_type="limit",
                quantity=amount, limit_price=price, client_order_id=client_order_id,
            )
        except FusionAPIError as e:
            self.logger.error(f"[FUSION] Limit-Order fehlgeschlagen: {e}")
            order.status = OrderStatus.REJECTED
            return order

        order.external_id = self._extract_id(response) or client_order_id
        self.pending_orders[order.external_id] = order
        self.logger.info(f"[FUSION] Limit-Order {order.external_id} @ {price}")
        return order

    async def place_protective_stop(self, symbol: str, side: str, quantity: float,
                                    stop_price: float) -> Optional[str]:
        """
        Platziert einen Stop-Market beim Venue als Sicherheitsnetz.

        Der Bot prüft Stop-Loss und Take-Profit im 1-Sekunden-Loop. Stirbt der
        Prozess — Crash, VPS-Neustart, Netzausfall —, ist eine offene Position
        ohne diesen Stop **völlig ungeschützt**. Der lokale Trailing-Stop
        bleibt die erste Ebene; dieser hier fängt den Ausfall ab.

        `side` ist die Richtung der SCHUTZ-Order, nicht der Position: eine
        Long-Position wird mit einem Sell-Stop abgesichert.

        Returns die Venue-Order-ID oder None.
        """
        if stop_price <= 0 or quantity <= 0:
            return None

        info = await self._pair_info(symbol)
        if info is not None:
            quantity = info.round_amount(quantity)
            stop_price = info.round_price(stop_price)

        venue_symbol = to_venue(symbol, VENUE)
        client_order_id = self._next_client_order_id("BP-STP")

        if self.shadow_mode:
            self.logger.warning(
                f"[SHADOW] Wuerde Schutz-Stop setzen: {side} {quantity} "
                f"{venue_symbol} @ Stop {stop_price}"
            )
            return None

        try:
            response = await self.client.create_order(
                pair=venue_symbol, side=side, order_type=ORDER_TYPE_STOP_MARKET,
                quantity=quantity, stop_price=stop_price,
                client_order_id=client_order_id,
            )
        except FusionAPIError as e:
            # Kein harter Fehler: die Position ist offen und der lokale Stop
            # greift weiterhin. Aber es muss auffallen.
            self.logger.critical(
                f"SCHUTZ-STOP FEHLGESCHLAGEN fuer {symbol}: {e}. Position ist "
                f"nur noch durch den laufenden Bot-Prozess abgesichert."
            )
            return None

        order_id = self._extract_id(response) or client_order_id
        self.protective_stops[symbol] = order_id
        self.logger.info(
            f"[FUSION] Schutz-Stop {order_id} fuer {symbol} @ {stop_price}"
        )
        return order_id

    async def cancel_protective_stop(self, symbol: str) -> bool:
        """
        Storniert den Schutz-Stop eines Symbols.

        Muss bei JEDEM lokalen Exit laufen. Bleibt der Stop stehen, verkauft
        das Venue später ein zweites Mal — bei Spot heisst das, dass Bestand
        verkauft wird, den der Bot gar nicht mehr als Position führt.
        """
        order_id = self.protective_stops.pop(symbol, None)
        if not order_id or self.shadow_mode:
            return True

        try:
            await self.client.cancel_order(order_id)
            self.logger.info(f"[FUSION] Schutz-Stop {order_id} fuer {symbol} storniert")
            return True
        except FusionAPIError as e:
            self.logger.critical(
                f"SCHUTZ-STOP {order_id} ({symbol}) konnte nicht storniert werden: {e}. "
                f"GEISTERORDER - bitte manuell im Fusion-Konto pruefen!"
            )
            return False

    async def check_protective_stops(self) -> List[str]:
        """
        Prüft, ob ein Schutz-Stop beim Venue ausgelöst hat.

        Wenn ja, ist die Position dort bereits geschlossen — der Bot muss das
        lokal nachvollziehen, sonst hält er eine Position für offen, die es
        nicht mehr gibt, und scheitert später beim Exit.

        Returns die Symbole, deren Stop ausgelöst hat.
        """
        if not self.protective_stops or self.shadow_mode:
            return []

        ausgeloest: List[str] = []

        for symbol, order_id in list(self.protective_stops.items()):
            try:
                response = await self.client.get_order(order_id)
            except FusionAPIError as e:
                self.logger.error(f"[FUSION] Schutz-Stop {order_id} nicht abrufbar: {e}")
                continue

            status = str(self._get(response, "status", "")).lower()

            if status in FILLED_STATES:
                self.protective_stops.pop(symbol, None)
                ausgeloest.append(symbol)
                self.logger.critical(
                    f"[FUSION] Schutz-Stop fuer {symbol} hat AUSGELOEST @ "
                    f"{self._filled_price(response)} - Position ist beim Venue geschlossen"
                )
            elif status in DEAD_STATES:
                self.protective_stops.pop(symbol, None)
                self.logger.warning(
                    f"[FUSION] Schutz-Stop fuer {symbol} beendet ({status}) - "
                    f"Position ist NICHT mehr venue-seitig abgesichert"
                )

        return ausgeloest

    async def check_pending_orders(self, current_prices: Dict[str, float]) -> List[Order]:
        """Fragt den Status der getrackten Orders ab und meldet neue Fills."""
        if not self.pending_orders or self.shadow_mode:
            return []

        filled: List[Order] = []

        for order_id, order in list(self.pending_orders.items()):
            try:
                response = await self.client.get_order(order_id)
            except FusionAPIError as e:
                self.logger.error(f"[FUSION] Status von Order {order_id} nicht abrufbar: {e}")
                continue

            status = str(self._get(response, "status", "")).lower()

            if status in FILLED_STATES:
                price = self._filled_price(response) or order.price or 0.0
                order.status = OrderStatus.FILLED
                order.filled_price = price
                order.filled_size = self._filled_amount(response)
                order.fees = self._response_fees(response, order.size)
                order.fill_timestamp = datetime.now()

                del self.pending_orders[order_id]
                filled.append(order)

                self.logger.info(f"[FUSION] Order {order_id} gefuellt @ {price}")
                await self._call_on_fill(ExecutionResult(
                    order=order, success=True, message="Filled",
                    execution_price=price, total_fees=order.fees,
                    slippage_cost=0.0, latency_ms=0,
                ))

            elif status in DEAD_STATES:
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                self.logger.warning(f"[FUSION] Order {order_id} beendet: {status}")

        return filled

    async def cancel_order(self, order_id: str) -> bool:
        if self.shadow_mode:
            self.pending_orders.pop(order_id, None)
            return True
        try:
            await self.client.cancel_order(order_id)
        except FusionAPIError as e:
            self.logger.error(f"[FUSION] Cancel fehlgeschlagen fuer {order_id}: {e}")
            return False
        self.pending_orders.pop(order_id, None)
        return True

    async def cancel_all_orders(self, symbol: str = None) -> int:
        targets = [oid for oid, o in self.pending_orders.items()
                   if symbol is None or o.symbol == symbol]
        cancelled = 0
        for order_id in targets:
            if await self.cancel_order(order_id):
                cancelled += 1
        if cancelled:
            self.logger.info(f"[FUSION] {cancelled} Order(s) storniert")
        return cancelled

    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        if symbol:
            return [o for o in self.pending_orders.values() if o.symbol == symbol]
        return list(self.pending_orders.values())

    async def close(self):
        await self.client.close()

    # ----- Konto ----------------------------------------------------

    async def fetch_balance(self) -> Dict[str, float]:
        """Bestände je Asset. Im Spot-Modell sind das zugleich die Positionen."""
        return await self.client.get_balances()

    async def fetch_open_orders(self, symbol: str = None) -> List[Dict]:
        pair = to_venue(symbol, VENUE) if symbol else None
        return await self.client.list_orders(pair=pair, status="open")

    def set_fees(self, crypto_maker: float = None, crypto_taker: float = None):
        if crypto_maker is not None:
            self.fees["crypto_maker"] = crypto_maker
        if crypto_taker is not None:
            self.fees["crypto_taker"] = crypto_taker

    # ----- Hilfen ---------------------------------------------------

    def _failure(self, order: Order, message: str, started: datetime) -> ExecutionResult:
        order.status = OrderStatus.REJECTED
        return ExecutionResult(
            order=order, success=False, message=message,
            execution_price=0.0, total_fees=0.0, slippage_cost=0.0,
            latency_ms=self._elapsed_ms(started),
        )

    def _shadow_fill(self, order: Order, venue_symbol: str, amount: float,
                     price: float, started: datetime) -> ExecutionResult:
        """Simuliert einen Fill mit minimalem Slippage, ohne Netzwerkverkehr."""
        self.logger.warning(
            f"[SHADOW] Wuerde Market-Order ausfuehren: {order.side} {amount} "
            f"{venue_symbol} @ ~{price:.2f} (~{order.size:.2f} EUR)"
        )
        fill_price = price * (1.0001 if order.side == "buy" else 0.9999)
        fees = self._estimate_fees(order.size)

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_size = amount
        order.fees = fees
        order.fill_timestamp = datetime.now()

        return ExecutionResult(
            order=order, success=True, message="Shadow fill",
            execution_price=fill_price, total_fees=fees,
            slippage_cost=abs(fill_price - price) * amount,
            latency_ms=self._elapsed_ms(started),
        )

    async def _result_from_response(self, order: Order, response, reference_price: float,
                                    started: datetime) -> ExecutionResult:
        """Übersetzt die API-Antwort in ein ExecutionResult."""
        status = str(self._get(response, "status", "")).lower()
        fill_price = self._filled_price(response) or reference_price
        fill_amount = self._filled_amount(response) or (order.size / reference_price)
        fees = self._response_fees(response, order.size)

        order.external_id = self._extract_id(response) or order.id
        order.filled_price = fill_price
        order.filled_size = fill_amount
        order.fees = fees
        order.fill_timestamp = datetime.now()

        if status in DEAD_STATES:
            return self._failure(order, f"Order abgelehnt: {status}", started)

        if status in FILLED_STATES or not status:
            order.status = OrderStatus.FILLED
        else:
            # Noch offen — weiter beobachten statt als gefüllt zu buchen
            order.status = OrderStatus.PENDING
            self.pending_orders[order.external_id] = order

        slippage = abs(fill_price - reference_price) * fill_amount
        self.logger.info(
            f"[EXEC-QUALITY] {order.symbol} {order.side} Referenz={reference_price:.2f} "
            f"Fill={fill_price:.2f} "
            f"Slippage={((fill_price - reference_price) / reference_price * 100):+.3f}% "
            f"Latenz={self._elapsed_ms(started)}ms"
        )

        result = ExecutionResult(
            order=order,
            success=order.status == OrderStatus.FILLED,
            message=f"Fusion order {order.external_id} ({status or 'filled'})",
            execution_price=fill_price, total_fees=fees,
            slippage_cost=slippage, latency_ms=self._elapsed_ms(started),
        )

        if result.success:
            await self._call_on_fill(result)
        return result

    @staticmethod
    def _elapsed_ms(started: datetime) -> int:
        return int((datetime.now() - started).total_seconds() * 1000)

    @staticmethod
    def _get(response, key: str, default=None):
        return response.get(key, default) if isinstance(response, dict) else default

    @classmethod
    def _extract_id(cls, response) -> Optional[str]:
        for key in ("order_id", "id", "orderId"):
            value = cls._get(response, key)
            if value:
                return str(value)
        return None

    @classmethod
    def _filled_price(cls, response) -> float:
        for key in ("average_price", "avg_price", "average", "price", "filled_price"):
            value = cls._get(response, key)
            if value is not None:
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    return price
        return 0.0

    @classmethod
    def _filled_amount(cls, response) -> float:
        for key in ("filled_amount", "filled", "executed_quantity", "quantity"):
            value = cls._get(response, key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _response_fees(self, response, notional: float) -> float:
        """Echte Gebühren aus der Antwort, sonst geschätzt."""
        for key in ("fee", "fee_amount", "fees"):
            value = self._get(response, key)
            if isinstance(value, dict):
                value = value.get("amount") or value.get("cost")
            if value is not None:
                try:
                    return abs(float(value))
                except (TypeError, ValueError):
                    continue
        return self._estimate_fees(notional)
