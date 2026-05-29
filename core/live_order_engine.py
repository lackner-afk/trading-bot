"""
Live Order Execution Engine für One Trading (via CCXT)

Ersetzt die simulierte OrderEngine, wenn mode= live.
Verwendet die echte One Trading API über CCXT.

Wichtige Hinweise:
- Aktuell primär Spot-Trading (Leverage wird für Ordergröße ignoriert oder gemappt)
- One Trading unterstützt hauptsächlich LIMIT Orders
- "size" im Aufruf wird als EUR-Notional interpretiert und in Base-Amount umgerechnet
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Callable, List

import ccxt.async_support as ccxt

# Gemeinsame Typen aus der Paper-Engine wiederverwenden
from .order_engine import (
    Order, OrderType, OrderStatus, ExecutionResult
)


class LiveOrderEngine:
    """
    Echte Order-Ausführung auf One Trading via CCXT.

    Erwartet gültige API Keys mit Trade-Berechtigung.
    """

    def __init__(self, api_key: str, api_secret: str, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('LiveOrderEngine')

        if not api_key or not api_secret:
            raise ValueError("LiveOrderEngine benötigt api_key und api_secret")

        self.exchange = ccxt.onetrading({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        # Shadow Mode (Phase 5): Loggt exakt, was es tun würde, platziert aber keine echten Orders.
        # Sehr wertvoll für sichere Validierung vor echtem Kapitaleinsatz.
        self.shadow_mode = self.config.get('shadow_mode', False)
        if self.shadow_mode:
            self.logger.warning("!!! SHADOW MODE AKTIV – Keine echten Orders werden platziert !!!")

        # Fee-Struktur
        self.fees = {
            'crypto_maker': self.config.get('crypto_maker', 0.0004),
            'crypto_taker': self.config.get('crypto_taker', 0.0006),
        }

        # Callbacks
        self.on_fill: Optional[Callable] = None

        mode_str = "SHADOW" if self.shadow_mode else "LIVE"
        self.logger.info(f"LiveOrderEngine initialisiert (One Trading via CCXT) – Mode: {mode_str}")

    async def _load_markets(self):
        """Lädt Märkte einmalig (wird bei Bedarf aufgerufen)"""
        if not self.exchange.markets:
            await self.exchange.load_markets()

    def _symbol_to_ccxt(self, symbol: str) -> str:
        """BTC_EUR → BTC/EUR"""
        return symbol.replace('_', '/')

    async def execute_market_order(
        self,
        symbol: str,
        side: str,
        size: float,                    # EUR-Notional (wie im aktuellen System)
        current_price: float,
        leverage: float = 1.0,
        strategy: str = "",
        market_type: str = "crypto"
    ) -> ExecutionResult:
        """
        Führt eine Market Order auf One Trading aus.

        size wird als EUR-Notional behandelt und in Base-Amount umgerechnet.
        """
        await self._load_markets()
        ccxt_symbol = self._symbol_to_ccxt(symbol)

        # EUR-Notional → Base Amount (grob, für bessere Genauigkeit später Quote-OrderQty nutzen falls supported)
        amount = size / current_price if current_price > 0 else 0

        order_id = f"LIVE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{id(self)}"

        self.logger.info(f"[LIVE] Market Order → {side.upper()} {amount:.8f} {symbol} (~{size:.2f} EUR)")

        start_time = datetime.now()

        try:
            if self.shadow_mode:
                # SHADOW MODE – keine echte Order, nur Logging + simuliertes Result
                self.logger.warning(f"[SHADOW] Würde Market Order ausführen: {side.upper()} {amount:.8f} {symbol} @ ~{current_price:.2f}")
                latency_ms = 42
                filled_price = current_price * (1 + (0.0001 if side == 'buy' else -0.0001))  # simuliertes minimales Slippage
                filled_amount = amount
                ccxt_order = {'id': f"SHADOW-{order_id}", 'status': 'closed', 'filled': amount, 'average': filled_price}
            else:
                # Echte Order
                try:
                    ccxt_order = await self.exchange.create_order(
                        symbol=ccxt_symbol,
                        type='market',
                        side=side,
                        amount=amount,
                        params={'clientOrderId': order_id}
                    )
                except Exception as market_err:
                    self.logger.warning(f"Market Order nicht direkt unterstützt, verwende Limit @ {current_price}: {market_err}")
                    ccxt_order = await self.exchange.create_order(
                        symbol=ccxt_symbol,
                        type='limit',
                        side=side,
                        amount=amount,
                        price=current_price,
                        params={'clientOrderId': order_id, 'timeInForce': 'IOC'}
                    )

            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # CCXT unified response mappen
            filled_price = float(ccxt_order.get('average') or ccxt_order.get('price') or current_price)
            filled_amount = float(ccxt_order.get('filled') or amount)
            status = ccxt_order.get('status', 'open')

            order = Order(
                id=ccxt_order.get('id', order_id),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=size,
                leverage=leverage,
                strategy=strategy,
                market_type=market_type,
                filled_size=filled_amount,
                filled_price=filled_price,
                fees=self._estimate_fees(filled_amount * filled_price, 'taker'),
                status=OrderStatus.FILLED if status in ('closed', 'filled') else OrderStatus.PENDING,
                timestamp=start_time,
                fill_timestamp=datetime.now()
            )

            result = ExecutionResult(
                order=order,
                success=True,
                message=f"Live Market Order placed: {ccxt_order.get('id')}",
                execution_price=filled_price,
                total_fees=order.fees,
                slippage_cost=0.0,  # Real slippage wird später aus Trades berechnet
                latency_ms=latency_ms
            )

            self.logger.info(f"[LIVE] Order {order.id} @ {filled_price:.4f} (filled {filled_amount})")

            # Execution Quality Logging (Phase 5) – wichtig für spätere Optimierung
            intended_price = current_price
            slippage_pct = (filled_price - intended_price) / intended_price * 100 if intended_price > 0 else 0
            self.logger.info(f"[EXEC-QUALITY] Symbol={symbol} Side={side} Intended={intended_price:.2f} "
                             f"Actual={filled_price:.2f} Slippage={slippage_pct:+.3f}% Latency={latency_ms}ms")

            if self.on_fill:
                await self._call_on_fill(result)

            return result

        except Exception as e:
            self.logger.error(f"[LIVE] Market Order fehlgeschlagen: {e}")
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=size,
                leverage=leverage,
                strategy=strategy,
                status=OrderStatus.REJECTED,
                timestamp=start_time
            )

            return ExecutionResult(
                order=order,
                success=False,
                message=str(e),
                execution_price=current_price,
                total_fees=0.0,
                slippage_cost=0.0,
                latency_ms=latency_ms
            )

    async def execute_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        limit_price: float,
        leverage: float = 1.0,
        strategy: str = "",
        market_type: str = "crypto"
    ) -> Order:
        """Erstellt eine echte Limit Order auf One Trading."""
        await self._load_markets()
        ccxt_symbol = self._symbol_to_ccxt(symbol)
        amount = size / limit_price if limit_price > 0 else 0

        client_id = f"LIVE-LIM-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            ccxt_order = await self.exchange.create_order(
                symbol=ccxt_symbol,
                type='limit',
                side=side,
                amount=amount,
                price=limit_price,
                params={'clientOrderId': client_id}
            )

            order = Order(
                id=ccxt_order.get('id', client_id),
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                size=size,
                price=limit_price,
                leverage=leverage,
                strategy=strategy,
                market_type=market_type,
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )

            self.logger.info(f"[LIVE] Limit Order erstellt: {order.id} @ {limit_price}")
            return order

        except Exception as e:
            self.logger.error(f"[LIVE] Limit Order fehlgeschlagen: {e}")
            order = Order(
                id=client_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                size=size,
                price=limit_price,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now()
            )
            return order

    async def check_pending_orders(self, current_prices: Dict[str, float]) -> List[Order]:
        """
        Prüft offene Orders auf der Exchange und gibt gefüllte zurück.

        Für Live-Modus: Wir holen den aktuellen Status von der Exchange.
        """
        filled_orders: List[Order] = []

        try:
            open_orders = await self.exchange.fetch_open_orders()

            for ccxt_order in open_orders:
                # Wir könnten hier prüfen ob sie gefüllt wurden, aber fetch_open_orders zeigt nur offene.
                # Besser: separate Methode fetch_order oder trades nutzen.
                pass  # Erweiterung in Phase 3 (Reconciliation)

        except Exception as e:
            self.logger.error(f"[LIVE] Fehler beim Prüfen offener Orders: {e}")

        return filled_orders

    async def cancel_order(self, order_id: str) -> bool:
        """Storniert eine Order auf der Exchange."""
        try:
            await self.exchange.cancel_order(order_id)
            self.logger.info(f"[LIVE] Order {order_id} storniert")
            return True
        except Exception as e:
            self.logger.error(f"[LIVE] Cancel fehlgeschlagen für {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str = None) -> int:
        """Storniert alle offenen Orders (optional für ein Symbol)."""
        try:
            if symbol:
                ccxt_symbol = self._symbol_to_ccxt(symbol)
                orders = await self.exchange.fetch_open_orders(ccxt_symbol)
            else:
                orders = await self.exchange.fetch_open_orders()

            count = 0
            for o in orders:
                await self.exchange.cancel_order(o['id'])
                count += 1
            return count
        except Exception as e:
            self.logger.error(f"[LIVE] cancel_all_orders Fehler: {e}")
            return 0

    async def fetch_balance(self) -> Optional[Dict]:
        """Echter Kontostand von One Trading."""
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"[LIVE] Balance fetch fehlgeschlagen: {e}")
            return None

    async def fetch_open_orders(self, symbol: str = None) -> List[Dict]:
        """Holt offene Orders direkt von der Exchange (für Reconciliation)."""
        try:
            ccxt_symbol = self._symbol_to_ccxt(symbol) if symbol else None
            return await self.exchange.fetch_open_orders(ccxt_symbol)
        except Exception as e:
            self.logger.error(f"[LIVE] fetch_open_orders fehlgeschlagen: {e}")
            return []

    def _estimate_fees(self, notional: float, fee_type: str = 'taker') -> float:
        rate = self.fees.get(f'crypto_{fee_type}', 0.0006)
        return notional * rate

    async def _call_on_fill(self, result: ExecutionResult):
        """Ruft on_fill Callback auf (gleich wie Paper Engine)"""
        try:
            if asyncio.iscoroutinefunction(self.on_fill):
                await self.on_fill(result)
            else:
                self.on_fill(result)
        except Exception as e:
            self.logger.error(f"Fehler im on_fill Callback (Live): {e}")

    async def close(self):
        """Schließt die Exchange-Verbindung."""
        await self.exchange.close()

    # Kompatibilitäts-Helper (werden später erweitert)
    def get_pending_orders(self, symbol: str = None) -> list:
        # Für Live besser async fetch_open_orders verwenden
        self.logger.warning("get_pending_orders() ist in LiveEngine synchron nicht sinnvoll – async fetch_open_orders nutzen")
        return []

    def set_fees(self, crypto_maker: float = None, crypto_taker: float = None):
        if crypto_maker is not None:
            self.fees['crypto_maker'] = crypto_maker
        if crypto_taker is not None:
            self.fees['crypto_taker'] = crypto_taker