"""
Order-Execution-Engine für Paper-Trading
Simuliert realistische Order-Ausführung mit Slippage, Fees und Latenz
"""

import asyncio
import random
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from enum import Enum


class OrderType(Enum):
    """Order-Typen"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order-Status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Eine Order"""
    id: str
    symbol: str
    side: str  # 'buy' oder 'sell'
    order_type: OrderType
    size: float
    price: Optional[float] = None  # Nur für Limit Orders
    stop_price: Optional[float] = None  # Für Stop Orders
    leverage: float = 1.0
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = None
    fill_timestamp: datetime = None
    strategy: str = ""
    market_type: str = "crypto"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExecutionResult:
    """Ergebnis einer Order-Ausführung"""
    order: Order
    success: bool
    message: str
    execution_price: float
    total_fees: float
    slippage_cost: float
    latency_ms: int


class OrderEngine:
    """
    Simulierte Order-Execution mit realistischen Marktbedingungen
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('OrderEngine')

        # Fee-Struktur
        self.fees = {
            'crypto_maker': self.config.get('crypto_maker', 0.0004),  # 0.04%
            'crypto_taker': self.config.get('crypto_taker', 0.0006),  # 0.06%
        }

        # Slippage-Parameter
        self.slippage_min = self.config.get('slippage_min', 0.0001)  # 0.01%
        self.slippage_max = self.config.get('slippage_max', 0.0005)  # 0.05%

        # Latenz-Simulation
        self.latency_min_ms = 50
        self.latency_max_ms = 200

        # Order-Counter für IDs
        self.order_counter = 0

        # Pending Orders
        self.pending_orders: Dict[str, Order] = {}

        # Callbacks
        self.on_fill: Optional[Callable] = None

    def _generate_order_id(self) -> str:
        """Generiert einzigartige Order-ID"""
        self.order_counter += 1
        return f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.order_counter:04d}"

    def _calculate_slippage(self, price: float, side: str, size: float) -> float:
        """
        Berechnet Slippage basierend auf Ordergröße und Marktbedingungen

        Größere Orders haben mehr Slippage
        """
        # Basis-Slippage
        base_slippage = random.uniform(self.slippage_min, self.slippage_max)

        # Größenabhängige Slippage (größere Orders = mehr Slippage)
        size_factor = min(1.5, 1.0 + (size / 10000) * 0.1)  # Max 50% mehr

        total_slippage = base_slippage * size_factor

        # Bei Kauf: Preis steigt, bei Verkauf: Preis fällt
        if side == 'buy':
            return price * (1 + total_slippage)
        else:
            return price * (1 - total_slippage)

    def _calculate_fees(self, size: float, price: float, order_type: OrderType,
                       market_type: str) -> float:
        """Berechnet Trading-Fees (size ist bereits in USD)"""
        notional = size  # size ist bereits in USD/Quote-Currency

        if order_type == OrderType.LIMIT:
            fee_rate = self.fees['crypto_maker']
        else:
            fee_rate = self.fees['crypto_taker']

        return notional * fee_rate

    async def _simulate_latency(self):
        """Simuliert Netzwerk-Latenz"""
        latency_ms = random.randint(self.latency_min_ms, self.latency_max_ms)
        await asyncio.sleep(latency_ms / 1000)
        return latency_ms

    def _should_partial_fill(self, size: float) -> tuple:
        """
        Entscheidet ob Order nur teilweise gefüllt wird

        Größere Orders haben höhere Chance auf Partial Fill
        """
        # Bei sehr kleinen Orders: immer voll füllen
        if size < 100:
            return False, size

        # Wahrscheinlichkeit für Partial Fill basierend auf Größe
        partial_prob = min(0.3, size / 50000)  # Max 30%

        if random.random() < partial_prob:
            # Teilfüllung: 50-90% der Order
            fill_ratio = random.uniform(0.5, 0.9)
            return True, size * fill_ratio

        return False, size

    async def execute_market_order(self, symbol: str, side: str, size: float,
                                  current_price: float, leverage: float = 1.0,
                                  strategy: str = "", market_type: str = "crypto") -> ExecutionResult:
        """
        Führt Market Order aus

        Args:
            symbol: Trading-Pair
            side: 'buy' oder 'sell'
            size: Ordergröße in Base-Currency
            current_price: Aktueller Marktpreis
            leverage: Hebel (nur für Futures)
            strategy: Name der Strategie
            market_type: 'crypto' oder 'polymarket'

        Returns:
            ExecutionResult mit allen Details
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=size,
            leverage=leverage,
            strategy=strategy,
            market_type=market_type
        )

        self.logger.info(f"Market Order erstellt: {order.id} - {side} {size} {symbol}")

        # Latenz simulieren
        latency_ms = await self._simulate_latency()

        # Slippage berechnen
        execution_price = self._calculate_slippage(current_price, side, size)
        slippage_cost = size * abs(execution_price - current_price) / current_price if current_price > 0 else 0

        # Partial Fill prüfen
        is_partial, filled_size = self._should_partial_fill(size)

        # Fees berechnen
        total_fees = self._calculate_fees(filled_size, execution_price, OrderType.MARKET, market_type)

        # Order aktualisieren
        order.filled_size = filled_size
        order.filled_price = execution_price
        order.fees = total_fees
        order.slippage = slippage_cost
        order.fill_timestamp = datetime.now()

        if is_partial:
            order.status = OrderStatus.PARTIAL
            message = f"Partial Fill: {filled_size}/{size}"
        else:
            order.status = OrderStatus.FILLED
            message = "Order vollständig ausgeführt"

        result = ExecutionResult(
            order=order,
            success=True,
            message=message,
            execution_price=execution_price,
            total_fees=total_fees,
            slippage_cost=slippage_cost,
            latency_ms=latency_ms
        )

        self.logger.info(f"Order {order.id} ausgeführt @ {execution_price:.4f} (Slippage: {slippage_cost:.4f})")

        # Callback aufrufen
        if self.on_fill:
            await self._call_on_fill(result)

        return result

    async def execute_limit_order(self, symbol: str, side: str, size: float,
                                 limit_price: float, leverage: float = 1.0,
                                 strategy: str = "", market_type: str = "crypto") -> Order:
        """
        Erstellt Limit Order (wird nicht sofort ausgeführt)

        Returns:
            Order-Objekt (pending)
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            size=size,
            price=limit_price,
            leverage=leverage,
            strategy=strategy,
            market_type=market_type,
            status=OrderStatus.PENDING
        )

        self.pending_orders[order.id] = order
        self.logger.info(f"Limit Order erstellt: {order.id} - {side} {size} {symbol} @ {limit_price}")

        return order

    async def check_pending_orders(self, current_prices: Dict[str, float]) -> list:
        """
        Prüft alle pending Orders gegen aktuelle Preise

        Returns:
            Liste der ausgeführten Orders
        """
        filled_orders = []

        for order_id, order in list(self.pending_orders.items()):
            if order.symbol not in current_prices:
                continue

            current_price = current_prices[order.symbol]

            # Limit Order Logik
            if order.order_type == OrderType.LIMIT:
                should_fill = False

                if order.side == 'buy' and current_price <= order.price:
                    should_fill = True
                elif order.side == 'sell' and current_price >= order.price:
                    should_fill = True

                if should_fill:
                    # Latenz simulieren
                    latency_ms = await self._simulate_latency()

                    # Bei Limit Orders: weniger Slippage, Maker Fees
                    execution_price = order.price  # Limit Preis garantiert
                    total_fees = self._calculate_fees(order.size, execution_price,
                                                     OrderType.LIMIT, order.market_type)

                    order.filled_size = order.size
                    order.filled_price = execution_price
                    order.fees = total_fees
                    order.status = OrderStatus.FILLED
                    order.fill_timestamp = datetime.now()

                    filled_orders.append(order)
                    del self.pending_orders[order_id]

                    self.logger.info(f"Limit Order {order_id} ausgeführt @ {execution_price}")

            # Stop Order Logik
            elif order.order_type == OrderType.STOP_MARKET:
                should_trigger = False

                if order.side == 'buy' and current_price >= order.stop_price:
                    should_trigger = True
                elif order.side == 'sell' and current_price <= order.stop_price:
                    should_trigger = True

                if should_trigger:
                    # Als Market Order ausführen
                    result = await self.execute_market_order(
                        order.symbol, order.side, order.size,
                        current_price, order.leverage, order.strategy, order.market_type
                    )
                    filled_orders.append(result.order)
                    del self.pending_orders[order_id]

        return filled_orders

    async def cancel_order(self, order_id: str) -> bool:
        """Storniert eine pending Order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            self.logger.info(f"Order {order_id} storniert")
            return True
        return False

    def cancel_all_orders(self, symbol: str = None) -> int:
        """
        Storniert alle pending Orders

        Args:
            symbol: Optional - nur Orders für dieses Symbol

        Returns:
            Anzahl stornierter Orders
        """
        cancelled = 0
        for order_id, order in list(self.pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                cancelled += 1

        self.logger.info(f"{cancelled} Orders storniert")
        return cancelled

    async def _call_on_fill(self, result: ExecutionResult):
        """Ruft Fill-Callback auf"""
        try:
            if asyncio.iscoroutinefunction(self.on_fill):
                await self.on_fill(result)
            else:
                self.on_fill(result)
        except Exception as e:
            self.logger.error(f"Fehler im on_fill Callback: {e}")

    def get_pending_orders(self, symbol: str = None) -> list:
        """Gibt alle pending Orders zurück"""
        if symbol:
            return [o for o in self.pending_orders.values() if o.symbol == symbol]
        return list(self.pending_orders.values())

    def get_order(self, order_id: str) -> Optional[Order]:
        """Gibt Order nach ID zurück"""
        return self.pending_orders.get(order_id)

    def set_fees(self, crypto_maker: float = None, crypto_taker: float = None):
        """Setzt Fee-Struktur"""
        if crypto_maker is not None:
            self.fees['crypto_maker'] = crypto_maker
        if crypto_taker is not None:
            self.fees['crypto_taker'] = crypto_taker

    def set_slippage(self, min_slippage: float, max_slippage: float):
        """Setzt Slippage-Parameter"""
        self.slippage_min = min_slippage
        self.slippage_max = max_slippage

    def get_stats(self) -> Dict:
        """Gibt Order-Statistiken zurück"""
        return {
            'pending_orders': len(self.pending_orders),
            'total_orders': self.order_counter,
            'fees': self.fees,
            'slippage_range': (self.slippage_min, self.slippage_max),
            'latency_range_ms': (self.latency_min_ms, self.latency_max_ms)
        }
