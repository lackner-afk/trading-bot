"""
Gemeinsames Interface für alle Order-Engines.

Bisher gab es kein formales Interface: `OrderEngine` (Paper) und
`LiveOrderEngine` (One Trading) hatten parallel gepflegte Signaturen, die
teils auseinanderliefen — `cancel_all_orders` war in der einen sync und in
der anderen async, `close()` existierte nur live. Mit einer neuen dritten
Engine (Bitpanda Fusion) wird das unhaltbar.

Diese Basisklasse hält die Schnittmenge fest, die `main.py` tatsächlich
nutzt. Sie erzwingt bewusst nur wenig: die bestehenden Engines sollen ohne
Umbau darunter passen.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

# Nur für Typprüfung importieren: core.order_engine definiert Order und
# ExecutionResult und erbt seinerseits von dieser Basisklasse — ein
# Laufzeit-Import wäre zirkulär.
if TYPE_CHECKING:  # pragma: no cover
    from core.order_engine import ExecutionResult, Order


class BaseOrderEngine(ABC):
    """
    Was jede Order-Engine können muss.

    Die `size`-Semantik ist durchgehend **Quote-Notional in EUR**, nicht
    die Base-Menge. Engines, die dem Venue eine Base-Menge schicken müssen,
    rechnen intern um (`amount = size / price`).
    """

    # Wird von main.py gesetzt, um Fills zu verarbeiten
    on_fill: Optional[Callable] = None

    @abstractmethod
    async def execute_market_order(self, symbol: str, side: str, size: float,
                                   current_price: float, leverage: float = 1.0,
                                   strategy: str = "",
                                   market_type: str = "crypto") -> "ExecutionResult":
        """Führt eine Market-Order aus. `size` ist EUR-Notional."""

    @abstractmethod
    async def execute_limit_order(self, symbol: str, side: str, size: float,
                                  limit_price: float, leverage: float = 1.0,
                                  strategy: str = "",
                                  market_type: str = "crypto") -> "Order":
        """Platziert eine Limit-Order. `size` ist EUR-Notional."""

    @abstractmethod
    async def check_pending_orders(self, current_prices: Dict[str, float]) -> List["Order"]:
        """
        Prüft offene Orders und gibt die neu gefüllten zurück.

        Wird aus dem 1s-Main-Loop aufgerufen. Eine Implementierung, die hier
        nichts tut, verarbeitet nie einen Limit-Fill — genau dieser Fehler
        steckte in der Live-Engine.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Storniert eine einzelne Order."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str = None) -> int:
        """
        Storniert alle offenen Orders und gibt deren Anzahl zurück.

        async in allen Implementierungen — die Paper-Variante war früher
        sync, was jeden gemeinsamen Aufrufer gebrochen hätte.
        """

    @abstractmethod
    def get_pending_orders(self, symbol: str = None) -> List["Order"]:
        """Die lokal getrackten offenen Orders."""

    async def close(self):
        """
        Baut Netzwerkverbindungen ab. Default no-op, damit `main.py::stop`
        einheitlich aufrufen kann, ohne die Engine-Art zu kennen.
        """
        return None

    async def _call_on_fill(self, result: "ExecutionResult"):
        """Ruft den Fill-Callback auf, sync wie async."""
        import asyncio
        if self.on_fill is None:
            return
        try:
            if asyncio.iscoroutinefunction(self.on_fill):
                await self.on_fill(result)
            else:
                self.on_fill(result)
        except Exception:  # pragma: no cover - Callback-Fehler dürfen nie durchschlagen
            import logging
            logging.getLogger(self.__class__.__name__).exception("Fehler im on_fill Callback")
