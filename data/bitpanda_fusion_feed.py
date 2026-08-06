"""
Marktdaten-Feed für Bitpanda Fusion.

Fusion liefert eigene OHLCV-Daten (Intervalle 1m, 5m, 10m, 15m, 30m, 1h, 4h,
1d — belegt durch das offizielle CLI). Der Bot muss seine Kurse also nicht
von einem Fremd-Venue beziehen; das vermeidet ein Preisbasis-Risiko, bei dem
Signale auf Kraken-Kursen entstünden, während die Ausführung auf Fusion läuft.

Die öffentliche Oberfläche ist bewusst identisch zu KrakenFeed und
OneTradingCCXTFeed, damit main.py den Feed austauschen kann, ohne dass eine
Strategie davon etwas merkt.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.fusion_client import FusionAPIError, FusionClient
from data.crypto_feed import CandleData, MarketData
from data.indicators import calculate_indicators, ohlcv_to_df
from data.symbols import SymbolRegistry

VENUE = "fusion"


class BitpandaFusionFeed:
    """
    Polling-Feed gegen die Fusion REST-API.

    Preise alle 5s über den Tickers-Endpoint, Kerzen alle 60s je Timeframe —
    dasselbe Muster wie die bestehenden Feeds.
    """

    DEFAULT_PAIRS = ['BTC_EUR', 'ETH_EUR', 'SOL_EUR']
    # Fusion kennt zusätzlich 10m, 30m und 4h; der Bot nutzt diese vier.
    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    def __init__(self, api_key: str = None, api_secret: str = None,
                 config: Dict = None, client: FusionClient = None):
        self.config = config or {}
        self.logger = logging.getLogger('BitpandaFusionFeed')

        self.pairs = self.config.get('pairs') or list(self.DEFAULT_PAIRS)
        # Quote-Alias: handelt der Bot auf einem getrennten Kapitaltopf
        # (z.B. EURCV), holt der Feed die Kurse derselben Paare.
        self.quote_asset = (self.config.get('quote_asset') or 'EUR').upper()
        self.symbols = SymbolRegistry(self.pairs, VENUE, quote_alias=self.quote_asset)

        self.client = client or FusionClient(
            api_key=api_key,
            host=self.config.get('host'),
            config=self.config,
        )

        self.price_interval = self.config.get('price_interval_seconds', 5)
        self.candle_interval = self.config.get('candle_interval_seconds', 60)
        self.candle_limit = self.config.get('candle_limit', 200)

        # Zustand (Attribute werden extern gelesen — Interface-Kompatibilität)
        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.highest_prices: Dict[str, float] = {}
        self.last_update: Optional[datetime] = None
        self.running = False
        self._ws_connected = False      # REST-Polling, kein WebSocket

        self.on_price_update = None
        self.on_candle_close = None

        self._tasks: List[asyncio.Task] = []

    # ----- Lifecycle ------------------------------------------------

    async def start(self):
        """Lädt initiale Kerzen und startet die Polling-Loops."""
        self.running = True
        self.logger.info(f"Fusion-Feed startet fuer {', '.join(self.pairs)}")

        try:
            await self.client.get_pairs()
        except FusionAPIError as e:
            self.logger.error(f"Handelspaare nicht abrufbar: {e}")

        await self._load_initial_candles()

        self._tasks = [
            asyncio.create_task(self._price_loop()),
            asyncio.create_task(self._candle_loop()),
        ]

    async def stop(self):
        self.running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        await self.client.close()
        self.logger.info("Fusion-Feed gestoppt")

    async def _load_initial_candles(self):
        for symbol in self.pairs:
            self.candle_history.setdefault(symbol, {})
            for tf in self.TIMEFRAMES:
                df = await self._fetch_candles(symbol, tf)
                if df is not None:
                    self.candle_history[symbol][tf] = df
        self.logger.info("Initiale Kerzen geladen")

    async def _fetch_candles(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            raw = await self.client.get_candles(
                self.symbols.to_venue(symbol), interval=timeframe, limit=self.candle_limit
            )
        except FusionAPIError as e:
            self.logger.debug(f"Kerzen {symbol} {timeframe}: {e}")
            return None

        if not raw:
            return None
        return calculate_indicators(ohlcv_to_df(raw))

    # ----- Loops ----------------------------------------------------

    async def _price_loop(self):
        while self.running:
            try:
                prices = await self.client.get_tickers(self.symbols.venue_symbols())

                for venue_symbol, price in prices.items():
                    symbol = self.symbols.from_venue(venue_symbol)
                    existing = self.market_data.get(symbol)

                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=price,
                        bid=price,
                        ask=price,
                        volume_24h=existing.volume_24h if existing else 0.0,
                        change_24h=existing.change_24h if existing else 0.0,
                        timestamp=datetime.now(),
                    )

                    if price > self.highest_prices.get(symbol, 0.0):
                        self.highest_prices[symbol] = price

                    if self.on_price_update:
                        await self._notify(self.on_price_update, symbol, price)

                if prices:
                    self.last_update = datetime.now()
                    self._ws_connected = True

            except FusionAPIError as e:
                self._ws_connected = False
                self.logger.error(f"Preis-Loop: {e}")
            except Exception as e:
                self._ws_connected = False
                self.logger.error(f"Preis-Loop unerwarteter Fehler: {e}")

            await asyncio.sleep(self.price_interval)

    async def _candle_loop(self):
        while self.running:
            await asyncio.sleep(self.candle_interval)
            if not self.running:
                break
            try:
                for symbol in self.pairs:
                    for tf in self.TIMEFRAMES:
                        df = await self._fetch_candles(symbol, tf)
                        if df is not None:
                            self.candle_history.setdefault(symbol, {})[tf] = df
                            if self.on_candle_close:
                                await self._notify(self.on_candle_close, symbol, tf)
            except Exception as e:
                self.logger.error(f"Kerzen-Loop: {e}")

    @staticmethod
    async def _notify(callback, *args):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception:
            logging.getLogger('BitpandaFusionFeed').exception("Callback-Fehler")

    # ----- Public Interface (identisch zu KrakenFeed) ----------------

    def get_price(self, symbol: str) -> Optional[float]:
        md = self.market_data.get(symbol)
        return md.price if md else None

    def get_prices(self) -> Dict[str, float]:
        return {s: md.price for s, md in self.market_data.items()}

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        return self.market_data.get(symbol)

    def get_candles(self, symbol: str, timeframe: str = '1m',
                    n: int = 50) -> Optional[pd.DataFrame]:
        df = self.candle_history.get(symbol, {}).get(timeframe)
        if df is None or df.empty:
            return None
        return df.tail(n)

    def get_latest_candle(self, symbol: str, timeframe: str = '1m') -> Optional[CandleData]:
        df = self.candle_history.get(symbol, {}).get(timeframe)
        if df is None or df.empty:
            return None

        row = df.iloc[-1]

        def value(name):
            raw = row.get(name)
            return float(raw) if raw is not None and raw == raw else None

        return CandleData(
            timestamp=row['timestamp'],
            open=float(row['open']), high=float(row['high']),
            low=float(row['low']), close=float(row['close']),
            volume=float(row['volume']),
            rsi=value('rsi'),
            bb_upper=value('bb_upper'), bb_middle=value('bb_middle'),
            bb_lower=value('bb_lower'),
            ema_9=value('ema_9'), ema_21=value('ema_21'),
            vwap=value('vwap'), volume_delta=value('volume_delta'),
        )

    def get_rsi(self, symbol: str, timeframe: str = '1m') -> Optional[float]:
        df = self.candle_history.get(symbol, {}).get(timeframe)
        if df is None or df.empty or 'rsi' not in df.columns:
            return None
        value = df['rsi'].iloc[-1]
        return float(value) if value == value else None

    def get_volume_spike(self, symbol: str, timeframe: str = '1m',
                         threshold: float = 2.0) -> bool:
        df = self.candle_history.get(symbol, {}).get(timeframe)
        if df is None or len(df) < 20:
            return False
        avg = df['volume'].tail(20).mean()
        return bool(avg > 0 and df['volume'].iloc[-1] > avg * threshold)

    def is_connected(self) -> bool:
        return self.running and self._ws_connected

    # ----- Nur Live-Feed --------------------------------------------

    async def get_balance(self) -> Dict[str, float]:
        """Bestände je Asset — im Spot-Modell zugleich die Positionen."""
        return await self.client.get_balances()

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        pair = self.symbols.to_venue(symbol) if symbol else None
        return await self.client.list_orders(pair=pair, status='open')
