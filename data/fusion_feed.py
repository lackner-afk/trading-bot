"""
Marktdaten-Feed für Bitpanda Fusion (REST).

Schnittstellen-kompatibel zu KrakenFeed/OneTradingCCXTFeed, damit der Bot den
Feed tauschen kann, ohne dass main.py etwas davon merkt.

Fusion-Eigenheiten, die hier gekapselt werden:
  * Paarformat ist BTC-EUR (Bindestrich), intern heißt es BTC_EUR.
  * Auch Marktdaten brauchen den x-api-key-Header — ohne Key gibt es 401.
  * /v1/tickers liefert nur einen Mittelkurs, kein bid/ask. Für die Spanne
    wird deshalb das Orderbuch geholt (siehe _orderbook_loop), solange das
    aktiviert ist; sonst wird ein Spread geschätzt.
  * Zeitstempel der Kerzen sind Unix-Sekunden, nicht Millisekunden.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable

import aiohttp
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands


@dataclass
class MarketData:
    """Aktueller Marktzustand eines Symbols (gleiche Form wie in den anderen Feeds)."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    timestamp: datetime


@dataclass
class CandleData:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FusionFeed:
    """Preise und Kerzen von Bitpanda Fusion."""

    BASE_URL = 'https://api.fusion.bitpanda.com'
    DEFAULT_PAIRS = ['BTC_EUR', 'ETH_EUR', 'SOL_EUR']
    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    # Wie kraken_feed.CANDLE_HISTORY: über 250, damit der 200-EMA-Trendfilter
    # nicht still auf eine kürzere Spanne zurückfällt.
    CANDLE_HISTORY = 300

    # Fusion deckelt eine Kerzen-Abfrage bei 1440 Bars.
    MAX_CANDLE_LIMIT = 1440

    def __init__(self, api_key: str = None, api_secret: str = None, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('FusionFeed')

        self.api_key = api_key
        self.pairs: List[str] = self.config.get('pairs') or list(self.DEFAULT_PAIRS)
        self.price_interval = float(self.config.get('price_interval_s', 5))
        self.candle_interval = float(self.config.get('candle_interval_s', 60))
        self.use_orderbook = bool(self.config.get('use_orderbook', True))

        # Muss hier stehen, nicht erst in _check_pairs: schlägt der Abruf von
        # /v1/pairs fehl, würde ein späterer Zugriff sonst mit AttributeError enden.
        self.min_order_amount: Dict[str, float] = {}

        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.last_update: Dict[str, datetime] = {}

        self.running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._tasks: List[asyncio.Task] = []
        self.on_price_update: Optional[Callable] = None

    # ===== Symbol-Übersetzung =====

    @staticmethod
    def _to_fusion(symbol: str) -> str:
        """BTC_EUR -> BTC-EUR"""
        return symbol.replace('_', '-')

    @staticmethod
    def _from_fusion(pair: str) -> str:
        """BTC-EUR -> BTC_EUR"""
        return pair.replace('-', '_')

    # ===== HTTP =====

    async def _get(self, path: str, params: Dict = None) -> Optional[object]:
        """Ein GET gegen die Fusion-API. Gibt None zurück, statt zu werfen."""
        if self._session is None:
            return None
        try:
            async with self._session.get(f"{self.BASE_URL}{path}", params=params) as resp:
                if resp.status == 401:
                    self.logger.error(
                        "Fusion antwortet 401 — API-Key fehlt, ist abgelaufen oder hat "
                        "keinen Read-Scope."
                    )
                    return None
                if resp.status == 429:
                    self.logger.warning("Fusion-Ratelimit erreicht (429) — warte kurz.")
                    await asyncio.sleep(5)
                    return None
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    self.logger.warning(f"Fusion {path} -> HTTP {resp.status}: {body}")
                    return None
                return await resp.json()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning(f"Fusion {path} fehlgeschlagen: {e!r}")
            return None

    # ===== Start / Stop =====

    async def start(self):
        if not self.api_key:
            raise RuntimeError(
                "FusionFeed braucht einen API-Key (BITPANDA_API_KEY). Auch Marktdaten "
                "sind bei Fusion nur mit Key abrufbar."
            )
        self.running = True
        self._session = aiohttp.ClientSession(
            headers={'x-api-key': self.api_key, 'Accept': 'application/json'},
            timeout=aiohttp.ClientTimeout(total=20),
        )
        await self._check_pairs()
        await self._load_initial_candles()

        self._tasks = [
            asyncio.create_task(self._price_loop()),
            asyncio.create_task(self._candle_loop()),
        ]
        self.logger.info(f"FusionFeed gestartet für {', '.join(self.pairs)}")

    async def stop(self):
        self.running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._session is not None:
            await self._session.close()
            self._session = None
        self.logger.info("FusionFeed gestoppt.")

    # ===== Paare prüfen =====

    async def _check_pairs(self):
        """
        Prüft beim Start, ob die konfigurierten Paare handelbar sind, und merkt
        sich die Mindestordergröße. Ein Paar, das die Börse nicht führt, würde
        sonst erst beim ersten Trade auffallen.
        """
        data = await self._get('/v1/pairs')
        if not data:
            self.logger.warning("Konnte /v1/pairs nicht laden — überspringe Prüfung.")
            return

        available = {}
        for entry in data:
            name = entry.get('pair')
            if name:
                available[self._from_fusion(name)] = entry

        for symbol in list(self.pairs):
            info = available.get(symbol)
            if info is None:
                self.logger.error(
                    f"{symbol} wird von Fusion nicht angeboten — Symbol wird ignoriert."
                )
                self.pairs.remove(symbol)
                continue
            try:
                self.min_order_amount[symbol] = float(info.get('minOrderAmount') or 0)
            except (TypeError, ValueError):
                self.min_order_amount[symbol] = 0.0
            self.logger.info(
                f"{symbol}: min. Ordervolumen {self.min_order_amount[symbol]:.2f} "
                f"{info.get('quoteAsset', '')}"
            )

    # ===== Kerzen =====

    async def _load_initial_candles(self):
        for symbol in self.pairs:
            self.candle_history.setdefault(symbol, {})
            for tf in self.TIMEFRAMES:
                df = await self._fetch_candles(symbol, tf, limit=self.CANDLE_HISTORY)
                if df is not None and len(df) > 0:
                    self.candle_history[symbol][tf] = self._calculate_indicators(df)
                    self.logger.debug(f"Geladen: {symbol} {tf} ({len(df)} Kerzen)")
                else:
                    self.candle_history[symbol][tf] = pd.DataFrame()

        # Startpreis aus der jüngsten 1m-Kerze, damit der Bot nicht ohne Preise dasteht
        for symbol in self.pairs:
            df = self.candle_history.get(symbol, {}).get('1m')
            if df is not None and len(df) > 0:
                price = float(df['close'].iloc[-1])
                spread = price * 0.0002
                self.market_data[symbol] = MarketData(
                    symbol=symbol, price=price,
                    bid=price - spread, ask=price + spread,
                    volume_24h=float(df['volume'].sum()), change_24h=0.0,
                    timestamp=datetime.now(),
                )
                self.last_update[symbol] = datetime.now()
                self.logger.info(f"Initialisiert: {symbol} @ {price:.4f}")

    async def _fetch_candles(self, symbol: str, timeframe: str,
                             limit: int = 300) -> Optional[pd.DataFrame]:
        data = await self._get(
            f"/v1/candles/{self._to_fusion(symbol)}",
            params={'interval': timeframe, 'limit': min(limit, self.MAX_CANDLE_LIMIT)},
        )
        if not data:
            return None
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            # Fusion liefert Unix-Sekunden
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ('open', 'high', 'low', 'close', 'volume'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=['close']).sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            self.logger.warning(f"Kerzen {symbol} {timeframe} unlesbar: {e!r}")
            return None

    async def _candle_loop(self):
        await asyncio.sleep(self.candle_interval)
        while self.running:
            try:
                for symbol in self.pairs:
                    for tf in self.TIMEFRAMES:
                        df_new = await self._fetch_candles(symbol, tf, limit=10)
                        if df_new is None or df_new.empty:
                            continue
                        existing = self.candle_history.get(symbol, {}).get(tf)
                        if existing is not None and len(existing) > 0:
                            df_new = pd.concat([existing, df_new]) \
                                .drop_duplicates(subset='timestamp') \
                                .sort_values('timestamp') \
                                .tail(self.CANDLE_HISTORY)
                        self.candle_history.setdefault(symbol, {})[tf] = \
                            self._calculate_indicators(df_new)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Kerzen-Loop Fehler: {e!r}")
            await asyncio.sleep(self.candle_interval)

    # ===== Preise =====

    async def _price_loop(self):
        while self.running:
            try:
                pairs = ','.join(self._to_fusion(s) for s in self.pairs)
                data = await self._get('/v1/tickers', params={'pair': pairs})
                if data:
                    for entry in data:
                        symbol = self._from_fusion(entry.get('pair', ''))
                        if symbol not in self.pairs:
                            continue
                        try:
                            price = float(entry.get('price'))
                        except (TypeError, ValueError):
                            continue
                        if price <= 0:
                            continue
                        bid, ask = await self._spread_for(symbol, price)
                        self.market_data[symbol] = MarketData(
                            symbol=symbol, price=price, bid=bid, ask=ask,
                            volume_24h=float(entry.get('volume') or 0),
                            change_24h=0.0, timestamp=datetime.now(),
                        )
                        self.last_update[symbol] = datetime.now()

                    if self.on_price_update:
                        prices = self.get_prices()
                        if prices:
                            if asyncio.iscoroutinefunction(self.on_price_update):
                                await self.on_price_update(prices)
                            else:
                                self.on_price_update(prices)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Preis-Loop Fehler: {e!r}")
            await asyncio.sleep(self.price_interval)

    async def _spread_for(self, symbol: str, price: float) -> tuple:
        """
        Echte Geld-/Briefkurse aus dem Orderbuch. Genau das macht eine
        Shadow-Phase aussagekräftig: die simulierte Ausführung rechnet dann mit
        der Spanne, die real zu zahlen wäre, statt mit einer Schätzung.
        """
        if self.use_orderbook:
            ob = await self._get(f"/v1/orderbook/{self._to_fusion(symbol)}",
                                 params={'level': 1})
            try:
                if ob:
                    bids = ob.get('bids') or []
                    asks = ob.get('asks') or []
                    if bids and asks:
                        bid = float(bids[0][0] if isinstance(bids[0], (list, tuple))
                                    else bids[0].get('price'))
                        ask = float(asks[0][0] if isinstance(asks[0], (list, tuple))
                                    else asks[0].get('price'))
                        if bid > 0 and ask > 0:
                            return bid, ask
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        spread = price * 0.0002
        return price - spread, price + spread

    # ===== Indikatoren (identisch zu den anderen Feeds) =====

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 20:
            return df
        df = df.copy()
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['vwap'] = (
            df['volume'] * (df['high'] + df['low'] + df['close']) / 3
        ).cumsum() / df['volume'].cumsum()
        df['volume_delta'] = df['volume'] * np.where(df['close'] > df['open'], 1, -1)
        return df

    # ===== Öffentliche Schnittstelle =====

    def get_price(self, symbol: str, max_age_seconds: Optional[float] = None) -> Optional[float]:
        md = self.market_data.get(symbol)
        if md is None:
            return None
        if max_age_seconds is not None and self.get_price_age(symbol) > max_age_seconds:
            return None
        return md.price

    def get_price_age(self, symbol: str) -> float:
        last = self.last_update.get(symbol)
        if last is None:
            return float('inf')
        return (datetime.now() - last).total_seconds()

    def get_prices(self, max_age_seconds: Optional[float] = None) -> Dict[str, float]:
        if max_age_seconds is None:
            return {s: m.price for s, m in self.market_data.items()}
        return {
            s: m.price for s, m in self.market_data.items()
            if self.get_price_age(s) <= max_age_seconds
        }

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        return self.market_data.get(symbol)

    def get_candles(self, symbol: str, timeframe: str = '1m',
                    n: int = 50) -> Optional[pd.DataFrame]:
        if symbol in self.candle_history and timeframe in self.candle_history[symbol]:
            df = self.candle_history[symbol][timeframe]
            return df.tail(n) if len(df) > 0 else None
        return None

    def get_latest_candle(self, symbol: str, timeframe: str = '1m') -> Optional[CandleData]:
        df = self.get_candles(symbol, timeframe, n=1)
        if df is None or len(df) == 0:
            return None
        row = df.iloc[-1]
        return CandleData(
            symbol=symbol, timeframe=timeframe, timestamp=row['timestamp'],
            open=float(row['open']), high=float(row['high']), low=float(row['low']),
            close=float(row['close']), volume=float(row['volume']),
        )

    def get_rsi(self, symbol: str, timeframe: str = '1m') -> Optional[float]:
        df = self.get_candles(symbol, timeframe, n=20)
        if df is None or 'rsi' not in df.columns or len(df) == 0:
            return None
        value = df['rsi'].iloc[-1]
        return None if pd.isna(value) else float(value)

    def get_volume_spike(self, symbol: str, timeframe: str = '1m',
                         threshold: float = 2.0) -> bool:
        df = self.get_candles(symbol, timeframe, n=20)
        if df is None or len(df) < 2:
            return False
        avg = df['volume'].iloc[:-1].mean()
        return bool(df['volume'].iloc[-1] > avg * threshold) if avg > 0 else False

    def is_connected(self) -> bool:
        return any(self.get_price_age(s) < 30 for s in self.pairs)
