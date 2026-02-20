"""
Kraken Datenfeed — Echtzeit EUR-Pairs via CCXT
Ersetzt OneTradingFeed mit aktivem Exchange (Kraken hat BTC/ETH/SOL/XRP EUR)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

from .crypto_feed import CandleData, MarketData


class KrakenFeed:
    """
    Kraken Live-Datenfeed via CCXT (kein API-Key nötig für Marktdaten)

    - Preis-Polling alle 5s via fetch_tickers()
    - Kerzen-Update alle 60s via fetch_ohlcv()
    - Symbole extern als BTC_EUR, intern als BTC/EUR
    """

    DEFAULT_PAIRS = ['BTC_EUR', 'ETH_EUR', 'SOL_EUR', 'XRP_EUR']
    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    def __init__(self, api_key: str = None, api_secret: str = None, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('KrakenFeed')

        self.pairs = self.config.get('pairs', self.DEFAULT_PAIRS)

        # Symbol-Mapping: BTC_EUR → BTC/EUR (CCXT-Format)
        self._to_ccxt = {p: p.replace('_', '/') for p in self.pairs}
        self._from_ccxt = {v: k for k, v in self._to_ccxt.items()}
        self._ccxt_pairs = list(self._to_ccxt.values())

        # Kraken Exchange (kein Key nötig für public endpoints)
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,
        })

        # Daten-Storage (keys immer im BTC_EUR Format)
        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.last_update: Dict[str, datetime] = {}

        # State
        self.running = False
        self._ws_connected = False  # Kompatibilität mit OneTradingFeed

        # Callbacks
        self.on_price_update: Optional[Callable] = None
        self.on_candle_close: Optional[Callable] = None

        # Tracking für highest_prices (Trailing-Stop)
        self.highest_prices: Dict[str, float] = {}

    async def start(self):
        """Startet den Feed"""
        self.running = True
        self.logger.info(f"Starte Kraken Feed für {self.pairs}")

        # Initiale Kerzen laden
        await self._load_initial_candles()

        # Preis-Polling (alle 5s)
        asyncio.create_task(self._price_loop())

        # Kerzen-Update (alle 60s)
        asyncio.create_task(self._candle_loop())

    async def stop(self):
        """Stoppt den Feed"""
        self.running = False
        await self.exchange.close()
        self.logger.info("Kraken Feed gestoppt")

    # ===== Initialisierung =====

    async def _load_initial_candles(self):
        """Lädt historische Kerzen für alle Pairs und Timeframes"""
        tasks = [
            self._fetch_and_store(symbol, tf)
            for symbol in self.pairs
            for tf in self.TIMEFRAMES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Initiale Preise aus 1m-Kerzen setzen
        for symbol in self.pairs:
            df = self.candle_history.get(symbol, {}).get('1m')
            if df is not None and len(df) > 0:
                price = float(df['close'].iloc[-1])
                spread = price * 0.0001
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    price=price,
                    bid=price - spread,
                    ask=price + spread,
                    volume_24h=float(df['volume'].sum()),
                    change_24h=0.0,
                    timestamp=datetime.now()
                )
                self.last_update[symbol] = datetime.now()
                self.logger.info(f"Initialisiert: {symbol} @ {price:.4f} EUR")

    async def _fetch_and_store(self, symbol: str, timeframe: str):
        """Holt Kerzen für ein Symbol/Timeframe und speichert sie"""
        self.candle_history.setdefault(symbol, {})
        ccxt_symbol = self._to_ccxt[symbol]
        try:
            ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=100)
            df = self._ohlcv_to_df(ohlcv)
            df = self._calculate_indicators(df)
            self.candle_history[symbol][timeframe] = df
            self.logger.debug(f"Geladen: {symbol} {timeframe} ({len(df)} Kerzen)")
        except Exception as e:
            self.logger.warning(f"Fehler beim Laden von {symbol} {timeframe}: {e}")
            self.candle_history[symbol][timeframe] = pd.DataFrame()

    # ===== Preis-Loop =====

    async def _price_loop(self):
        """Aktualisiert Preise alle 5 Sekunden via fetch_tickers"""
        while self.running:
            try:
                tickers = await self.exchange.fetch_tickers(self._ccxt_pairs)

                for ccxt_sym, ticker in tickers.items():
                    symbol = self._from_ccxt.get(ccxt_sym)
                    if not symbol:
                        continue

                    price = ticker.get('last') or ticker.get('close', 0)
                    if not price or price <= 0:
                        continue

                    spread = price * 0.0002
                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=float(price),
                        bid=float(ticker.get('bid') or price - spread),
                        ask=float(ticker.get('ask') or price + spread),
                        volume_24h=float(ticker.get('baseVolume') or 0),
                        change_24h=float(ticker.get('percentage') or 0),
                        timestamp=datetime.now()
                    )
                    self.last_update[symbol] = datetime.now()
                    self._ws_connected = True

                # Callback aufrufen
                if self.on_price_update:
                    prices = self.get_prices()
                    if prices:
                        if asyncio.iscoroutinefunction(self.on_price_update):
                            await self.on_price_update(prices)
                        else:
                            self.on_price_update(prices)

            except Exception as e:
                self.logger.error(f"Preis-Loop Fehler: {e}")
                self._ws_connected = False

            await asyncio.sleep(5)

    # ===== Kerzen-Loop =====

    async def _candle_loop(self):
        """Aktualisiert Kerzen alle 60 Sekunden"""
        await asyncio.sleep(60)

        while self.running:
            try:
                for symbol in self.pairs:
                    ccxt_sym = self._to_ccxt[symbol]
                    for tf in self.TIMEFRAMES:
                        try:
                            ohlcv = await self.exchange.fetch_ohlcv(ccxt_sym, tf, limit=10)
                            df_new = self._ohlcv_to_df(ohlcv)
                            existing = self.candle_history.get(symbol, {}).get(tf)
                            if existing is not None and len(existing) > 0:
                                df_new = pd.concat([existing, df_new]) \
                                    .drop_duplicates(subset='timestamp') \
                                    .sort_values('timestamp') \
                                    .tail(100)
                            df_new = self._calculate_indicators(df_new)
                            self.candle_history.setdefault(symbol, {})[tf] = df_new
                        except Exception as e:
                            self.logger.debug(f"Kerzen-Update {symbol} {tf}: {e}")

            except Exception as e:
                self.logger.error(f"Kerzen-Loop Fehler: {e}")

            await asyncio.sleep(60)

    # ===== Hilfsmethoden =====

    def _ohlcv_to_df(self, ohlcv: List) -> pd.DataFrame:
        """Konvertiert CCXT OHLCV-Liste zu DataFrame"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet RSI, BB, EMA, VWAP"""
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

    # ===== Public Interface (kompatibel mit OneTradingFeed) =====

    def get_price(self, symbol: str) -> Optional[float]:
        md = self.market_data.get(symbol)
        return md.price if md else None

    def get_prices(self) -> Dict[str, float]:
        return {s: m.price for s, m in self.market_data.items()}

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        return self.market_data.get(symbol)

    def get_candles(self, symbol: str, timeframe: str = '1m', n: int = 50) -> Optional[pd.DataFrame]:
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
            timestamp=row['timestamp'],
            open=row['open'], high=row['high'], low=row['low'],
            close=row['close'], volume=row['volume'],
            rsi=row.get('rsi'), bb_upper=row.get('bb_upper'),
            bb_middle=row.get('bb_middle'), bb_lower=row.get('bb_lower'),
            ema_9=row.get('ema_9'), ema_21=row.get('ema_21'),
            vwap=row.get('vwap'), volume_delta=row.get('volume_delta')
        )

    def get_rsi(self, symbol: str, timeframe: str = '1m') -> Optional[float]:
        c = self.get_latest_candle(symbol, timeframe)
        return c.rsi if c else None

    def get_volume_spike(self, symbol: str, timeframe: str = '1m', threshold: float = 2.0) -> bool:
        df = self.get_candles(symbol, timeframe, n=20)
        if df is None or len(df) < 10:
            return False
        avg = df['volume'].iloc[:-1].mean()
        return df['volume'].iloc[-1] > avg * threshold if avg > 0 else False

    def is_connected(self) -> bool:
        return any(
            (datetime.now() - self.last_update.get(s, datetime.min)).total_seconds() < 30
            for s in self.pairs
        )
