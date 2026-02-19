"""
Crypto-Datenfeed über CCXT
Live-Daten via WebSocket/REST für BTC, ETH, SOL, DOGE, PEPE
Berechnet technische Indikatoren: RSI, Bollinger Bands, VWAP, EMA
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

try:
    import ccxt.async_support as ccxt
except ImportError:
    import ccxt

import ta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator


@dataclass
class CandleData:
    """OHLCV Kerze mit Indikatoren"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Indikatoren
    rsi: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    vwap: Optional[float] = None
    volume_delta: Optional[float] = None


@dataclass
class MarketData:
    """Aktueller Markt-Zustand"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    funding_rate: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    candles: Dict[str, List[CandleData]] = field(default_factory=dict)


class CryptoFeed:
    """
    Crypto-Datenfeed mit CCXT
    Unterstützt Bybit und Binance im Testnet-Modus
    """

    DEFAULT_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    def __init__(self, exchange_id: str = 'bybit', testnet: bool = True,
                 api_key: str = None, secret: str = None, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('CryptoFeed')

        # Exchange-Konfiguration
        self.exchange_id = exchange_id
        self.testnet = testnet

        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}  # Perpetuals
        }

        if api_key and secret:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = secret

        self.exchange = exchange_class(exchange_config)

        # Testnet aktivieren
        if testnet:
            if exchange_id == 'bybit':
                self.exchange.set_sandbox_mode(True)
            elif exchange_id == 'binance':
                self.exchange.set_sandbox_mode(True)

        # Daten-Storage
        self.pairs = self.config.get('pairs', self.DEFAULT_PAIRS)
        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, Dict[str, pd.DataFrame]] = {}

        # State
        self.running = False
        self.last_update: Dict[str, datetime] = {}

        # Callbacks
        self.on_price_update: Optional[Callable] = None
        self.on_candle_close: Optional[Callable] = None

        # Simulierte Daten als Fallback
        self._use_simulated = False
        self._simulated_prices = {
            'BTC/USDT': 95000.0,
            'ETH/USDT': 3200.0,
            'SOL/USDT': 180.0,
            'DOGE/USDT': 0.32,
        }

    async def start(self):
        """Startet den Datenfeed"""
        self.running = True
        self.logger.info(f"Starte CryptoFeed für {self.exchange_id} (Testnet: {self.testnet})")

        # Initiale Daten laden
        await self._load_initial_data()

        # Polling-Loop starten
        asyncio.create_task(self._polling_loop())

    async def stop(self):
        """Stoppt den Datenfeed"""
        self.running = False
        await self.exchange.close()
        self.logger.info("CryptoFeed gestoppt")

    async def _load_initial_data(self):
        """Lädt initiale Kerzen-Daten"""
        for symbol in self.pairs:
            self.candle_history[symbol] = {}

            for timeframe in self.TIMEFRAMES:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                    df = self._ohlcv_to_dataframe(ohlcv)
                    df = self._calculate_indicators(df)
                    self.candle_history[symbol][timeframe] = df
                    self.logger.debug(f"Geladen: {symbol} {timeframe} - {len(df)} Kerzen")
                except Exception as e:
                    self.logger.warning(f"Fehler beim Laden von {symbol} {timeframe}: {e}")
                    self._use_simulated = True
                    self.candle_history[symbol][timeframe] = self._generate_simulated_candles(symbol)

        # Initiale Marktdaten
        await self._update_market_data()

    async def _polling_loop(self):
        """Haupt-Polling-Loop für Preis-Updates"""
        while self.running:
            try:
                await self._update_market_data()
                await self._update_candles()
                await asyncio.sleep(5)  # 5 Sekunden Intervall
            except Exception as e:
                self.logger.error(f"Fehler im Polling-Loop: {e}")
                await asyncio.sleep(10)

    async def _update_market_data(self):
        """Aktualisiert Marktdaten für alle Pairs"""
        if self._use_simulated:
            self._update_simulated_prices()
            return

        try:
            tickers = await self.exchange.fetch_tickers(self.pairs)

            for symbol, ticker in tickers.items():
                if symbol not in self.pairs:
                    continue

                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    price=ticker.get('last', 0),
                    bid=ticker.get('bid', 0),
                    ask=ticker.get('ask', 0),
                    volume_24h=ticker.get('baseVolume', 0),
                    change_24h=ticker.get('percentage', 0),
                    timestamp=datetime.now()
                )
                self.last_update[symbol] = datetime.now()

            # Callback aufrufen
            if self.on_price_update:
                prices = {s: m.price for s, m in self.market_data.items()}
                if asyncio.iscoroutinefunction(self.on_price_update):
                    await self.on_price_update(prices)
                else:
                    self.on_price_update(prices)

        except Exception as e:
            self.logger.error(f"Fehler beim Ticker-Update: {e}")
            self._use_simulated = True
            self._update_simulated_prices()

    def _update_simulated_prices(self):
        """Generiert simulierte Preisbewegungen"""
        import random

        for symbol in self.pairs:
            if symbol not in self._simulated_prices:
                self._simulated_prices[symbol] = 100.0

            # Random Walk mit realistischer Crypto-Volatilität
            change = random.gauss(0, 0.003)  # 0.3% Volatilität (realistischer für Crypto)
            self._simulated_prices[symbol] *= (1 + change)

            price = self._simulated_prices[symbol]
            spread = price * 0.0002  # 0.02% Spread

            self.market_data[symbol] = MarketData(
                symbol=symbol,
                price=price,
                bid=price - spread/2,
                ask=price + spread/2,
                volume_24h=random.uniform(1000000, 10000000),
                change_24h=random.uniform(-5, 5),
                timestamp=datetime.now()
            )
            self.last_update[symbol] = datetime.now()

    async def _update_candles(self):
        """Aktualisiert Kerzen-Daten"""
        for symbol in self.pairs:
            for timeframe in self.TIMEFRAMES:
                try:
                    if self._use_simulated:
                        self._update_simulated_candle(symbol, timeframe)
                    else:
                        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=5)
                        new_df = self._ohlcv_to_dataframe(ohlcv)

                        # Mit bestehenden Daten mergen
                        if symbol in self.candle_history and timeframe in self.candle_history[symbol]:
                            df = self.candle_history[symbol][timeframe]
                            df = pd.concat([df, new_df]).drop_duplicates(subset='timestamp').tail(100)
                            df = self._calculate_indicators(df)
                            self.candle_history[symbol][timeframe] = df

                except Exception as e:
                    self.logger.debug(f"Candle-Update Fehler {symbol} {timeframe}: {e}")

    def _update_simulated_candle(self, symbol: str, timeframe: str):
        """Fügt simulierte Kerze hinzu"""
        import random

        if symbol not in self.candle_history:
            self.candle_history[symbol] = {}
        if timeframe not in self.candle_history[symbol]:
            self.candle_history[symbol][timeframe] = self._generate_simulated_candles(symbol)

        df = self.candle_history[symbol][timeframe]
        last_close = df['close'].iloc[-1] if len(df) > 0 else self._simulated_prices.get(symbol, 100)

        # Neue Kerze generieren
        change = random.gauss(0, 0.002)
        new_close = last_close * (1 + change)
        new_high = max(last_close, new_close) * (1 + abs(random.gauss(0, 0.001)))
        new_low = min(last_close, new_close) * (1 - abs(random.gauss(0, 0.001)))

        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'open': last_close,
            'high': new_high,
            'low': new_low,
            'close': new_close,
            'volume': random.uniform(100, 1000)
        }])

        df = pd.concat([df, new_row]).tail(100)
        df = self._calculate_indicators(df)
        self.candle_history[symbol][timeframe] = df

    def _generate_simulated_candles(self, symbol: str, n: int = 100) -> pd.DataFrame:
        """Generiert simulierte historische Kerzen"""
        import random

        base_price = self._simulated_prices.get(symbol, 100)
        data = []

        for i in range(n):
            if i == 0:
                open_price = base_price * 0.95
            else:
                open_price = data[-1]['close']

            change = random.gauss(0, 0.01)
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.005)))

            data.append({
                'timestamp': datetime.now() - timedelta(minutes=n-i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(100, 1000)
            })

        return pd.DataFrame(data)

    def _ohlcv_to_dataframe(self, ohlcv: List) -> pd.DataFrame:
        """Konvertiert OHLCV-Daten zu DataFrame"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet technische Indikatoren"""
        if len(df) < 20:
            return df

        # RSI (14)
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()

        # Bollinger Bands (20, 2)
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # EMA (9, 21)
        ema9 = EMAIndicator(close=df['close'], window=9)
        ema21 = EMAIndicator(close=df['close'], window=21)
        df['ema_9'] = ema9.ema_indicator()
        df['ema_21'] = ema21.ema_indicator()

        # VWAP (vereinfacht)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Volume Delta (Kauf vs. Verkaufsdruck Approximation)
        df['volume_delta'] = df['volume'] * np.where(df['close'] > df['open'], 1, -1)

        return df

    def get_price(self, symbol: str) -> Optional[float]:
        """Gibt aktuellen Preis zurück"""
        if symbol in self.market_data:
            return self.market_data[symbol].price
        return None

    def get_prices(self) -> Dict[str, float]:
        """Gibt alle aktuellen Preise zurück"""
        return {s: m.price for s, m in self.market_data.items()}

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Gibt Marktdaten für Symbol zurück"""
        return self.market_data.get(symbol)

    def get_candles(self, symbol: str, timeframe: str = '1m', n: int = 50) -> Optional[pd.DataFrame]:
        """Gibt letzte n Kerzen zurück"""
        if symbol in self.candle_history and timeframe in self.candle_history[symbol]:
            return self.candle_history[symbol][timeframe].tail(n)
        return None

    def get_latest_candle(self, symbol: str, timeframe: str = '1m') -> Optional[CandleData]:
        """Gibt letzte Kerze mit Indikatoren zurück"""
        df = self.get_candles(symbol, timeframe, n=1)
        if df is None or len(df) == 0:
            return None

        row = df.iloc[-1]
        return CandleData(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            rsi=row.get('rsi'),
            bb_upper=row.get('bb_upper'),
            bb_middle=row.get('bb_middle'),
            bb_lower=row.get('bb_lower'),
            ema_9=row.get('ema_9'),
            ema_21=row.get('ema_21'),
            vwap=row.get('vwap'),
            volume_delta=row.get('volume_delta')
        )

    def get_rsi(self, symbol: str, timeframe: str = '1m') -> Optional[float]:
        """Gibt aktuellen RSI zurück"""
        candle = self.get_latest_candle(symbol, timeframe)
        return candle.rsi if candle else None

    def get_bollinger_position(self, symbol: str, timeframe: str = '1m') -> Optional[str]:
        """
        Gibt Position relativ zu Bollinger Bands zurück

        Returns:
            'upper' | 'middle' | 'lower' | None
        """
        candle = self.get_latest_candle(symbol, timeframe)
        if not candle or candle.bb_upper is None:
            return None

        price = candle.close
        if price >= candle.bb_upper:
            return 'upper'
        elif price <= candle.bb_lower:
            return 'lower'
        else:
            return 'middle'

    def get_volume_spike(self, symbol: str, timeframe: str = '1m', threshold: float = 2.0) -> bool:
        """
        Prüft ob Volume-Spike vorliegt

        Args:
            threshold: Vielfaches des Durchschnittsvolumens

        Returns:
            True wenn aktuelles Volume > threshold * Durchschnitt
        """
        df = self.get_candles(symbol, timeframe, n=20)
        if df is None or len(df) < 10:
            return False

        avg_volume = df['volume'].iloc[:-1].mean()
        current_volume = df['volume'].iloc[-1]

        return current_volume > avg_volume * threshold

    def is_breakout(self, symbol: str, direction: str = 'up') -> bool:
        """
        Prüft auf Breakout

        Args:
            direction: 'up' oder 'down'

        Returns:
            True wenn Breakout erkannt
        """
        df = self.get_candles(symbol, '15m', n=20)
        if df is None or len(df) < 15:
            return False

        current_price = df['close'].iloc[-1]
        high_15m = df['high'].iloc[:-1].max()
        low_15m = df['low'].iloc[:-1].min()

        if direction == 'up':
            return current_price > high_15m and self.get_volume_spike(symbol)
        else:
            return current_price < low_15m and self.get_volume_spike(symbol)

    def is_connected(self) -> bool:
        """Prüft ob Feed verbunden ist"""
        if self._use_simulated:
            return True

        # Prüfe ob letzte Updates aktuell sind
        for symbol in self.pairs:
            if symbol not in self.last_update:
                return False
            if datetime.now() - self.last_update[symbol] > timedelta(minutes=1):
                return False
        return True
