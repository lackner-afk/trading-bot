"""
One Trading Datenfeed
Live-Daten via WebSocket (PRICE_TICKS) und REST API (Candlesticks)
Dokumentation: https://docs.onetrading.com/websocket
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import aiohttp
import websockets
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

from .crypto_feed import CandleData, MarketData


class OneTradingFeed:
    """
    One Trading Live-Datenfeed
    WebSocket für Echtzeit-Preisticks, REST für historische Kerzen
    """

    WS_URL = "wss://streams.fast.onetrading.com"
    REST_BASE = "https://api.onetrading.com/fast/v1"

    DEFAULT_PAIRS = ['BTC_EUR', 'ETH_EUR', 'SOL_EUR', 'XRP_EUR']
    TIMEFRAMES = ['1m', '5m', '15m', '1h']

    # One Trading Timeframe-Mapping
    TIMEFRAME_MAP = {
        '1m':  {'unit': 'MINUTES', 'period': 1},
        '5m':  {'unit': 'MINUTES', 'period': 5},
        '15m': {'unit': 'MINUTES', 'period': 15},
        '1h':  {'unit': 'HOURS',   'period': 1},
    }

    def __init__(self, api_key: str = None, api_secret: str = None, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('OneTradingFeed')

        self.api_key = api_key
        self.api_secret = api_secret

        # Trading-Pairs (Format: BTC_USDT, ETH_EUR usw.)
        self.pairs = self.config.get('pairs', self.DEFAULT_PAIRS)

        # Daten-Storage
        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, Dict[str, pd.DataFrame]] = {}

        # State
        self.running = False
        self.last_update: Dict[str, datetime] = {}
        self._ws_connected = False

        # Callbacks
        self.on_price_update: Optional[Callable] = None
        self.on_candle_close: Optional[Callable] = None

        # Simulierte Daten als Fallback
        self._use_simulated = False
        self._simulated_prices = {
            'BTC_EUR': 87000.0,
            'ETH_EUR': 2700.0,
            'SOL_EUR': 155.0,
            'XRP_EUR': 2.30,
        }

    async def start(self):
        """Startet den Datenfeed"""
        self.running = True
        self.logger.info(f"Starte One Trading Feed für {self.pairs}")

        # Historische Kerzen via REST laden
        await self._load_initial_candles()

        # WebSocket-Loop starten (mit Auto-Reconnect)
        asyncio.create_task(self._websocket_loop())

        # Periodisches Candle-Update via REST (alle 60s)
        asyncio.create_task(self._candle_refresh_loop())

    async def stop(self):
        """Stoppt den Datenfeed"""
        self.running = False
        self.logger.info("One Trading Feed gestoppt")

    # ===== REST API =====

    async def _load_initial_candles(self):
        """Lädt initiale historische Kerzen via REST - alle parallel"""
        async with aiohttp.ClientSession() as session:
            # Alle (symbol, timeframe)-Kombinationen parallel laden
            tasks = [
                self._load_one(session, symbol, timeframe)
                for symbol in self.pairs
                for timeframe in self.TIMEFRAMES
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Initiale MarketData aus letzter 1m-Kerze
            for symbol in self.pairs:
                df_1m = self.candle_history.get(symbol, {}).get('1m')
                if df_1m is not None and len(df_1m) > 0:
                    last_close = float(df_1m['close'].iloc[-1])
                    spread = last_close * 0.0001
                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=last_close,
                        bid=last_close - spread,
                        ask=last_close + spread,
                        volume_24h=float(df_1m['volume'].tail(1440).sum()),
                        change_24h=0.0,
                        timestamp=datetime.now()
                    )
                    self.last_update[symbol] = datetime.now()

    async def _load_one(self, session: aiohttp.ClientSession, symbol: str, timeframe: str):
        """Lädt Kerzen für ein einzelnes Symbol/Timeframe"""
        self.candle_history.setdefault(symbol, {})
        try:
            df = await self._fetch_candles(session, symbol, timeframe, count=100)
            if df is not None and len(df) > 0:
                df = self._calculate_indicators(df)
                self.candle_history[symbol][timeframe] = df
                self.logger.debug(f"Geladen: {symbol} {timeframe} - {len(df)} Kerzen")
            else:
                raise ValueError("Leere Antwort vom Server")
        except Exception as e:
            self.logger.warning(f"REST-Fehler {symbol} {timeframe}: {e} - Fallback auf Simulation")
            self._use_simulated = True
            self.candle_history[symbol][timeframe] = self._generate_simulated_candles(symbol)

    async def _fetch_candles(self, session: aiohttp.ClientSession, symbol: str,
                             timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Holt Kerzen von der One Trading REST API"""
        tf = self.TIMEFRAME_MAP.get(timeframe)
        if not tf:
            return None

        # Zeitfenster berechnen (API braucht from + to)
        now = datetime.now(timezone.utc)
        minutes_back = {
            '1m': count,
            '5m': count * 5,
            '15m': count * 15,
            '1h': count * 60,
        }.get(timeframe, count)
        frm = (now - timedelta(minutes=minutes_back)).strftime('%Y-%m-%dT%H:%M:%SZ')
        to  = now.strftime('%Y-%m-%dT%H:%M:%SZ')

        url = f"{self.REST_BASE}/candlesticks/{symbol}"
        params = {
            'unit': tf['unit'],
            'period': str(tf['period']),
            'from': frm,
            'to': to,
        }

        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                text = await resp.text()
                self.logger.warning(f"REST {resp.status} für {symbol} {timeframe}: {text[:100]}")
                return None

            data = await resp.json()
            candles = data.get('candlesticks', [])
            if not candles:
                return None

            rows = [{
                'timestamp': pd.to_datetime(c['time']),
                'open': float(c['open']),
                'high': float(c['high']),
                'low': float(c['low']),
                'close': float(c['close']),
                'volume': float(c['volume']),
            } for c in candles]

            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

    async def _candle_refresh_loop(self):
        """Aktualisiert die letzten Kerzen periodisch via REST"""
        await asyncio.sleep(60)  # Erster Refresh nach 60s

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    for symbol in self.pairs:
                        if self._use_simulated:
                            self._update_simulated_data(symbol)
                            continue

                        latest_close = None
                        for timeframe in self.TIMEFRAMES:
                            try:
                                df_new = await self._fetch_candles(session, symbol, timeframe, count=10)
                                if df_new is not None and len(df_new) > 0:
                                    existing = self.candle_history.get(symbol, {}).get(timeframe)
                                    if existing is not None:
                                        df_new = pd.concat([existing, df_new]) \
                                            .drop_duplicates(subset='timestamp') \
                                            .sort_values('timestamp') \
                                            .tail(100)
                                    df_new = self._calculate_indicators(df_new)
                                    self.candle_history.setdefault(symbol, {})[timeframe] = df_new
                                    # 1m-Kerze → market_data updaten
                                    if timeframe == '1m':
                                        latest_close = float(df_new['close'].iloc[-1])
                            except Exception as e:
                                self.logger.debug(f"Candle-Refresh {symbol} {timeframe}: {e}")

                        # market_data aus REST-Preis updaten (Fallback wenn WS still ist)
                        if latest_close and latest_close > 0:
                            existing_md = self.market_data.get(symbol)
                            # Nur updaten wenn WS-Daten älter als 90s sind
                            last = self.last_update.get(symbol, datetime.min)
                            if (datetime.now() - last).total_seconds() > 90:
                                spread = latest_close * 0.0001
                                self.market_data[symbol] = MarketData(
                                    symbol=symbol,
                                    price=latest_close,
                                    bid=latest_close - spread,
                                    ask=latest_close + spread,
                                    volume_24h=existing_md.volume_24h if existing_md else 0.0,
                                    change_24h=existing_md.change_24h if existing_md else 0.0,
                                    timestamp=datetime.now()
                                )
                                self.last_update[symbol] = datetime.now()
                                self.logger.debug(f"Preis via REST aktualisiert: {symbol} @ {latest_close:.4f}")

                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Candle-Refresh-Loop Fehler: {e}")
                await asyncio.sleep(60)

    # ===== WebSocket =====

    async def _websocket_loop(self):
        """WebSocket-Loop mit automatischer Wiederverbindung"""
        while self.running:
            try:
                self.logger.info(f"Verbinde WebSocket: {self.WS_URL}")
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws_connected = True
                    self.logger.info("WebSocket verbunden - sende Subscription...")

                    await self._subscribe_price_ticks(ws)

                    async for message in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            self.logger.error(f"Nachrichtenverarbeitung fehlgeschlagen: {e}")

            except Exception as e:
                self._ws_connected = False
                self.logger.error(f"WebSocket-Fehler: {e}")
                if self.running:
                    self.logger.info("WebSocket-Wiederverbindung in 5 Sekunden...")
                    await asyncio.sleep(5)

    async def _subscribe_price_ticks(self, ws):
        """Abonniert PRICE_TICKS für alle konfigurierten Pairs"""
        # Laut Doku: eine Subscription pro SUBSCRIBE-Nachricht
        subscribe_msg = {
            "type": "SUBSCRIBE",
            "channels": [
                {
                    "name": "PRICE_TICKS",
                    "instrument_codes": self.pairs
                }
            ]
        }
        await ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"PRICE_TICKS abonniert: {self.pairs}")

    async def _handle_message(self, data: dict):
        """Verarbeitet eingehende WebSocket-Nachrichten"""
        msg_type = data.get('type')

        if msg_type == 'PRICE_TICK':
            await self._handle_price_tick(data)
        elif msg_type == 'SUBSCRIPTIONS':
            self.logger.info(f"Subscription bestätigt: {data.get('channels')}")
        elif msg_type == 'ERROR':
            self.logger.error(f"WS Fehler vom Server: {data.get('error')}")

    async def _handle_price_tick(self, tick: dict):
        """Verarbeitet einzelnen Preis-Tick und aktualisiert MarketData"""
        symbol = tick.get('instrument_code')
        if symbol not in self.pairs:
            return

        try:
            price = float(tick['price'])
            best_bid = float(tick.get('best_bid') or price * 0.9999)
            best_ask = float(tick.get('best_ask') or price * 1.0001)

            # Bestehende 24h-Werte beibehalten
            existing = self.market_data.get(symbol)
            vol_24h = existing.volume_24h if existing else 0.0
            chg_24h = existing.change_24h if existing else 0.0

            self.market_data[symbol] = MarketData(
                symbol=symbol,
                price=price,
                bid=best_bid,
                ask=best_ask,
                volume_24h=vol_24h,
                change_24h=chg_24h,
                timestamp=datetime.now()
            )
            self.last_update[symbol] = datetime.now()

        except (KeyError, ValueError) as e:
            self.logger.debug(f"Ungültiger Tick für {symbol}: {e}")
            return

        # Callback aufrufen
        if self.on_price_update:
            prices = {s: m.price for s, m in self.market_data.items()}
            if asyncio.iscoroutinefunction(self.on_price_update):
                await self.on_price_update(prices)
            else:
                self.on_price_update(prices)

    # ===== Technische Indikatoren =====

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet RSI, Bollinger Bands, EMA, VWAP"""
        if len(df) < 20:
            return df

        df = df.copy()

        # RSI (14)
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()

        # Bollinger Bands (20, 2)
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # EMA (9, 21)
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()

        # VWAP (vereinfacht)
        df['vwap'] = (
            df['volume'] * (df['high'] + df['low'] + df['close']) / 3
        ).cumsum() / df['volume'].cumsum()

        # Volume Delta (Kauf- vs. Verkaufsdruck)
        df['volume_delta'] = df['volume'] * np.where(df['close'] > df['open'], 1, -1)

        return df

    # ===== Simulierter Fallback =====

    def _update_simulated_data(self, symbol: str):
        """Aktualisiert simulierte Preise (Fallback wenn API nicht erreichbar)"""
        import random

        if symbol not in self._simulated_prices:
            self._simulated_prices[symbol] = 100.0

        change = random.gauss(0, 0.003)
        self._simulated_prices[symbol] *= (1 + change)
        price = self._simulated_prices[symbol]
        spread = price * 0.0002

        self.market_data[symbol] = MarketData(
            symbol=symbol,
            price=price,
            bid=price - spread / 2,
            ask=price + spread / 2,
            volume_24h=random.uniform(1_000_000, 10_000_000),
            change_24h=random.uniform(-5, 5),
            timestamp=datetime.now()
        )
        self.last_update[symbol] = datetime.now()

        for timeframe in self.TIMEFRAMES:
            self._append_simulated_candle(symbol, timeframe)

    def _append_simulated_candle(self, symbol: str, timeframe: str):
        """Fügt eine neue simulierte Kerze zur History hinzu"""
        import random

        if symbol not in self.candle_history or timeframe not in self.candle_history[symbol]:
            self.candle_history.setdefault(symbol, {})[timeframe] = \
                self._generate_simulated_candles(symbol)
            return

        df = self.candle_history[symbol][timeframe]
        last_close = float(df['close'].iloc[-1]) if len(df) > 0 else self._simulated_prices.get(symbol, 100)

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
            open_price = data[-1]['close'] if i > 0 else base_price * 0.95
            change = random.gauss(0, 0.01)
            close_price = open_price * (1 + change)
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=n - i),
                'open': open_price,
                'high': max(open_price, close_price) * (1 + abs(random.gauss(0, 0.005))),
                'low': min(open_price, close_price) * (1 - abs(random.gauss(0, 0.005))),
                'close': close_price,
                'volume': random.uniform(100, 1000)
            })

        return pd.DataFrame(data)

    # ===== Public Interface (kompatibel mit CryptoFeed) =====

    def get_price(self, symbol: str) -> Optional[float]:
        """Gibt aktuellen Preis zurück"""
        md = self.market_data.get(symbol)
        return md.price if md else None

    def get_prices(self) -> Dict[str, float]:
        """Gibt alle aktuellen Preise zurück"""
        return {s: m.price for s, m in self.market_data.items()}

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Gibt vollständige Marktdaten zurück"""
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
        """Gibt Position relativ zu Bollinger Bands zurück: 'upper' | 'middle' | 'lower'"""
        candle = self.get_latest_candle(symbol, timeframe)
        if not candle or candle.bb_upper is None:
            return None

        if candle.close >= candle.bb_upper:
            return 'upper'
        elif candle.close <= candle.bb_lower:
            return 'lower'
        return 'middle'

    def get_volume_spike(self, symbol: str, timeframe: str = '1m', threshold: float = 2.0) -> bool:
        """Prüft ob Volume-Spike vorliegt (aktuell > threshold * Durchschnitt)"""
        df = self.get_candles(symbol, timeframe, n=20)
        if df is None or len(df) < 10:
            return False
        avg_volume = df['volume'].iloc[:-1].mean()
        current_volume = df['volume'].iloc[-1]
        return current_volume > avg_volume * threshold

    def is_breakout(self, symbol: str, direction: str = 'up') -> bool:
        """Prüft ob Preisausbruch vorliegt"""
        df = self.get_candles(symbol, '15m', n=20)
        if df is None or len(df) < 15:
            return False
        current_price = df['close'].iloc[-1]
        if direction == 'up':
            return current_price > df['high'].iloc[:-1].max() and self.get_volume_spike(symbol)
        return current_price < df['low'].iloc[:-1].min() and self.get_volume_spike(symbol)

    def is_connected(self) -> bool:
        """Prüft ob Feed aktive Daten liefert"""
        if self._use_simulated:
            return True
        if not self._ws_connected:
            return False
        return any(
            datetime.now() - self.last_update.get(s, datetime.min) < timedelta(minutes=2)
            for s in self.pairs
        )
