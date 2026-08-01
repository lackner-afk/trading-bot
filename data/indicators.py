"""
Technische Indikatoren — eine Implementierung für alle Feeds und den Backtester.

Vorher existierte `_calculate_indicators` dreifach: in `kraken_feed.py`,
`onetrading_ccxt_feed.py` und `backtester.py`. Solange die Varianten
auseinanderlaufen können, ist jede behauptete Parität zwischen Backtest und
Live unbelegt — die Strategie sähe im Test andere Zahlen als im Betrieb.
"""

from typing import List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

# Unterhalb dieser Kerzenzahl liefern BB(20) und EMA(21) nur NaN
MIN_CANDLES = 20


def ohlcv_to_df(ohlcv: List) -> pd.DataFrame:
    """CCXT-OHLCV-Liste [[ts_ms, o, h, l, c, v], ...] -> DataFrame."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt RSI(14), Bollinger(20, 2), EMA(9/21), VWAP und Volume-Delta.

    Gibt den DataFrame unverändert zurück, wenn zu wenige Kerzen vorliegen —
    dasselbe Verhalten wie die bisherigen Einzelimplementierungen.
    """
    if df is None or len(df) < MIN_CANDLES:
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


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average True Range als einzelner Wert (vereinfacht: High-Low-Mittel).

    Entspricht der Berechnung, die der Bot bislang inline nutzt; hier
    zentral, damit Feed, Exit-Logik und Backtest denselben Wert sehen.
    """
    if df is None or len(df) < period:
        return 0.0
    value = (df['high'] - df['low']).rolling(period).mean().iloc[-1]
    return float(value) if value == value and value > 0 else 0.0
