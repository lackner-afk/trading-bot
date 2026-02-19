"""
Crypto-Scalping-Strategie
High-Leverage Momentum/Breakout-Trading auf 1m-Kerzen
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class SignalType(Enum):
    """Signal-Typen"""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NONE = "none"


@dataclass
class ScalperSignal:
    """Ein Scalping-Signal"""
    signal_type: SignalType
    symbol: str
    price: float
    confidence: float  # 0-1
    reason: str
    take_profit: float
    stop_loss: float
    suggested_leverage: int
    atr_value: float = 0.0   # ATR zum Signalzeitpunkt (für Risk-Sizing in main.py)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CryptoScalper:
    """
    High-Frequency Scalping-Strategie

    Signale basierend auf:
    - RSI-Extremwerte + Bollinger-Touch + Volume-Spike → Mean-Reversion
    - Breakout über 15m-High/Low mit Volume → Momentum
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('CryptoScalper')

        # Parameter aus Config
        self.leverage = self.config.get('leverage', 20)
        self.max_leverage = min(self.config.get('max_leverage', 50), 50)
        self.pairs = self.config.get('pairs', ['BTC_EUR', 'ETH_EUR', 'SOL_EUR'])
        self.timeframe = self.config.get('timeframe', '1m')

        # Take-Profit / Stop-Loss
        self.take_profit_pct = self.config.get('take_profit', 0.015)   # 1.5%
        self.stop_loss_pct = self.config.get('stop_loss', 0.008)       # 0.8%
        self.trailing_stop_pct = self.config.get('trailing_stop', 0.005)  # 0.5%

        # Signal-Parameter (strenger für weniger, bessere Signale)
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.volume_spike_threshold = 3.0  # 3x Durchschnitt

        # Signal-Cooldown: 60 Sekunden pro Symbol nach Signal
        self._signal_cooldowns: Dict[str, datetime] = {}
        self.cooldown_seconds = 60

        # State
        self.active_signals: Dict[str, ScalperSignal] = {}
        self.signal_history: List[ScalperSignal] = []
        self.highest_prices: Dict[str, float] = {}  # Für Trailing-Stop

    def analyze(self, symbol: str, candles: pd.DataFrame,
               current_price: float) -> Optional[ScalperSignal]:
        """
        Analysiert Marktdaten und generiert Signal

        Args:
            symbol: Trading-Pair
            candles: DataFrame mit OHLCV + Indikatoren
            current_price: Aktueller Preis

        Returns:
            ScalperSignal oder None
        """
        if candles is None or len(candles) < 20:
            return None

        # Cooldown prüfen
        if symbol in self._signal_cooldowns:
            if datetime.now() < self._signal_cooldowns[symbol]:
                return None
            del self._signal_cooldowns[symbol]

        latest = candles.iloc[-1]

        # Prüfe auf Mean-Reversion Signal
        mr_signal = self._check_mean_reversion(symbol, candles, latest, current_price)
        if mr_signal:
            self._signal_cooldowns[symbol] = datetime.now() + timedelta(seconds=self.cooldown_seconds)
            self.signal_history.append(mr_signal)
            return mr_signal

        # Prüfe auf Breakout Signal
        bo_signal = self._check_breakout(symbol, candles, latest, current_price)
        if bo_signal:
            self._signal_cooldowns[symbol] = datetime.now() + timedelta(seconds=self.cooldown_seconds)
            self.signal_history.append(bo_signal)
            return bo_signal

        return None

    def _check_mean_reversion(self, symbol: str, candles: pd.DataFrame,
                              latest: pd.Series, current_price: float) -> Optional[ScalperSignal]:
        """
        Prüft auf Mean-Reversion Signale

        LONG: RSI < 30 + Bollinger Lower Touch + Volume Spike
        SHORT: RSI > 70 + Bollinger Upper Touch + Volume Spike
        """
        rsi = latest.get('rsi')
        bb_upper = latest.get('bb_upper')
        bb_lower = latest.get('bb_lower')
        volume = latest.get('volume', 0)
        avg_volume = candles['volume'].rolling(20).mean().iloc[-1]

        if rsi is None or bb_upper is None:
            return None

        volume_spike = volume > avg_volume * self.volume_spike_threshold

        # LONG Signal
        if rsi < self.rsi_oversold and current_price <= bb_lower and volume_spike:
            confidence = self._calculate_confidence(
                rsi_distance=self.rsi_oversold - rsi,
                bb_distance=abs(current_price - bb_lower) / current_price,
                volume_ratio=volume / avg_volume if avg_volume > 0 else 1
            )

            return ScalperSignal(
                signal_type=SignalType.LONG,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Mean-Reversion LONG: RSI={rsi:.1f}, BB-Lower Touch, Volume {volume/avg_volume:.1f}x",
                take_profit=current_price * (1 + self.take_profit_pct),
                stop_loss=current_price * (1 - self.stop_loss_pct),
                suggested_leverage=self._calculate_leverage(confidence)
            )

        # SHORT Signal
        if rsi > self.rsi_overbought and current_price >= bb_upper and volume_spike:
            confidence = self._calculate_confidence(
                rsi_distance=rsi - self.rsi_overbought,
                bb_distance=abs(current_price - bb_upper) / current_price,
                volume_ratio=volume / avg_volume if avg_volume > 0 else 1
            )

            return ScalperSignal(
                signal_type=SignalType.SHORT,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Mean-Reversion SHORT: RSI={rsi:.1f}, BB-Upper Touch, Volume {volume/avg_volume:.1f}x",
                take_profit=current_price * (1 - self.take_profit_pct),
                stop_loss=current_price * (1 + self.stop_loss_pct),
                suggested_leverage=self._calculate_leverage(confidence)
            )

        return None

    def _check_breakout(self, symbol: str, candles: pd.DataFrame,
                        latest: pd.Series, current_price: float) -> Optional[ScalperSignal]:
        """
        Prüft auf Breakout-Signale

        LONG: Preis durchbricht 15m-High mit >2x Volume
        SHORT: Preis durchbricht 15m-Low mit >2x Volume
        """
        # Approximiere 15m-Daten (15 1m-Kerzen)
        if len(candles) < 15:
            return None

        recent = candles.tail(15)
        high_15m = recent['high'].max()
        low_15m = recent['low'].min()

        volume = latest.get('volume', 0)
        avg_volume = candles['volume'].rolling(20).mean().iloc[-1]
        volume_spike = volume > avg_volume * self.volume_spike_threshold

        if not volume_spike:
            return None

        # Breakout nach oben
        if current_price > high_15m:
            confidence = self._calculate_breakout_confidence(
                price=current_price,
                level=high_15m,
                volume_ratio=volume / avg_volume if avg_volume > 0 else 1
            )

            # Größere TP/SL für Breakouts
            tp = self.take_profit_pct * 2
            sl = self.stop_loss_pct * 1.5

            return ScalperSignal(
                signal_type=SignalType.LONG,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Breakout LONG: Über 15m-High ({high_15m:.2f}), Volume {volume/avg_volume:.1f}x",
                take_profit=current_price * (1 + tp),
                stop_loss=current_price * (1 - sl),
                suggested_leverage=self._calculate_leverage(confidence * 0.8)  # Weniger Leverage bei Breakouts
            )

        # Breakout nach unten
        if current_price < low_15m:
            confidence = self._calculate_breakout_confidence(
                price=current_price,
                level=low_15m,
                volume_ratio=volume / avg_volume if avg_volume > 0 else 1
            )

            tp = self.take_profit_pct * 2
            sl = self.stop_loss_pct * 1.5

            return ScalperSignal(
                signal_type=SignalType.SHORT,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Breakout SHORT: Unter 15m-Low ({low_15m:.2f}), Volume {volume/avg_volume:.1f}x",
                take_profit=current_price * (1 - tp),
                stop_loss=current_price * (1 + sl),
                suggested_leverage=self._calculate_leverage(confidence * 0.8)
            )

        return None

    def _calculate_confidence(self, rsi_distance: float, bb_distance: float,
                              volume_ratio: float) -> float:
        """
        Berechnet Signal-Confidence

        Faktoren:
        - RSI-Abstand vom Extremwert
        - Bollinger-Band Durchdringung
        - Volume-Stärke
        """
        # RSI-Score (0-0.4): Je extremer, desto besser
        rsi_score = min(0.4, rsi_distance / 50)

        # BB-Score (0-0.3): Je weiter außerhalb, desto besser
        bb_score = min(0.3, bb_distance * 10)

        # Volume-Score (0-0.3): Je höher, desto besser
        volume_score = min(0.3, (volume_ratio - 1) / 5)

        total = rsi_score + bb_score + volume_score
        return min(0.95, max(0.3, total))

    def _calculate_breakout_confidence(self, price: float, level: float,
                                       volume_ratio: float) -> float:
        """Berechnet Breakout-Confidence"""
        # Wie weit über/unter dem Level?
        distance = abs(price - level) / level
        distance_score = min(0.4, distance * 20)

        # Volume-Stärke
        volume_score = min(0.4, (volume_ratio - 1) / 5)

        # Basis-Confidence für Breakouts (etwas niedriger)
        base = 0.2

        total = base + distance_score + volume_score
        return min(0.85, max(0.3, total))

    def _calculate_leverage(self, confidence: float) -> int:
        """
        Berechnet empfohlenen Leverage basierend auf Confidence

        Höhere Confidence = Höherer Leverage (bis max)
        """
        base_leverage = self.leverage
        max_lev = self.max_leverage

        # Skaliere Leverage mit Confidence
        # Bei 50% Confidence: base_leverage
        # Bei 100% Confidence: max_leverage
        if confidence >= 0.7:
            lev = int(base_leverage + (max_lev - base_leverage) * (confidence - 0.5) * 2)
        else:
            lev = int(base_leverage * confidence * 2)

        return max(5, min(max_lev, lev))

    def check_exit_conditions(self, symbol: str, entry_price: float,
                             current_price: float, side: str,
                             highest_since_entry: float = None) -> Tuple[bool, str]:
        """
        Prüft Exit-Bedingungen

        Args:
            symbol: Trading-Pair
            entry_price: Einstiegspreis
            current_price: Aktueller Preis
            side: 'long' oder 'short'
            highest_since_entry: Höchster Preis seit Entry (für Trailing-Stop)

        Returns:
            (should_exit, reason)
        """
        if side == 'long':
            # Take-Profit
            if current_price >= entry_price * (1 + self.take_profit_pct):
                return True, "Take-Profit erreicht"

            # Stop-Loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                return True, "Stop-Loss getriggert"

            # Trailing-Stop
            if highest_since_entry and current_price >= entry_price * (1 + self.trailing_stop_pct):
                trailing_stop = highest_since_entry * (1 - self.trailing_stop_pct)
                if current_price <= trailing_stop:
                    return True, f"Trailing-Stop getriggert ({trailing_stop:.2f})"

        else:  # short
            # Take-Profit
            if current_price <= entry_price * (1 - self.take_profit_pct):
                return True, "Take-Profit erreicht"

            # Stop-Loss
            if current_price >= entry_price * (1 + self.stop_loss_pct):
                return True, "Stop-Loss getriggert"

            # Trailing-Stop (für Short: niedrigster Preis)
            if highest_since_entry and current_price <= entry_price * (1 - self.trailing_stop_pct):
                trailing_stop = highest_since_entry * (1 + self.trailing_stop_pct)
                if current_price >= trailing_stop:
                    return True, f"Trailing-Stop getriggert ({trailing_stop:.2f})"

        return False, ""

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """
        Generiert Signal für Backtester

        Args:
            df: DataFrame mit Preisdaten
            idx: Aktueller Index

        Returns:
            'long', 'short', 'close' oder None
        """
        if idx < 20:
            return None

        row = df.iloc[idx]
        price = row['close']
        rsi = row.get('rsi')
        bb_upper = row.get('bb_upper')
        bb_lower = row.get('bb_lower')

        if rsi is None or bb_upper is None:
            return None

        # Volume-Analyse
        recent_volume = df.iloc[max(0, idx-20):idx]['volume']
        avg_volume = recent_volume.mean()
        current_volume = row['volume']
        volume_spike = current_volume > avg_volume * self.volume_spike_threshold

        # Mean-Reversion Signale (strengere Schwellen)
        if rsi < self.rsi_oversold and price <= bb_lower and volume_spike:
            return 'long'

        if rsi > self.rsi_overbought and price >= bb_upper and volume_spike:
            return 'short'

        # Breakout Signale
        if idx >= 15:
            recent = df.iloc[idx-15:idx]
            high_15 = recent['high'].max()
            low_15 = recent['low'].min()

            if price > high_15 and volume_spike:
                return 'long'
            if price < low_15 and volume_spike:
                return 'short'

        return None

    def get_active_signals(self) -> Dict[str, ScalperSignal]:
        """Gibt aktive Signale zurück"""
        return self.active_signals.copy()

    def get_signal_history(self, n: int = 50) -> List[ScalperSignal]:
        """Gibt letzte n Signale zurück"""
        return self.signal_history[-n:]

    def get_statistics(self) -> Dict:
        """Gibt Strategie-Statistiken zurück"""
        if not self.signal_history:
            return {'total_signals': 0}

        long_signals = [s for s in self.signal_history if s.signal_type == SignalType.LONG]
        short_signals = [s for s in self.signal_history if s.signal_type == SignalType.SHORT]
        avg_confidence = sum(s.confidence for s in self.signal_history) / len(self.signal_history)

        return {
            'total_signals': len(self.signal_history),
            'long_signals': len(long_signals),
            'short_signals': len(short_signals),
            'avg_confidence': avg_confidence,
            'leverage': self.leverage,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct
        }
