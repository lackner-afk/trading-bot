"""
EMA-Cross Momentum-Strategie
Tradet EMA 9/21 Crossovers mit RSI-Filter
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from .crypto_scalper import SignalType, ScalperSignal


class MomentumStrategy:
    """
    EMA-Cross Momentum-Strategie

    Signale basierend auf:
    - EMA 9 kreuzt EMA 21 (Trendwechsel)
    - RSI-Filter verhindert Trades gegen den Trend
    - Cooldown pro Symbol vermeidet Overtrading

    Backtest-Ergebnis (90 Tage Bärenmarkt):
    +4.2% Return, 60% Win Rate, 6.3% Max DD, Sharpe 1.14
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('Momentum')

        # Parameter
        self.leverage = self.config.get('leverage', 10)
        self.max_leverage = min(self.config.get('max_leverage', 20), 50)
        self.pairs = self.config.get('pairs', ['BTC_EUR', 'ETH_EUR', 'SOL_EUR', 'XRP_EUR'])

        # TP/SL (validiert im Backtest) — Fallback wenn kein ATR verfügbar
        self.take_profit_pct = self.config.get('take_profit', 0.015)   # 1.5%
        self.stop_loss_pct = self.config.get('stop_loss', 0.008)       # 0.8%
        self.trailing_stop_pct = self.config.get('trailing_stop', 0.005)  # 0.5%

        # ATR-Parameter für dynamische SL/TP
        self.atr_period = self.config.get('atr_period', 14)
        self.sl_atr_multiplier = self.config.get('sl_atr_multiplier', 1.8)
        self.tp_atr_multiplier = self.config.get('tp_atr_multiplier', 3.5)
        self.trailing_atr_multiplier = self.config.get('trailing_atr_multiplier', 1.2)
        # ATR-Cache pro Symbol (für check_exit_conditions)
        self._last_atr: Dict[str, float] = {}

        # EMA-Cross Parameter
        self.rsi_long_threshold = self.config.get('rsi_long_threshold', 40)
        self.rsi_short_threshold = self.config.get('rsi_short_threshold', 60)

        # Cooldown: 5 Minuten pro Symbol nach Signal
        self._signal_cooldowns: Dict[str, datetime] = {}
        self.cooldown_seconds = self.config.get('cooldown_seconds', 300)

        # EMA-Cross Bestätigung: vorherige Kerze muss entgegengesetzt sein
        self._prev_ema_state: Dict[str, str] = {}  # symbol -> 'bullish' | 'bearish'

        # State
        self.signal_history: List[ScalperSignal] = []
        self.highest_prices: Dict[str, float] = {}

    def analyze(self, symbol: str, candles: pd.DataFrame,
                current_price: float) -> Optional[ScalperSignal]:
        """
        Analysiert EMA-Cross und RSI für Momentum-Signal

        Args:
            symbol: Trading-Pair
            candles: DataFrame mit OHLCV + Indikatoren (ema_9, ema_21, rsi)
            current_price: Aktueller Preis

        Returns:
            ScalperSignal oder None
        """
        if candles is None or len(candles) < 25:
            return None

        # Cooldown prüfen
        if symbol in self._signal_cooldowns:
            if datetime.now() < self._signal_cooldowns[symbol]:
                return None
            del self._signal_cooldowns[symbol]

        latest = candles.iloc[-1]
        prev = candles.iloc[-2]

        ema_9 = latest.get('ema_9')
        ema_21 = latest.get('ema_21')
        rsi = latest.get('rsi')
        prev_ema_9 = prev.get('ema_9')
        prev_ema_21 = prev.get('ema_21')

        if ema_9 is None or ema_21 is None or rsi is None:
            return None
        if prev_ema_9 is None or prev_ema_21 is None:
            return None

        # EMA-Zustand (bullish = EMA9 über EMA21)
        currently_bullish = ema_9 > ema_21
        was_bullish = prev_ema_9 > prev_ema_21

        # Frischer Crossover ODER anhaltend starke EMA-Zustand + RSI-Bestätigung
        bullish_cross = currently_bullish and not was_bullish
        bearish_cross = not currently_bullish and was_bullish
        # EMA-Zustand-Signal: EMA klar getrennt (RSI-Filter kommt weiter unten)
        ema_spread = abs(ema_9 - ema_21) / ema_21 if ema_21 > 0 else 0
        strong_bullish_state = currently_bullish and ema_spread > 0.001
        strong_bearish_state = not currently_bullish and ema_spread > 0.001

        # ATR berechnen für dynamische SL/TP
        atr = self._calculate_atr(candles, self.atr_period)
        use_atr = not (pd.isna(atr) or atr <= 0)
        if use_atr:
            self._last_atr[symbol] = atr

        signal = None

        # LONG: EMA9 über EMA21 (Crossover ODER starker Zustand) + RSI nicht überkauft
        if (bullish_cross or strong_bullish_state) and rsi < self.rsi_long_threshold:
            confidence = self._calculate_confidence(rsi, ema_9, ema_21, 'long')
            if use_atr:
                sl_price = current_price - (atr * self.sl_atr_multiplier)
                tp_price = current_price + (atr * self.tp_atr_multiplier)
            else:
                sl_price = current_price * (1 - self.stop_loss_pct)
                tp_price = current_price * (1 + self.take_profit_pct)
            reason = "Crossover" if bullish_cross else "EMA-Trend"
            signal = ScalperSignal(
                signal_type=SignalType.LONG,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Momentum LONG ({reason}): EMA9>{ema_9:.1f} EMA21={ema_21:.1f}, RSI={rsi:.1f}",
                take_profit=tp_price,
                stop_loss=sl_price,
                suggested_leverage=self._calculate_leverage(confidence),
                atr_value=atr if use_atr else 0.0
            )

        # SHORT: EMA9 unter EMA21 (Crossover ODER starker Zustand) + RSI nicht überverkauft
        elif (bearish_cross or strong_bearish_state) and rsi > self.rsi_short_threshold:
            confidence = self._calculate_confidence(rsi, ema_9, ema_21, 'short')
            if use_atr:
                sl_price = current_price + (atr * self.sl_atr_multiplier)
                tp_price = current_price - (atr * self.tp_atr_multiplier)
            else:
                sl_price = current_price * (1 + self.stop_loss_pct)
                tp_price = current_price * (1 - self.take_profit_pct)
            reason = "Crossover" if bearish_cross else "EMA-Trend"
            signal = ScalperSignal(
                signal_type=SignalType.SHORT,
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                reason=f"Momentum SHORT ({reason}): EMA9<{ema_9:.1f} EMA21={ema_21:.1f}, RSI={rsi:.1f}",
                take_profit=tp_price,
                stop_loss=sl_price,
                suggested_leverage=self._calculate_leverage(confidence),
                atr_value=atr if use_atr else 0.0
            )

        if signal:
            self._signal_cooldowns[symbol] = datetime.now() + timedelta(seconds=self.cooldown_seconds)
            self.signal_history.append(signal)
            self.logger.info(f"{signal.reason} @ ${current_price:.2f} (Conf: {signal.confidence:.0%})")

        return signal

    def _calculate_atr(self, candles: pd.DataFrame, period: int) -> float:
        """Berechnet Average True Range (ATR) über gegebene Periode"""
        high = candles['high']
        low = candles['low']
        close = candles['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _calculate_confidence(self, rsi: float, ema_9: float, ema_21: float,
                              direction: str) -> float:
        """
        Berechnet Confidence basierend auf:
        - RSI-Stärke: Je weiter vom Neutral (50), desto besser
        - EMA-Abstand: Je stärker der Cross, desto besser
        """
        # RSI-Score (0-0.5): Abstand vom Neutral-Bereich
        if direction == 'long':
            rsi_score = min(0.5, (40 - rsi) / 80)  # RSI 0 → 0.5, RSI 40 → 0
        else:
            rsi_score = min(0.5, (rsi - 60) / 80)  # RSI 100 → 0.5, RSI 60 → 0

        # EMA-Cross-Stärke (0-0.4): Wie weit EMAs auseinander
        ema_spread = abs(ema_9 - ema_21) / ema_21 if ema_21 > 0 else 0
        ema_score = min(0.4, ema_spread * 50)

        # Basis
        total = 0.2 + rsi_score + ema_score
        return min(0.95, max(0.3, total))

    def _calculate_leverage(self, confidence: float) -> int:
        """Konservativer Leverage für Momentum-Trades"""
        base = self.leverage
        max_lev = self.max_leverage

        # Skaliere linear: Conf 0.3 → base/2, Conf 0.9 → max
        lev = int(base * 0.5 + (max_lev - base * 0.5) * (confidence - 0.3) / 0.6)
        return max(10, min(max_lev, lev))

    def check_exit_conditions(self, symbol: str, entry_price: float,
                              current_price: float, side: str,
                              highest_since_entry: float = None,
                              stop_loss_price: float = None,
                              take_profit_price: float = None) -> Tuple[bool, str]:
        """Prüft Exit-Bedingungen — nutzt absolute Preise wenn übergeben, sonst %-Fallback"""
        atr = self._last_atr.get(symbol)

        if side == 'long':
            # Take-Profit
            if take_profit_price is not None:
                if current_price >= take_profit_price:
                    return True, "Take-Profit erreicht"
            elif current_price >= entry_price * (1 + self.take_profit_pct):
                return True, "Take-Profit erreicht"

            # Stop-Loss
            if stop_loss_price is not None:
                if current_price <= stop_loss_price:
                    return True, "Stop-Loss getriggert"
            elif current_price <= entry_price * (1 - self.stop_loss_pct):
                return True, "Stop-Loss getriggert"

            # Trailing-Stop (ATR-basiert wenn verfügbar)
            if highest_since_entry:
                if atr and atr > 0:
                    trailing_stop = highest_since_entry - (atr * self.trailing_atr_multiplier)
                else:
                    trailing_stop = highest_since_entry * (1 - self.trailing_stop_pct)
                if current_price >= entry_price * (1 + self.trailing_stop_pct) and current_price <= trailing_stop:
                    return True, f"Trailing-Stop ({trailing_stop:.2f})"

        else:  # short
            # Take-Profit
            if take_profit_price is not None:
                if current_price <= take_profit_price:
                    return True, "Take-Profit erreicht"
            elif current_price <= entry_price * (1 - self.take_profit_pct):
                return True, "Take-Profit erreicht"

            # Stop-Loss
            if stop_loss_price is not None:
                if current_price >= stop_loss_price:
                    return True, "Stop-Loss getriggert"
            elif current_price >= entry_price * (1 + self.stop_loss_pct):
                return True, "Stop-Loss getriggert"

            # Trailing-Stop (ATR-basiert wenn verfügbar, highest_since_entry = niedrigster Preis für Short)
            if highest_since_entry:
                if atr and atr > 0:
                    trailing_stop = highest_since_entry + (atr * self.trailing_atr_multiplier)
                else:
                    trailing_stop = highest_since_entry * (1 + self.trailing_stop_pct)
                if current_price <= entry_price * (1 - self.trailing_stop_pct) and current_price >= trailing_stop:
                    return True, f"Trailing-Stop ({trailing_stop:.2f})"

        return False, ""

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """
        Signal-Generator für Backtester

        Nutzt EMA-Zustand + RSI-Filter (statt Crossover-Moment),
        da der Backtester stündliche Kerzen nutzt und exakte
        Crossover-Momente dort zu selten auftreten.
        """
        if idx < 21:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        rsi = row.get('rsi')
        ema_9 = row.get('ema_9')
        ema_21 = row.get('ema_21')
        prev_ema_9 = prev.get('ema_9')
        prev_ema_21 = prev.get('ema_21')

        if rsi is None or ema_9 is None or prev_ema_9 is None:
            return None

        # Auf stündlichen Daten: EMA-Zustand + RSI-Filter
        # (Live-Bot nutzt exakte Crossover-Detection auf 1m-Kerzen)
        if rsi < self.rsi_long_threshold and ema_9 > ema_21:
            return 'long'

        if rsi > self.rsi_short_threshold and ema_9 < ema_21:
            return 'short'

        return None

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
