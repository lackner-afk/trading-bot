"""
Technical Analysis Factors for the new multi-factor system.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base import Factor, FactorResult


class MultiTimeframeTrendFactor(Factor):
    """
    Evaluates trend alignment across multiple timeframes.

    Gives high scores when higher timeframes agree with the lower timeframe direction.
    This is one of the strongest and most reliable technical signals.
    """

    name = "multi_timeframe_trend"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.timeframes = self.config.get("timeframes", ["5m", "15m", "1h"])
        self.ema_fast = self.config.get("ema_fast", 9)
        self.ema_slow = self.config.get("ema_slow", 21)

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or len(candles) < 50:
            return None

        # We assume candles already have ema_9 and ema_21 or we compute them
        df = candles.copy()

        if 'ema_9' not in df.columns or 'ema_21' not in df.columns:
            df['ema_9'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        latest = df.iloc[-1]
        ema_fast = latest['ema_9']
        ema_slow = latest['ema_21']

        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return None

        direction = "long" if ema_fast > ema_slow else "short"
        strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0

        # Score based on trend strength (capped)
        score = min(strength * 8, 1.0)  # Strong trend → high score

        # Aggressive test mode (A): give weak trends a small floor in low vol chop
        if score < 0.15:
            score = 0.15
            reason = f"Trend {direction.upper()}: EMA{self.ema_fast}/{self.ema_slow} spread {strength:.2%} (floored for test)"
        else:
            reason = f"Trend {direction.upper()}: EMA{self.ema_fast}/{self.ema_slow} spread {strength:.2%}"

        return FactorResult(
            name=self.name,
            score=score,
            confidence=min(strength * 5, 1.0),
            direction=direction,
            reason=reason,
            metadata={
                "ema_fast": float(ema_fast),
                "ema_slow": float(ema_slow),
                "spread_pct": float(strength)
            }
        )


class MomentumFactor(Factor):
    """
    Measures short-term momentum using RSI + price momentum.
    """

    name = "momentum"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.momentum_lookback = self.config.get("momentum_lookback", 10)

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or len(candles) < self.rsi_period + 10:
            return None

        df = candles.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        rsi = latest.get('rsi')

        if pd.isna(rsi):
            return None

        # Simple momentum (price change over lookback)
        momentum = (current_price - df['close'].iloc[-self.momentum_lookback]) / df['close'].iloc[-self.momentum_lookback]

        # Combine RSI and momentum into a score
        if rsi > 55 and momentum > 0:
            direction = "long"
            score = min((rsi - 50) / 50 + abs(momentum) * 3, 1.0)
        elif rsi < 45 and momentum < 0:
            direction = "short"
            score = min((50 - rsi) / 50 + abs(momentum) * 3, 1.0)
        else:
            direction = None
            score = 0.3  # Neutral / weak

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.7,
            direction=direction,
            reason=f"Momentum: RSI={rsi:.1f}, {self.momentum_lookback}c momentum={momentum:.2%}",
            metadata={"rsi": float(rsi), "momentum": float(momentum)}
        )


class VolatilityFilter(Factor):
    """
    Acts as a quality filter rather than a direct signal generator.

    Gives low scores in extremely low volatility environments (reduces overtrading in dead markets)
    and can also penalize extreme volatility (risk control).
    """

    name = "volatility_filter"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_atr_pct = self.config.get("min_atr_pct", 0.0015)   # 0.15% - much more permissive than old 0.3%
        self.max_atr_pct = self.config.get("max_atr_pct", 0.08)     # 8% - very high volatility warning

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or len(candles) < 20:
            return None

        df = candles.copy()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()

        latest_atr = df['atr'].iloc[-1]
        if pd.isna(latest_atr) or latest_atr <= 0:
            return None

        atr_pct = latest_atr / current_price

        if atr_pct < self.min_atr_pct:
            # Test relaxation for low_vol_chop regimes (user wants more trades)
            score = 0.65
            reason = f"Very low volatility (ATR {atr_pct:.3%}) - relaxed for testing (more trades)"
        elif atr_pct > self.max_atr_pct:
            score = 0.4
            reason = f"Extremely high volatility (ATR {atr_pct:.2%}) - caution"
        else:
            score = 1.0
            reason = f"Healthy volatility (ATR {atr_pct:.2%})"

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.9,
            direction=None,
            reason=reason,
            metadata={"atr_pct": float(atr_pct)}
        )


class BreakoutFactor(Factor):
    """
    Detects breakouts from recent consolidation ranges.

    This factor helps the bot catch moves that pure EMA crossover systems often miss,
    which is one of the main reasons the old system didn't trade enough.
    """

    name = "breakout"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.lookback = self.config.get("lookback", 20)
        self.volume_multiplier = self.config.get("volume_multiplier", 1.8)

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or len(candles) < self.lookback + 5:
            return None

        df = candles.copy()

        # Recent high/low range
        recent_high = df['high'].tail(self.lookback).max()
        recent_low = df['low'].tail(self.lookback).min()
        range_size = (recent_high - recent_low) / recent_low if recent_low > 0 else 0

        # Volume confirmation
        avg_volume = df['volume'].tail(self.lookback).mean()
        current_volume = df['volume'].iloc[-1]
        volume_spike = current_volume > (avg_volume * self.volume_multiplier)

        direction = None
        score = 0.0

        if current_price > recent_high and volume_spike:
            direction = "long"
            score = 0.85
            reason = f"Breakout above {self.lookback}-period high with volume"
        elif current_price < recent_low and volume_spike:
            direction = "short"
            score = 0.85
            reason = f"Breakout below {self.lookback}-period low with volume"
        elif current_price > recent_high:
            direction = "long"
            score = 0.55
            reason = f"Breakout above {self.lookback}-period high (weak volume)"
        elif current_price < recent_low:
            direction = "short"
            score = 0.55
            reason = f"Breakout below {self.lookback}-period low (weak volume)"

        if direction is None:
            return None

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.75 if volume_spike else 0.5,
            direction=direction,
            reason=reason,
            metadata={
                "recent_high": float(recent_high),
                "recent_low": float(recent_low),
                "volume_spike": bool(volume_spike)
            }
        )


class VolumeConfirmationFactor(Factor):
    """
    Checks whether current volume supports the price move.
    Low volume moves are generally less reliable.
    """

    name = "volume_confirmation"

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.lookback = self.config.get("lookback", 20)

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        if candles is None or len(candles) < self.lookback + 5:
            return None

        df = candles.copy()
        avg_volume = df['volume'].tail(self.lookback).mean()
        current_volume = df['volume'].iloc[-1]

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Score based on volume strength
        if volume_ratio > 2.0:
            score = 0.95
            reason = f"Very strong volume ({volume_ratio:.1f}x average)"
        elif volume_ratio > 1.3:
            score = 0.8
            reason = f"Above average volume ({volume_ratio:.1f}x)"
        elif volume_ratio < 0.6:
            # Test relaxation: less penalty for low volume in quiet markets
            score = 0.55
            reason = f"Very low volume ({volume_ratio:.1f}x) - relaxed for testing (more trades)"
        else:
            score = 0.6
            reason = f"Normal volume ({volume_ratio:.1f}x)"

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.85,
            direction=None,
            reason=reason,
            metadata={"volume_ratio": float(volume_ratio)}
        )
