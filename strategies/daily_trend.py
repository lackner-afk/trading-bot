"""
DailyTrendStrategy — langsame Trendfolge auf Tageskerzen, Long/Flat, Spot.

Warum diese Strategie (siehe docs/TREND_TAGESBASIS.md):
  * Bitpanda Fusion ist Spot: kein Short, kein Hebel, 0,25 % Gebühr je Seite,
    25 € Mindestorder. Bei 0,50 % Round-Trip ist jede 5-Minuten-Strategie
    rechnerisch chancenlos (docs/BACKTEST_BEFUNDE.md).
  * Der einzige Ansatz mit belastbarer Evidenz über viele Marktphasen ist
    langsame Trendfolge: im Aufwärtstrend investiert, sonst in Cash. Der
    robuste Befund der Literatur ist nicht "mehr Rendite", sondern "deutlich
    kleinerer Drawdown bei ähnlicher Rendite".
  * 10–25 Trades pro Jahr statt hunderte — die Gebühr wird zur Nebensache.

Regeln (alle auf ABGESCHLOSSENEN Tageskerzen, UTC):
  Einstieg:  Schlusskurs > SMA(n) * (1 + entry_buffer)
             und optional SMA steigt (SMA heute > SMA vor `slope_days` Tagen)
  Ausstieg:  Schlusskurs < SMA(n) * (1 - exit_buffer)
  Notstopp:  entry * (1 - max_loss_pct) — wird im Haupt-Loop gegen Live-Preise
             geprüft, damit ein Crash innerhalb eines Tages begrenzt bleibt.

Die Hysterese (entry_buffer / exit_buffer) verhindert das ständige Rein-Raus
um die SMA herum, das bei 0,50 % Round-Trip die Rendite auffrisst.

Die gleiche Funktion `compute_state` treibt Live-Bot und Backtester, damit
beide dieselben Signale sehen (die Freqtrade-Community dokumentiert, wie oft
Backtest und Live sonst auseinanderlaufen).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from .crypto_scalper import ScalperSignal, SignalType


@dataclass(frozen=True)
class TrendParams:
    """Parameter der Trendfolge. Unveränderlich, damit Backtest-Sweeps sie hashen können."""
    ma_days: int = 100
    entry_buffer_pct: float = 0.01
    exit_buffer_pct: float = 0.02
    require_rising_ma: bool = True
    slope_days: int = 10
    max_loss_pct: float = 0.08

    @classmethod
    def from_config(cls, config: Dict) -> "TrendParams":
        config = config or {}
        return cls(
            ma_days=int(config.get('ma_days', cls.ma_days)),
            entry_buffer_pct=float(config.get('entry_buffer_pct', cls.entry_buffer_pct)),
            exit_buffer_pct=float(config.get('exit_buffer_pct', cls.exit_buffer_pct)),
            require_rising_ma=bool(config.get('require_rising_ma', cls.require_rising_ma)),
            slope_days=int(config.get('slope_days', cls.slope_days)),
            max_loss_pct=float(config.get('max_loss_pct', cls.max_loss_pct)),
        )

    @property
    def warmup_bars(self) -> int:
        """So viele Kerzen braucht es, bis das erste Signal gültig ist."""
        return self.ma_days + (self.slope_days if self.require_rising_ma else 0)


def compute_state(candles: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    """
    Berechnet je Kerze, ob die Einstiegs- bzw. Ausstiegsbedingung erfüllt ist.

    Jede Zeile verwendet nur Daten bis einschließlich dieser Zeile — kein Blick
    in die Zukunft. Der Aufrufer ist dafür verantwortlich, die noch laufende
    Kerze vorher abzuschneiden.

    Rückgabe: DataFrame mit Spalten
      sma, entry_ok, exit_ok  (bool, NaN-Zeilen der Aufwärmphase sind False)
    """
    close = pd.to_numeric(candles['close'], errors='coerce')
    sma = close.rolling(params.ma_days, min_periods=params.ma_days).mean()

    entry_ok = close > sma * (1.0 + params.entry_buffer_pct)
    if params.require_rising_ma:
        rising = sma > sma.shift(params.slope_days)
        entry_ok = entry_ok & rising
    exit_ok = close < sma * (1.0 - params.exit_buffer_pct)

    out = pd.DataFrame({
        'sma': sma,
        'entry_ok': entry_ok.fillna(False).astype(bool),
        'exit_ok': exit_ok.fillna(False).astype(bool),
    }, index=candles.index)
    return out


class DailyTrendStrategy:
    """Erzeugt Long/Flat-Signale aus abgeschlossenen Tageskerzen."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('DailyTrend')
        self.params = TrendParams.from_config(self.config)
        self.pairs = list(self.config.get('pairs', ['BTC_EUR', 'ETH_EUR']))
        # Anteil des Eigenkapitals je Symbol. Der RiskManager kappt ohnehin bei
        # 20 % — mehr anzufragen bringt nur eine REDUCE_SIZE-Meldung.
        self.allocation_pct = float(self.config.get('allocation_pct', 0.20))
        # Mindestordergröße der Börse in EUR (Fusion: 25 € bei BTC-EUR).
        self.min_order_amount = float(self.config.get('min_order_amount', 25.0))
        self.interval_seconds = int(self.config.get('interval_seconds', 3600))

        self.signals_generated = 0
        self.exits_signalled = 0

    # ----- Signale -----

    def analyze(self, symbol: str, daily_candles: pd.DataFrame,
                current_price: float) -> Optional[ScalperSignal]:
        """
        Einstiegssignal, wenn die letzte ABGESCHLOSSENE Tageskerze die
        Trendbedingung erfüllt. Gibt None zurück, wenn nichts zu tun ist.
        """
        if daily_candles is None or len(daily_candles) < self.params.warmup_bars:
            return None
        if current_price is None or current_price <= 0:
            return None

        state = compute_state(daily_candles, self.params)
        last = state.iloc[-1]
        if not bool(last['entry_ok']):
            return None

        self.signals_generated += 1
        stop_loss = current_price * (1.0 - self.params.max_loss_pct)
        return ScalperSignal(
            signal_type=SignalType.LONG,
            symbol=symbol,
            price=current_price,
            confidence=1.0,
            reason=(f"Tagesschluss {float(daily_candles['close'].iloc[-1]):.2f} über "
                    f"SMA{self.params.ma_days} {float(last['sma']):.2f} "
                    f"(+{self.params.entry_buffer_pct:.0%} Puffer)"),
            take_profit=0.0,           # kein Ziel — der Trend läuft, bis er bricht
            stop_loss=stop_loss,
            suggested_leverage=1,      # Spot
            atr_value=0.0,
            timestamp=datetime.now(),
        )

    def check_trend_exit(self, symbol: str, daily_candles: pd.DataFrame) -> Tuple[bool, str]:
        """Trend gebrochen? Prüft die letzte abgeschlossene Tageskerze."""
        if daily_candles is None or len(daily_candles) < self.params.ma_days:
            return False, ''
        state = compute_state(daily_candles, self.params)
        last = state.iloc[-1]
        if bool(last['exit_ok']):
            self.exits_signalled += 1
            return True, (f"Trend gebrochen: Tagesschluss "
                          f"{float(daily_candles['close'].iloc[-1]):.2f} unter "
                          f"SMA{self.params.ma_days} {float(last['sma']):.2f} "
                          f"(-{self.params.exit_buffer_pct:.0%} Puffer)")
        return False, ''

    # ----- Sizing -----

    def target_notional(self, equity: float) -> float:
        """Gewünschte Positionsgröße in EUR vor Risk-Checks."""
        return max(0.0, equity * self.allocation_pct)

    def min_equity_for_trade(self, max_position_pct: float) -> float:
        """
        Ab welchem Eigenkapital eine Order überhaupt die Mindestgröße erreicht.
        Mit 20 %-Kappung und 25 € Mindestorder sind das 125 €.
        """
        cap = min(self.allocation_pct, max_position_pct)
        if cap <= 0:
            return float('inf')
        return self.min_order_amount / cap

    def get_statistics(self) -> Dict:
        return {
            'pairs': self.pairs,
            'params': self.params.__dict__,
            'signals_generated': self.signals_generated,
            'exits_signalled': self.exits_signalled,
        }
