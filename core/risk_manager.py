"""
Risk-Management für Paper-Trading
Implementiert Stop-Loss, Drawdown-Limits, Position-Sizing und Kelly-Criterion
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class RiskAction(Enum):
    """Mögliche Risk-Aktionen"""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    CLOSE_ALL = "close_all"
    COOLDOWN = "cooldown"


@dataclass
class RiskCheck:
    """Ergebnis einer Risiko-Prüfung"""
    action: RiskAction
    reason: str
    suggested_size: Optional[float] = None
    cooldown_until: Optional[datetime] = None


class RiskManager:
    """
    Risiko-Management mit harten Regeln
    """

    # Harte Regeln - NICHT überschreibbar
    MAX_RISK_PER_TRADE = 0.02       # 2% des Portfolios
    MAX_DAILY_DRAWDOWN = 0.10       # 10% → Pause bis nächster Tag
    MAX_POSITION_SIZE = 0.20        # 20% des Portfolios pro Position
    MAX_LEVERAGE = 50               # Absolutes Leverage-Limit
    MAX_CONCURRENT_POSITIONS = 5
    COOLDOWN_AFTER_LOSSES = 3       # Anzahl Verluste für Cooldown
    COOLDOWN_DURATION = 300         # 5 Minuten Pause

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('RiskManager')
        self.config = config or {}

        # Überschreibbare Parameter (mit Limits)
        self.max_risk_per_trade = min(
            self.config.get('max_risk_per_trade', self.MAX_RISK_PER_TRADE),
            self.MAX_RISK_PER_TRADE
        )
        self.max_daily_drawdown = min(
            self.config.get('max_daily_drawdown', self.MAX_DAILY_DRAWDOWN),
            self.MAX_DAILY_DRAWDOWN
        )
        self.max_leverage = min(
            self.config.get('max_leverage', self.MAX_LEVERAGE),
            self.MAX_LEVERAGE
        )
        self.cooldown_seconds = self.config.get('cooldown_seconds', self.COOLDOWN_DURATION)

        # State
        self.cooldown_until: Optional[datetime] = None
        self.warnings: list = []

    def size_from_risk(self, equity: float, sl_distance_pct: float) -> float:
        """
        Berechnet Positionsgröße so dass genau max_risk_per_trade riskiert wird.

        Args:
            equity: Aktuelles Portfolio-Eigenkapital
            sl_distance_pct: |entry - sl_price| / entry_price (z.B. 0.012 = 1.2%)

        Returns:
            Empfohlene Positionsgröße in EUR
        """
        if sl_distance_pct <= 0:
            return equity * 0.05   # Fallback
        raw_size = (equity * self.max_risk_per_trade) / sl_distance_pct
        return min(raw_size, equity * self.MAX_POSITION_SIZE)

    def check_trade(self, portfolio_equity: float, position_size: float,
                   leverage: float, current_positions: int,
                   consecutive_losses: int, daily_drawdown: float,
                   sl_distance_pct: float = None) -> RiskCheck:
        """
        Prüft ob ein Trade den Risiko-Regeln entspricht

        Args:
            portfolio_equity: Aktuelles Portfolio-Eigenkapital
            position_size: Gewünschte Positionsgröße in EUR
            leverage: Gewünschter Leverage
            current_positions: Anzahl aktuell offener Positionen
            consecutive_losses: Anzahl aufeinanderfolgender Verluste
            daily_drawdown: Aktueller täglicher Drawdown (0.0 - 1.0)
            sl_distance_pct: Optionaler ATR-basierter SL-Abstand als Bruchteil (z.B. 0.012)

        Returns:
            RiskCheck mit Aktion und Begründung
        """

        # 1. Cooldown aktiv?
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds
            return RiskCheck(
                action=RiskAction.COOLDOWN,
                reason=f"Cooldown aktiv - noch {remaining}s warten",
                cooldown_until=self.cooldown_until
            )

        # 2. Täglicher Drawdown-Limit
        if daily_drawdown >= self.max_daily_drawdown:
            return RiskCheck(
                action=RiskAction.CLOSE_ALL,
                reason=f"Täglicher Drawdown ({daily_drawdown:.1%}) erreicht Limit ({self.max_daily_drawdown:.1%})"
            )

        # 3. Warnung bei hohem Drawdown
        if daily_drawdown >= 0.05:
            self._add_warning(f"Drawdown bei {daily_drawdown:.1%}")

        # 4. Consecutive Losses → Cooldown
        if consecutive_losses >= self.COOLDOWN_AFTER_LOSSES:
            self.cooldown_until = datetime.now() + timedelta(seconds=self.cooldown_seconds)
            return RiskCheck(
                action=RiskAction.COOLDOWN,
                reason=f"{consecutive_losses} Verluste in Folge - {self.cooldown_seconds}s Pause",
                cooldown_until=self.cooldown_until
            )

        # 5. Max Positionen
        if current_positions >= self.MAX_CONCURRENT_POSITIONS:
            return RiskCheck(
                action=RiskAction.BLOCK,
                reason=f"Maximum Positionen ({self.MAX_CONCURRENT_POSITIONS}) erreicht"
            )

        # 6. Leverage-Limit
        if leverage > self.max_leverage:
            return RiskCheck(
                action=RiskAction.BLOCK,
                reason=f"Leverage ({leverage}x) überschreitet Limit ({self.max_leverage}x)"
            )

        # 7. Position-Size-Limit
        max_size = portfolio_equity * self.MAX_POSITION_SIZE
        if position_size > max_size:
            return RiskCheck(
                action=RiskAction.REDUCE_SIZE,
                reason=f"Positionsgröße reduziert auf {self.MAX_POSITION_SIZE:.0%} des Portfolios",
                suggested_size=max_size
            )

        # 8. Risk per Trade
        if sl_distance_pct and sl_distance_pct > 0:
            risk_amount = position_size * sl_distance_pct
        else:
            risk_amount = position_size * leverage * 0.01  # Fallback: Annahme 1% Stop-Loss
        max_risk = portfolio_equity * self.max_risk_per_trade
        if risk_amount > max_risk:
            if sl_distance_pct and sl_distance_pct > 0:
                adjusted_size = max_risk / sl_distance_pct
            else:
                adjusted_size = max_risk / (leverage * 0.01)
            return RiskCheck(
                action=RiskAction.REDUCE_SIZE,
                reason=f"Risiko pro Trade begrenzt auf {self.max_risk_per_trade:.1%}",
                suggested_size=adjusted_size
            )

        # 9. Dynamische Größenreduktion bei Drawdown
        if daily_drawdown > 0.03:  # Bei >3% Drawdown
            reduction_factor = 1.0 - (daily_drawdown / self.max_daily_drawdown)
            adjusted_size = position_size * reduction_factor
            if adjusted_size < position_size:
                return RiskCheck(
                    action=RiskAction.REDUCE_SIZE,
                    reason=f"Größe reduziert wegen Drawdown ({daily_drawdown:.1%})",
                    suggested_size=adjusted_size
                )

        return RiskCheck(action=RiskAction.ALLOW, reason="Trade erlaubt")

    def calculate_position_size(self, portfolio_equity: float, win_rate: float,
                                avg_win: float, avg_loss: float,
                                leverage: float = 1.0) -> float:
        """
        Berechnet optimale Positionsgröße nach Kelly-Criterion (Halb-Kelly)

        Args:
            portfolio_equity: Portfolio-Eigenkapital
            win_rate: Historische Gewinnrate (0.0 - 1.0)
            avg_win: Durchschnittlicher Gewinn
            avg_loss: Durchschnittlicher Verlust (positiver Wert)
            leverage: Geplanter Leverage

        Returns:
            Empfohlene Positionsgröße in USD
        """
        if win_rate <= 0 or avg_loss <= 0:
            # Fallback: Feste Größe von 5%
            return portfolio_equity * 0.05

        # Kelly-Formel: f* = (p * b - q) / b
        # p = Gewinnwahrscheinlichkeit
        # q = Verlustwahrscheinlichkeit (1-p)
        # b = Gewinn/Verlust-Verhältnis

        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1

        kelly = (p * b - q) / b if b > 0 else 0

        # Halb-Kelly für mehr Sicherheit
        half_kelly = kelly / 2

        # Begrenzen auf MAX_POSITION_SIZE
        half_kelly = max(0, min(half_kelly, self.MAX_POSITION_SIZE))

        # Position Size berechnen
        position_size = portfolio_equity * half_kelly

        # Leverage berücksichtigen
        position_size = position_size / leverage if leverage > 1 else position_size

        return max(0, position_size)

    def check_stop_loss(self, entry_price: float, current_price: float,
                       side: str, stop_loss_pct: float) -> Tuple[bool, float]:
        """
        Prüft ob Stop-Loss getriggert wurde

        Returns:
            (triggered, trigger_price)
        """
        if side == 'long':
            stop_price = entry_price * (1 - stop_loss_pct)
            triggered = current_price <= stop_price
        else:
            stop_price = entry_price * (1 + stop_loss_pct)
            triggered = current_price >= stop_price

        return triggered, stop_price

    def check_take_profit(self, entry_price: float, current_price: float,
                         side: str, take_profit_pct: float) -> Tuple[bool, float]:
        """
        Prüft ob Take-Profit getriggert wurde

        Returns:
            (triggered, trigger_price)
        """
        if side == 'long':
            tp_price = entry_price * (1 + take_profit_pct)
            triggered = current_price >= tp_price
        else:
            tp_price = entry_price * (1 - take_profit_pct)
            triggered = current_price <= tp_price

        return triggered, tp_price

    def calculate_trailing_stop(self, entry_price: float, highest_price: float,
                               side: str, trailing_pct: float) -> float:
        """
        Berechnet Trailing-Stop-Preis

        Args:
            entry_price: Einstiegspreis
            highest_price: Höchster Preis seit Entry (für Long) / Niedrigster für Short
            side: 'long' oder 'short'
            trailing_pct: Trailing-Stop Prozentsatz

        Returns:
            Trailing-Stop-Preis
        """
        if side == 'long':
            return highest_price * (1 - trailing_pct)
        else:
            return highest_price * (1 + trailing_pct)

    def get_exposure(self, positions: Dict, portfolio_equity: float) -> float:
        """
        Berechnet Gesamt-Exposure des Portfolios

        Returns:
            Exposure als Faktor des Eigenkapitals (z.B. 2.5 = 250%)
        """
        if portfolio_equity <= 0:
            return 0.0

        total_exposure = sum(
            pos.size * pos.leverage
            for pos in positions.values()
        )

        return total_exposure / portfolio_equity

    def _add_warning(self, message: str):
        """Fügt Warnung hinzu (max 10 behalten)"""
        self.warnings.append({
            'time': datetime.now(),
            'message': message
        })
        self.warnings = self.warnings[-10:]
        self.logger.warning(message)

    def get_warnings(self) -> list:
        """Gibt aktuelle Warnungen zurück"""
        # Nur Warnungen der letzten Stunde
        cutoff = datetime.now() - timedelta(hours=1)
        return [w for w in self.warnings if w['time'] > cutoff]

    def get_metrics(self, portfolio_equity: float, daily_drawdown: float,
                   sharpe: float) -> Dict:
        """
        Gibt Risk-Metriken zurück

        Returns:
            Dict mit allen relevanten Risk-Metriken
        """
        return {
            'equity': portfolio_equity,
            'daily_drawdown': daily_drawdown,
            'daily_drawdown_limit': self.max_daily_drawdown,
            'drawdown_utilization': daily_drawdown / self.max_daily_drawdown if self.max_daily_drawdown > 0 else 0,
            'sharpe_ratio': sharpe,
            'cooldown_active': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'warnings': len(self.get_warnings()),
            'status': self._get_status(daily_drawdown, sharpe)
        }

    def _get_status(self, daily_drawdown: float, sharpe: float) -> str:
        """Ermittelt Risiko-Status"""
        if daily_drawdown >= self.max_daily_drawdown:
            return 'CRITICAL'
        elif daily_drawdown >= 0.05:
            return 'WARNING'
        elif daily_drawdown >= 0.03:
            return 'CAUTION'
        else:
            return 'OK'

    def reset_cooldown(self):
        """Setzt Cooldown manuell zurück"""
        self.cooldown_until = None
