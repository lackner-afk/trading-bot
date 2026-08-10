"""
Signal Aggregator - The new central brain of the trading bot.

Combines multiple FactorResults into a single high-quality TradeSignal
using a confluence scoring system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .factors.base import FactorResult


@dataclass
class TradeSignal:
    """Final aggregated trading signal."""
    symbol: str
    direction: str                    # "long" or "short"
    confidence: float                 # 0.0 - 1.0 overall conviction
    confluence_score: float           # 0.0 - 1.0 gewichteter Faktor-Mittelwert
    suggested_leverage: float
    take_profit: float
    stop_loss: float
    reason: str
    factor_breakdown: Dict[str, FactorResult] = field(default_factory=dict)
    timestamp: datetime = None
    # True = kein Entry, sondern die Aufforderung eine offene Gegenposition
    # zu schließen (Spot-Semantik für SHORT-Signale bei long-only)
    is_exit_signal: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def confluence_score_10(self) -> float:
        """
        Score auf einer 0–10-Skala — nur für die Anzeige.

        Die Entscheidungsmathematik läuft durchgängig auf 0–1. Genau die
        Vermischung beider Skalen war die Ursache dafür, dass der Bot nie
        gehandelt hat.
        """
        return self.confluence_score * 10.0


class SignalAggregator:
    """
    Combines multiple factors into one trading decision using a confluence system.

    This is the central component that will eventually replace the old
    MomentumStrategy / CryptoScalper logic.
    """

    # Der Score ist ein gewichteter Mittelwert von Faktor-Scores aus [0,1]
    # bei Gewichten mit Summe 1.0 — er kann 1.0 nie überschreiten. Ein
    # konfigurierter Schwellwert darüber wäre unerreichbar.
    MAX_POSSIBLE_SCORE = 1.0

    def __init__(self, config: Dict = None, constraints=None):
        self.config = config or {}
        self.min_confluence = self.config.get("min_confluence_score", 0.55)
        self.base_leverage = self.config.get("base_leverage", 8)
        self.min_technical_factors = self.config.get("min_technical_factors", 2)
        # Mindestvorsprung der Gewinnerrichtung gegenüber der Gegenrichtung
        self.direction_margin = self.config.get("direction_margin", 1.2)
        # Mindest-Konviktion der Faktoren, die tatsächlich eine Richtung nennen.
        #
        # Nötig, weil richtungslose Faktoren (volatility_filter,
        # volume_confirmation, macro_news_filter) voll in den Gesamtscore
        # einzahlen, aber nichts über die Richtung aussagen. Ohne diese Hürde
        # reicht ein ruhiger Markt mit "gesunder Volatilität" (Score 1.0) und
        # normalem Volumen (0.68), um die Schwelle zu reißen — obwohl der
        # Trend-Faktor bei 0.08 liegt und faktisch keine Richtung existiert.
        #
        # Sauberer wäre, Filter-Faktoren multiplikativ statt additiv wirken zu
        # lassen; das ändert aber die gesamte Kalibrierung und gehört an den
        # Backtest (P3), nicht in diesen Fix.
        self.min_directional_score = self.config.get("min_directional_score", 0.35)

        # Spot-/Leverage-Beschränkungen des Ziel-Venues
        from core.market_constraints import MarketConstraints
        self.constraints = constraints or MarketConstraints()

        self._validate_thresholds()

        # Base weights (will be adjusted by regime)
        self.base_weights = self.config.get("factor_weights", {
            "technical": 0.58,
            "sentiment": 0.25,
            "macro_news": 0.17
        })
        self.weights = self.base_weights.copy()

    def _validate_thresholds(self):
        """
        Fängt den Skalenfehler ab, der den Bot seine gesamte bisherige
        Laufzeit gekostet hat: `min_confluence_score` stand auf 3.5, während
        der Score konstruktionsbedingt nie über 1.0 gehen kann. Der Wert
        wurde dreimal "getunt", ohne dass jemand die Unerreichbarkeit bemerkte.
        """
        if self.min_confluence > self.MAX_POSSIBLE_SCORE:
            raise RuntimeError(
                f"min_confluence_score={self.min_confluence} ist unerreichbar. "
                f"Der Confluence-Score ist ein gewichteter Mittelwert von "
                f"Faktor-Scores aus [0,1] und liegt damit selbst in [0,1]. "
                f"Der Wert sieht nach der alten 0-10-Skala aus - teile ihn durch 10 "
                f"(z.B. 3.5 -> 0.35). Sinnvoller Bereich: 0.5 bis 0.75."
            )
        if self.min_confluence <= 0:
            raise RuntimeError(
                f"min_confluence_score={self.min_confluence} muss groesser 0 sein - "
                f"sonst wird jedes beliebige Faktor-Rauschen zu einem Trade."
            )

    def aggregate(self,
                  symbol: str,
                  current_price: float,
                  factor_results: List[FactorResult],
                  regime: Optional[str] = None,
                  regime_characteristics: Dict = None,
                  macro_risk_multiplier: float = 1.0) -> Optional[TradeSignal]:

        if not factor_results:
            return None

        # Apply regime-adjusted weights if regime information is provided
        if regime:
            adjusted_weights = self.get_regime_adjusted_weights(regime, regime_characteristics)
            self.weights = adjusted_weights

        # Categorize factors
        tech_results = [f for f in factor_results if any(x in f.name for x in ["trend", "momentum", "breakout", "volume", "volatility", "technical"])]
        sentiment_results = [f for f in factor_results if "sentiment" in f.name or f.name == "sentiment"]
        macro_results = [f for f in factor_results if any(x in f.name for x in ["macro", "news", "cpi", "event", "macro_news_filter"])]

        # Gewichteter Mittelwert NUR über tatsächlich vorhandene Kategorien.
        # Vorher lieferte _average_score() für eine leere Kategorie 0.0 und
        # zog den Gesamtscore stumm nach unten: fiel z.B. der Sentiment-Feed
        # aus (HTTP-Fehler -> Faktor liefert None), sank total_score um bis zu
        # 0.50 Gewichtsanteil, ohne dass irgendwo etwas geloggt wurde. Das war
        # eine zweite, unabhängige Ursache für stille Signal-Aushungerung.
        categories = [
            ("technical", tech_results, self.weights["technical"]),
            ("sentiment", sentiment_results, self.weights["sentiment"]),
            ("macro_news", macro_results, self.weights["macro_news"]),
        ]
        present = [(name, results, w) for name, results, w in categories if results]
        missing = [name for name, results, _ in categories if not results]

        weight_sum = sum(w for _, _, w in present)
        if weight_sum <= 0:
            print(f"[AGGREGATOR REJECT] {symbol} | Keine gewichtete Kategorie vorhanden")
            return None

        contributions = {
            name: self._average_score(results) * (w / weight_sum)
            for name, results, w in present
        }
        tech_score = contributions.get("technical", 0.0)
        sent_score = contributions.get("sentiment", 0.0)
        macro_score = contributions.get("macro_news", 0.0)
        total_score = sum(contributions.values())

        # Richtungs-Voting: nach score*confidence gewichtet, damit ein
        # unsicherer Faktor nicht dasselbe Stimmgewicht hat wie ein sicherer.
        # Faktoren ohne Richtung (Volatility, Volume, Macro) zahlen auf den
        # Score ein, stimmen aber nicht mit ab.
        long_vote = sum(f.score * f.confidence for f in factor_results if f.direction == "long")
        short_vote = sum(f.score * f.confidence for f in factor_results if f.direction == "short")

        def _reject(reason: str):
            print(f"[AGGREGATOR REJECT] {symbol} | {reason} | "
                  f"total={total_score:.3f} (min {self.min_confluence}) | "
                  f"tech={tech_score:.3f} sent={sent_score:.3f} macro={macro_score:.3f} | "
                  f"long_vote={long_vote:.3f} short_vote={short_vote:.3f} | "
                  f"tech_factors={len(tech_results)}"
                  + (f" | fehlende Kategorien: {','.join(missing)}" if missing else ""))
            return None

        if total_score < self.min_confluence:
            return _reject("Score unter Schwelle")

        # Mindestvorsprung verlangen — bei knappem Gleichstand ist die
        # Richtung Rauschen, nicht Signal.
        if long_vote >= short_vote * self.direction_margin and long_vote > 0:
            direction = "long"
        elif short_vote >= long_vote * self.direction_margin and short_vote > 0:
            direction = "short"
        else:
            return _reject("Richtung uneindeutig (kein Vorsprung)")

        # Require at least X technical factors to have decent conviction
        if len(tech_results) < self.min_technical_factors:
            return _reject(f"Zu wenige technische Faktoren: {len(tech_results)} < {self.min_technical_factors}")

        # Die Faktoren, die tatsächlich diese Richtung nennen, müssen selbst
        # Konviktion haben — sonst trägt allein das Filter-Rauschen den Score.
        directional = [f for f in factor_results if f.direction == direction]
        directional_score = self._average_score(directional)
        if directional_score < self.min_directional_score:
            return _reject(
                f"Richtungs-Konviktion zu schwach: {directional_score:.3f} < "
                f"{self.min_directional_score} (nur Filter-Faktoren tragen den Score)"
            )

        # Spot-Venue: SHORT ist nicht handelbar. Je nach Policy wird das
        # Signal verworfen oder als Exit für eine offene Long-Position
        # weitergereicht ("verkauf, was du hast").
        is_exit_signal = False
        if direction == "short" and not self.constraints.allow_short:
            if self.constraints.short_signal_policy == "ignore":
                return _reject("SHORT auf Spot-Venue nicht handelbar (policy=ignore)")
            is_exit_signal = True

        # confidence IST der gewichtete 0-1-Score. Vorher wurde hier durch 9.5
        # geteilt (Rest der alten 0-10-Skala), wodurch confidence nie über
        # 0.105 kam und am Gate in main.py scheiterte.
        confidence = min(total_score, 1.0)

        # Dynamic leverage based on confluence + regime + macro events
        leverage = self._calculate_leverage(confidence, regime) * macro_risk_multiplier
        leverage = self.constraints.clamp_leverage(leverage)

        # Simple but reasonable TP/SL (will be improved with ATR later)
        tp_pct = 0.016 + (confidence * 0.012)
        sl_pct = 0.008 + (confidence * 0.005)

        if direction == "long":
            take_profit = current_price * (1 + tp_pct)
            stop_loss = current_price * (1 - sl_pct)
        else:
            take_profit = current_price * (1 - tp_pct)
            stop_loss = current_price * (1 + sl_pct)

        # Anzeige weiterhin auf der vertrauten 0-10-Skala, die Entscheidung
        # darüber lief aber komplett auf 0-1.
        reason = (
            f"Confluence {total_score * 10:.1f}/10 | "
            f"Tech {tech_score * 10:.1f} | Sent {sent_score * 10:.1f} | Macro {macro_score * 10:.1f}"
        )
        if is_exit_signal:
            reason = "EXIT (Short-Signal auf Spot) | " + reason

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            confluence_score=total_score,
            suggested_leverage=leverage,
            take_profit=take_profit,
            stop_loss=stop_loss,
            reason=reason,
            factor_breakdown={f.name: f for f in factor_results},
            is_exit_signal=is_exit_signal
        )

    def _average_score(self, results: List[FactorResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def get_regime_adjusted_weights(self, regime_name: str, characteristics: Dict = None) -> Dict[str, float]:
        """
        Returns factor weights adjusted for the current regime.
        This is the core of Phase 4 adaptive weighting.
        """
        weights = self.base_weights.copy()
        chars = characteristics or {}

        vol_level = chars.get("volatility_level", 0.4)
        trend_strength = chars.get("trend_strength", 0.5)

        if regime_name == "trending":
            # Favor technical trend/momentum heavily
            weights["technical"] = 0.72
            weights["sentiment"] = 0.18
            weights["macro_news"] = 0.10

        elif regime_name == "ranging":
            # Sentiment and mean-reversion (via technical) become more important
            weights["technical"] = 0.48
            weights["sentiment"] = 0.35
            weights["macro_news"] = 0.17

        elif regime_name == "high_vol_event":
            # Reduce overall risk, rely more on macro and strong technical confirmation
            weights["technical"] = 0.50
            weights["sentiment"] = 0.15
            weights["macro_news"] = 0.35

        elif regime_name == "low_vol_chop":
            # Aggressive Test-Mode (A): heavily favor sentiment + macro, reduce technical penalty
            weights["technical"] = 0.35
            weights["sentiment"] = 0.50
            weights["macro_news"] = 0.15

        elif regime_name == "event_driven":
            weights["technical"] = 0.40
            weights["sentiment"] = 0.20
            weights["macro_news"] = 0.40

        # Slight volatility adjustment
        if vol_level > 0.7:
            weights["technical"] *= 0.9
            weights["macro_news"] *= 1.15
        elif vol_level < 0.2:
            weights["technical"] *= 0.85
            weights["sentiment"] *= 1.2

        # Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

    def set_weights_for_regime(self, regime_name: str, characteristics: Dict = None):
        """Apply regime-adjusted weights to the aggregator."""
        self.weights = self.get_regime_adjusted_weights(regime_name, characteristics)

    def _calculate_leverage(self, confidence: float, regime: Optional[str], macro_risk_multiplier: float = 1.0) -> float:
        # Auf einem Spot-Venue gibt es keinen Hebel. Der Early Return umgeht
        # insbesondere den max(2.0, ...)-Floor weiter unten, der sonst ein
        # Leverage-Minimum von 2x erzwingen würde.
        if self.constraints.spot_only:
            return 1.0

        base = self.base_leverage * (0.6 + confidence * 0.7)

        if regime == "high_vol_event":
            base *= 0.55
        elif regime == "low_vol_chop":
            base *= 0.45
        elif regime == "trending":
            base *= 1.2
        elif regime == "event_driven":
            base *= 0.65

        # Apply macro event risk reduction
        base *= macro_risk_multiplier

        return max(2.0, min(base, 18.0))
