"""
Sentiment Factor (Phase 2)

Provides sentiment-based conviction to the multi-factor system.

Datenquelle: Alternative.me Fear & Greed Index (kostenlos, kein API-Key).

**Warum die Historie zählt.** Dieser Faktor ist im aktuellen Faktorenset der
einzige, der zuverlässig eine Richtung *und* einen hohen Score liefert:
`volatility_filter` und `volume_confirmation` sind richtungslos, `momentum`
liefert nur bei RSI > 55 bzw. < 45 eine Richtung, `breakout` nur bei echtem
Ausbruch, und `multi_timeframe_trend` liegt auf 5m fast immer am Score-Floor.
Der Bot handelt damit faktisch "kaufe bei Fear".

Wenn der Backtest den *heutigen* F&G-Wert auf 90 Tage Historie anwendet,
misst er also genau den Faktor falsch, der die Strategie steuert — und zwar
unauffällig falsch. Deshalb hält dieser Faktor die vollständige Tages-Historie
und schlägt den Wert zum jeweiligen Kerzendatum nach. Live ist das dieselbe
Codepfad-Logik: das Kerzendatum ist dann schlicht heute.

The factor acts differently depending on the regime:
- In "ranging" or "low_vol_chop": Extreme fear/greed are strong contrarian signals
- In "trending": Sentiment should mostly confirm the trend (not fight it)
- In "high_vol_event": Sentiment is de-weighted (too noisy)
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .base import Factor, FactorResult

logger = logging.getLogger(__name__)

HISTORY_URL = "https://api.alternative.me/fng/?limit=0&format=json"
CURRENT_URL = "https://api.alternative.me/fng/"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "fng_cache.json"


class SentimentFactor(Factor):
    """
    Sentiment scoring factor.

    Uses Fear & Greed Index as primary source for now.
    Score is normalized to 0-1 and interpreted depending on regime.
    """

    name = "sentiment"

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Wie oft die Historie erneuert wird. Der Index aktualisiert nur einmal
        # taeglich, haeufigere Abrufe bringen nichts.
        self.refresh_seconds = self.config.get("refresh_seconds", 6 * 3600)
        self.cache_path = Path(self.config.get("cache_path", DEFAULT_CACHE_PATH))
        # Wie weit zurueck ein aelterer Wert genutzt wird, wenn ein Tag fehlt
        self.max_staleness_days = self.config.get("max_staleness_days", 7)
        # Netzwerk abschaltbar — im Backtest reicht der Plattencache
        self.allow_network = self.config.get("allow_network", True)

        # date -> 0..100
        self._history: Dict[date, float] = {}
        self._loaded_at: Optional[datetime] = None

        # Weights for different regimes (can be tuned)
        self.regime_weights = self.config.get("regime_weights", {
            "ranging": 0.9,
            "low_vol_chop": 1.0,
            "trending": 0.6,
            "high_vol_event": 0.3,
            "event_driven": 0.4,
            "unknown": 0.7
        })

    # ----- Historie laden -------------------------------------------

    def _load_from_disk(self) -> bool:
        """Laedt die gecachte Historie. Macht Backtests offline reproduzierbar."""
        if not self.cache_path.exists():
            return False
        try:
            raw = json.loads(self.cache_path.read_text())
            entries = raw.get("history", {})
            self._history = {
                date.fromisoformat(k): float(v) for k, v in entries.items()
            }
            fetched = raw.get("fetched_at")
            self._loaded_at = datetime.fromisoformat(fetched) if fetched else None
            return bool(self._history)
        except Exception as e:
            logger.warning(f"F&G-Cache nicht lesbar ({self.cache_path}): {e}")
            return False

    def _save_to_disk(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({
                "fetched_at": datetime.now().isoformat(),
                "history": {d.isoformat(): v for d, v in sorted(self._history.items())},
            }, indent=1))
        except Exception as e:
            logger.warning(f"F&G-Cache nicht schreibbar ({self.cache_path}): {e}")

    def _fetch_history(self) -> bool:
        """Holt die vollstaendige Tages-Historie (limit=0)."""
        if not self.allow_network:
            return False
        try:
            response = requests.get(HISTORY_URL, timeout=20)
            payload = response.json()
        except Exception as e:
            logger.warning(f"F&G-Historie nicht abrufbar: {e}")
            return False

        entries = payload.get("data") or []
        history: Dict[date, float] = {}
        for entry in entries:
            try:
                day = datetime.utcfromtimestamp(int(entry["timestamp"])).date()
                history[day] = float(entry["value"])
            except (KeyError, TypeError, ValueError):
                continue

        if not history:
            return False

        self._history = history
        self._loaded_at = datetime.now()
        self._save_to_disk()
        logger.info(
            f"F&G-Historie geladen: {len(history)} Tage "
            f"({min(history)} bis {max(history)})"
        )
        return True

    def _fetch_current_only(self) -> bool:
        """
        Fallback: nur den aktuellen Wert holen.

        Greift, wenn der Historie-Endpunkt nicht erreichbar ist. Der Live-Betrieb
        laeuft damit weiter; ein Backtest wuerde dann aber wieder einen einzigen
        Wert auf die ganze Historie anwenden — deshalb meldet is_historical()
        in dem Fall False.
        """
        if not self.allow_network:
            return False
        try:
            response = requests.get(CURRENT_URL, timeout=10)
            payload = response.json()
            entries = payload.get("data") or []
            if not entries:
                return False
            value = float(entries[0]["value"])
            day = datetime.utcfromtimestamp(int(entries[0]["timestamp"])).date() \
                if entries[0].get("timestamp") else datetime.now().date()
        except Exception as e:
            logger.warning(f"F&G nicht abrufbar: {e}")
            return False

        self._history[day] = value
        self._loaded_at = datetime.now()
        return True

    def _ensure_history(self):
        """Sorgt dafuer, dass Daten vorliegen — Platte, dann Netz."""
        if self._history and self._loaded_at is not None:
            age = (datetime.now() - self._loaded_at).total_seconds()
            if age < self.refresh_seconds:
                return

        if not self._history and self._load_from_disk():
            age = ((datetime.now() - self._loaded_at).total_seconds()
                   if self._loaded_at else float("inf"))
            if age < self.refresh_seconds:
                return

        if not self._fetch_history():
            self._fetch_current_only()

    def set_history(self, history: Dict[date, float]):
        """Historie direkt setzen — fuer Tests und reproduzierbare Backtests."""
        self._history = dict(history)
        self._loaded_at = datetime.now()

    def is_historical(self) -> bool:
        """True, wenn mehr als ein Tag vorliegt — Voraussetzung fuer Backtests."""
        return len(self._history) > 1

    # ----- Nachschlagen ---------------------------------------------

    def _value_for(self, as_of: date) -> Optional[float]:
        """
        F&G-Wert fuer einen Tag. Fehlt der Tag, wird der naechstaeltere genommen
        (der Index ist der zuletzt bekannte Stand, nicht interpoliert).
        """
        if not self._history:
            return None

        exact = self._history.get(as_of)
        if exact is not None:
            return exact

        older = [d for d in self._history if d <= as_of]
        if not older:
            # Kerze liegt vor dem Beginn der Historie
            return None

        nearest = max(older)
        if (as_of - nearest).days > self.max_staleness_days:
            return None
        return self._history[nearest]

    @staticmethod
    def _as_of_from_candles(candles: pd.DataFrame) -> date:
        """Datum der letzten Kerze — im Live-Betrieb ist das heute."""
        try:
            if candles is not None and len(candles) and 'timestamp' in candles.columns:
                return pd.Timestamp(candles['timestamp'].iloc[-1]).date()
        except Exception:
            pass
        return datetime.now().date()

    # ----- Faktor ---------------------------------------------------

    def calculate(self, symbol: str, candles: pd.DataFrame,
                  current_price: float, **kwargs) -> Optional[FactorResult]:

        regime = kwargs.get("regime", "unknown")

        self._ensure_history()
        as_of = kwargs.get("as_of") or self._as_of_from_candles(candles)
        fng_value = self._value_for(as_of)

        if fng_value is None:
            return None

        # Get regime-specific weight
        regime_weight = self.regime_weights.get(regime, 0.7)

        # Interpretation logic
        if fng_value < 25:           # Extreme Fear
            if regime in ["ranging", "low_vol_chop"]:
                direction = "long"
                score = 0.9 * regime_weight
                reason = f"Extreme Fear ({fng_value}) → strong contrarian long signal"
            else:
                direction = "long"
                score = 0.65 * regime_weight
                reason = f"Extreme Fear ({fng_value}) → cautious long bias"

        elif fng_value > 75:         # Extreme Greed
            if regime in ["ranging", "low_vol_chop"]:
                direction = "short"
                score = 0.9 * regime_weight
                reason = f"Extreme Greed ({fng_value}) → strong contrarian short signal"
            else:
                direction = None
                score = 0.4
                reason = f"Extreme Greed ({fng_value}) → reduce conviction"

        elif fng_value < 45:         # Fear
            direction = "long"
            score = 0.65 * regime_weight
            reason = f"Fear zone ({fng_value}) → mild bullish sentiment"

        elif fng_value > 55:         # Greed
            direction = "short" if regime in ["ranging", "low_vol_chop"] else None
            score = 0.55 * regime_weight
            reason = f"Greed zone ({fng_value})"

        else:                        # Neutral
            direction = None
            score = 0.5
            reason = f"Neutral sentiment ({fng_value})"

        return FactorResult(
            name=self.name,
            score=score,
            confidence=0.75,   # Sentiment is useful but noisy
            direction=direction,
            reason=reason,
            metadata={
                "fear_and_greed": fng_value,
                "as_of": as_of.isoformat(),
                "regime": regime,
                "regime_weight": regime_weight
            }
        )
