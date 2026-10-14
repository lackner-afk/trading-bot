"""
Tages-Trendfilter (Test T1, docs/BACKTEST_BEFUNDE.md): eine Coin halten, solange
mindestens 3 von 5 einfachen Trendregeln auf abgeschlossenen Tageskerzen
Aufwärtstrend sagen — sonst draußen bleiben.

Regeln (vorab festgelegt, nicht optimiert):
  sma50 / sma100 / sma200   Schlusskurs über dem gleitenden Durchschnitt
  mom30 / mom90             Schlusskurs höher als vor 30 / 90 Tagen

Der Wert liegt im Crash-Schutz, nicht im Timing: 2022 verlor Halten 65–76 %,
die Mehrheitsregel 21–37 %. Keine Einzelregel war überall die beste, darum die
Mehrheit. tools/backtest_daily.py rechnet mit denselben Funktionen.

Kurse: öffentliche Kraken-Tageskerzen in EUR (kein Key nötig). Fusion liefert
die Ausführungspreise wie bei jeder anderen Strategie.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

RULES = ['sma50', 'sma100', 'sma200', 'mom30', 'mom90']
MIN_VOTES = 3
# Längster Rückblick + Puffer; Kraken liefert bis zu 720 Tageskerzen
DAILY_HISTORY = 400


def rule_signal(close: pd.Series, rule: str) -> pd.Series:
    """True = die Regel will am Tagesschluss halten."""
    if rule.startswith('sma'):
        n = int(rule[3:])
        return close > close.rolling(n).mean()
    if rule.startswith('mom'):
        n = int(rule[3:])
        return close > close.shift(n)
    raise ValueError(rule)


def vote_count(close: pd.Series) -> pd.Series:
    """Anzahl der Regeln, die je Tag halten wollen (0–5)."""
    return sum(rule_signal(close, r).astype(int) for r in RULES)


@dataclass
class TrendVote:
    symbol: str
    votes: int
    hold: bool
    candle_date: datetime   # Datum der zuletzt abgeschlossenen Tageskerze (UTC)
    close: float
    rules: Dict[str, bool]


class DailyTrendStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('DailyTrend')
        self.min_votes = int(self.config.get('min_votes', MIN_VOTES))
        self.last_votes: Dict[str, TrendVote] = {}
        self._exchange = None

    @staticmethod
    def kraken_symbol(symbol: str) -> str:
        """BTC_EUR -> BTC/EUR"""
        return symbol.replace('_', '/')

    async def fetch_daily_closes(self, symbol: str) -> Optional[pd.Series]:
        """Abgeschlossene Tageskerzen (Schlusskurse) von Kraken, ältestes zuerst."""
        import ccxt.async_support as ccxt
        if self._exchange is None:
            self._exchange = ccxt.kraken({'enableRateLimit': True})
        rows = await self._exchange.fetch_ohlcv(
            self.kraken_symbol(symbol), '1d', limit=DAILY_HISTORY
        )
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset='ts', keep='last')
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
        # Die heutige Kerze ist noch offen — nur auf abgeschlossenen entscheiden
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        df = df[df['date'] < today]
        return df.set_index('date')['close'].astype(float)

    def evaluate(self, symbol: str, close: pd.Series) -> Optional[TrendVote]:
        if close is None or len(close) <= max(int(r[3:]) for r in RULES):
            self.logger.warning(f"{symbol}: zu wenig Tageskerzen ({0 if close is None else len(close)})")
            return None
        rules = {r: bool(rule_signal(close, r).iloc[-1]) for r in RULES}
        votes = sum(rules.values())
        vote = TrendVote(
            symbol=symbol,
            votes=votes,
            hold=votes >= self.min_votes,
            candle_date=close.index[-1].to_pydatetime(),
            close=float(close.iloc[-1]),
            rules=rules,
        )
        self.last_votes[symbol] = vote
        return vote

    async def close(self):
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
