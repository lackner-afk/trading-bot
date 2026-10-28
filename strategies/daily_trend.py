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

Zweite Regel (`rule: donchian`, Panda-9 seit 28.10.2026): Donchian-Ausbruch —
Kauf über dem 20-Tage-Hoch, Verkauf unter dem 10-Tage-Tief. Gleicher Ablauf,
nur eine andere Trendfamilie, damit beide Paper-Bots Verschiedenes testen.

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


def donchian_hold(high: pd.Series, low: pd.Series, close: pd.Series,
                  entry: int, exit: int) -> pd.Series:
    """
    Donchian-Ausbruch (Turtle-Regel): einsteigen, wenn der Schlusskurs über dem
    höchsten Hoch der vorigen `entry` Tage liegt; halten, bis er unter das
    tiefste Tief der vorigen `exit` Tage fällt. True = am Tagesschluss halten.

    Anders als die Mehrheitsregel hängt der Zustand vom Weg ab — dieselbe
    Kerze kann je nach Vorgeschichte halten oder nicht. Darum wird er über die
    ganze Historie fortgeschrieben; live und im Backtest dieselbe Funktion.
    """
    upper = high.rolling(entry).max().shift(1)
    lower = low.rolling(exit).min().shift(1)
    state = False
    out = []
    for c, up, lo in zip(close.to_numpy(), upper.to_numpy(), lower.to_numpy()):
        if not state and c > up:
            state = True
        elif state and c < lo:
            state = False
        out.append(state)
    return pd.Series(out, index=close.index)


@dataclass
class TrendVote:
    symbol: str
    votes: int
    hold: bool
    candle_date: datetime   # Datum der zuletzt abgeschlossenen Tageskerze (UTC)
    close: float
    rules: Dict[str, bool]
    # Nur Donchian: Kanal für die nächste Kerze (Ausbruchs-Hoch, Ausstiegs-Tief)
    upper: Optional[float] = None
    lower: Optional[float] = None

    def describe(self) -> str:
        """Einzeiler für Log und Dashboard."""
        verdict = 'halten' if self.hold else 'draußen'
        if self.upper is not None:
            return (f"Schluss {self.close:.2f} · Hoch {self.upper:.2f} · Tief {self.lower:.2f} → {verdict}")
        ja = ', '.join(r for r, ok in self.rules.items() if ok) or 'keine'
        return f"{self.votes}/5 Regeln ({ja}) → {verdict}"


class DailyTrendStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('DailyTrend')
        self.min_votes = int(self.config.get('min_votes', MIN_VOTES))
        self.rule = self.config.get('rule', 'vote')       # vote | donchian
        self.entry_days = int(self.config.get('entry_days', 20))
        self.exit_days = int(self.config.get('exit_days', 10))
        if self.rule not in ('vote', 'donchian'):
            raise ValueError(f"daily_trend.rule unbekannt: {self.rule}")
        self.last_votes: Dict[str, TrendVote] = {}
        self._exchange = None

    @staticmethod
    def kraken_symbol(symbol: str) -> str:
        """BTC_EUR -> BTC/EUR"""
        return symbol.replace('_', '/')

    @property
    def label(self) -> str:
        if self.rule == 'donchian':
            return f"Donchian {self.entry_days}/{self.exit_days} Tage"
        return f"mindestens {self.min_votes} von 5 Regeln"

    async def fetch_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        """Abgeschlossene Tageskerzen (Hoch, Tief, Schluss) von Kraken, ältestes zuerst."""
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
        return df.set_index('date')[['high', 'low', 'close']].astype(float)

    def evaluate(self, symbol: str, daily: pd.DataFrame) -> Optional[TrendVote]:
        need = self.entry_days * 3 if self.rule == 'donchian' else max(int(r[3:]) for r in RULES)
        if daily is None or len(daily) <= need:
            self.logger.warning(f"{symbol}: zu wenig Tageskerzen ({0 if daily is None else len(daily)})")
            return None
        close = daily['close']
        if self.rule == 'donchian':
            hold = bool(donchian_hold(daily['high'], daily['low'], close,
                                      self.entry_days, self.exit_days).iloc[-1])
            vote = TrendVote(
                symbol=symbol,
                votes=int(hold),
                hold=hold,
                candle_date=close.index[-1].to_pydatetime(),
                close=float(close.iloc[-1]),
                rules={'ausbruch': hold},
                upper=float(daily['high'].iloc[-self.entry_days:].max()),
                lower=float(daily['low'].iloc[-self.exit_days:].min()),
            )
            self.last_votes[symbol] = vote
            return vote
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
