"""
Adapter, der die ConfluenceStrategy im Backtester nutzbar macht.

Das Problem: `Backtester.run_backtest` ruft `strategy_func(df, idx)` auf,
während `ConfluenceStrategy.analyze(symbol, candles, current_price)` ein
Symbol und ein Kerzenfenster erwartet. Weil die alte Signatur kein Symbol
kennt, war die einzige produktiv aktive Strategie des Repos nicht
backtestbar — und damit nie validiert.

Der Adapter schliesst die Lücke und garantiert dabei das Wichtigste:
**kein Look-ahead**. Die Strategie sieht ausschliesslich Zeilen bis
einschliesslich `idx`, nie eine Zeile aus der Zukunft.
"""

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Fensterlänge, die an die Strategie geht. Entspricht dem Live-Verhalten
# (main.py holt get_candles(..., n=80)) und begrenzt zugleich den Aufwand:
# fünf Faktoren machen je ein candles.copy(), bei 26.000 5m-Kerzen pro
# Symbol wären das sonst ~180.000 DataFrame-Kopien.
DEFAULT_WINDOW = 200
DEFAULT_WARMUP = 80


def make_confluence_func(strategy, symbol: str,
                         warmup: int = DEFAULT_WARMUP,
                         window: int = DEFAULT_WINDOW,
                         min_confidence: float = 0.0) -> Callable:
    """
    Baut aus einer ConfluenceStrategy eine `f(df, idx)`-Funktion.

    Args:
        strategy: ConfluenceStrategy-Instanz
        symbol: Symbol, für das diese Funktion gilt
        warmup: Kerzen, bevor überhaupt ein Signal erzeugt wird
        window: Länge des rückwärtsgerichteten Fensters
        min_confidence: Optionale zusätzliche Schwelle

    Returns:
        Callable, das ein dict {'action', 'stop_loss', 'take_profit'} liefert
        oder None.
    """

    def f(df: pd.DataFrame, idx: int):
        if idx < warmup:
            return None

        # Rückwärtsgerichtetes Fenster — der entscheidende Punkt:
        # iloc[start : idx + 1] enthält niemals eine Zeile > idx.
        start = max(0, idx - window)
        hist = df.iloc[start: idx + 1]

        try:
            price = float(df['close'].iloc[idx])
        except (KeyError, IndexError):
            return None

        if price <= 0:
            return None

        try:
            signal = strategy.analyze(symbol, hist, price)
        except Exception as e:
            logger.debug(f"[Backtest] analyze() fuer {symbol}@{idx} fehlgeschlagen: {e}")
            return None

        if signal is None:
            return None
        if signal.confidence < min_confidence:
            return None

        # Spot-Semantik: ein SHORT-Signal ist kein Entry, sondern der Auftrag
        # eine offene Long-Position zu schliessen. Der Backtester versteht
        # 'close' — damit läuft der Backtest in derselben Logik wie live.
        action = 'close' if signal.is_exit_signal else signal.direction

        return {
            'action': action,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'confidence': signal.confidence,
        }

    return f


def build_strategy_funcs(strategy_factory: Callable, symbols: List[str],
                         warmup: int = DEFAULT_WARMUP,
                         window: int = DEFAULT_WINDOW) -> Dict[str, Callable]:
    """
    Baut das `strategy_funcs`-Dict für run_backtest.

    `strategy_factory` wird PRO SYMBOL aufgerufen. Das ist Absicht: die
    ConfluenceStrategy hält Zustand (RegimeDetector-Cache, Faktor-Zustände),
    der sich zwischen Symbolen nicht vermischen darf.
    """
    funcs: Dict[str, Callable] = {}
    for symbol in symbols:
        funcs[symbol] = make_confluence_func(
            strategy_factory(), symbol, warmup=warmup, window=window
        )
    return funcs


def prepare_candles_for_confluence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass alle Indikatoren vorliegen, die die Faktoren erwarten.

    Der Backtester berechnet rsi/bb/ema bereits in _calculate_indicators();
    diese Funktion ergänzt nur, was fehlt, damit der Adapter auch mit
    fremden DataFrames funktioniert.
    """
    out = df
    needed = ('rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'ema_9', 'ema_21')
    if all(col in out.columns for col in needed):
        return out

    out = df.copy()
    if 'ema_9' not in out.columns:
        out['ema_9'] = out['close'].ewm(span=9, adjust=False).mean()
    if 'ema_21' not in out.columns:
        out['ema_21'] = out['close'].ewm(span=21, adjust=False).mean()
    if 'bb_middle' not in out.columns:
        out['bb_middle'] = out['close'].rolling(20).mean()
        std = out['close'].rolling(20).std()
        out['bb_upper'] = out['bb_middle'] + 2 * std
        out['bb_lower'] = out['bb_middle'] - 2 * std
    if 'rsi' not in out.columns:
        delta = out['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        out['rsi'] = (100 - 100 / (1 + gain / loss.replace(0, pd.NA))).fillna(50.0)

    return out
