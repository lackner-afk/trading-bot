#!/usr/bin/env python3
"""
Standalone Backtest-Runner
Testet alle Strategien auf historischen Daten
"""

import asyncio
import logging
from datetime import datetime

from data.backtester import Backtester
from strategies.crypto_scalper import CryptoScalper
from strategies.momentum import MomentumStrategy


# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('BacktestRunner')


def scalper_strategy(df, idx):
    """Scalper-Strategie für Backtest"""
    scalper = CryptoScalper()
    return scalper.generate_signal(df, idx)


def momentum_strategy(df, idx):
    """EMA-Cross Momentum mit Crossover-Detection"""
    momentum = MomentumStrategy()
    return momentum.generate_signal(df, idx)


def mean_reversion_strategy(df, idx):
    """Mean-Reversion-Strategie"""
    if idx < 20:
        return None

    rsi = df.iloc[idx].get('rsi')
    bb_upper = df.iloc[idx].get('bb_upper')
    bb_lower = df.iloc[idx].get('bb_lower')
    close = df.iloc[idx]['close']

    if rsi is None or bb_upper is None:
        return None

    # Long bei überverkauft + Bollinger Lower
    if rsi < 30 and close <= bb_lower:
        return 'long'

    # Short bei überkauft + Bollinger Upper
    if rsi > 70 and close >= bb_upper:
        return 'short'

    return None


def breakout_strategy(df, idx):
    """Breakout-Strategie"""
    if idx < 20:
        return None

    close = df.iloc[idx]['close']
    volume = df.iloc[idx]['volume']
    avg_volume = df.iloc[max(0, idx-20):idx]['volume'].mean()

    # 20-Perioden High/Low
    high_20 = df.iloc[max(0, idx-20):idx]['high'].max()
    low_20 = df.iloc[max(0, idx-20):idx]['low'].min()

    # Volume Spike erforderlich
    if volume < avg_volume * 1.5:
        return None

    # Breakout nach oben
    if close > high_20:
        return 'long'

    # Breakout nach unten
    if close < low_20:
        return 'short'

    return None


async def run_backtests():
    """Führt alle Backtests durch"""
    print("\n" + "=" * 70)
    print("PAPER-TRADING-BOT BACKTEST")
    print("=" * 70)

    # Backtester initialisieren
    backtester = Backtester(initial_capital=10000)

    # Historische Daten laden
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    print(f"\nLade historische Daten für: {', '.join(symbols)}...")

    await backtester.load_data(symbols, days=90)

    if not backtester.price_data:
        print("FEHLER: Keine Daten verfügbar!")
        return

    print(f"Daten geladen: {sum(len(df) for df in backtester.price_data.values())} Datenpunkte")

    # Strategien definieren
    strategies = [
        ('Scalper (RSI+BB+Volume)', scalper_strategy),
        ('Momentum (EMA Cross)', momentum_strategy),
        ('Mean Reversion (RSI+BB)', mean_reversion_strategy),
        ('Breakout (20-Period)', breakout_strategy),
    ]

    results = []

    # Jede Strategie testen
    for name, strategy_func in strategies:
        print(f"\nTeste: {name}...")

        result = backtester.run_backtest(
            strategy_func=strategy_func,
            symbols=symbols,
            leverage=10,
            strategy_name=name
        )

        if result:
            results.append(result)
            backtester.print_results(result)

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    if results:
        # Sortiere nach Return
        results.sort(key=lambda r: r.total_return_pct, reverse=True)

        print(f"\n{'Strategie':<30} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Max DD':>10}")
        print("-" * 70)

        for r in results:
            print(f"{r.strategy_name:<30} "
                  f"{r.total_return_pct:>9.1%} "
                  f"{r.sharpe_ratio:>8.2f} "
                  f"{r.win_rate:>9.1%} "
                  f"{r.max_drawdown:>9.1%}")

        # Beste Strategie
        best = results[0]
        print(f"\n🏆 BESTE STRATEGIE: {best.strategy_name}")
        print(f"   Return: {best.total_return_pct:.1%} | Alpha: {best.alpha:.1%}")

    print("\n" + "=" * 70)


def run_grid_search():
    """Führt Parameter-Optimierung durch"""
    print("\n" + "=" * 70)
    print("PARAMETER GRID-SEARCH")
    print("=" * 70)

    class ParameterizedScalper:
        def __init__(self, rsi_oversold=30, rsi_overbought=70, volume_threshold=2.0):
            self.rsi_oversold = rsi_oversold
            self.rsi_overbought = rsi_overbought
            self.volume_threshold = volume_threshold

        def generate_signal(self, df, idx):
            if idx < 20:
                return None

            row = df.iloc[idx]
            rsi = row.get('rsi')
            bb_upper = row.get('bb_upper')
            bb_lower = row.get('bb_lower')
            volume = row['volume']

            if rsi is None or bb_upper is None:
                return None

            avg_volume = df.iloc[max(0, idx-20):idx]['volume'].mean()
            volume_spike = volume > avg_volume * self.volume_threshold

            if rsi < self.rsi_oversold and row['close'] <= bb_lower and volume_spike:
                return 'long'

            if rsi > self.rsi_overbought and row['close'] >= bb_upper and volume_spike:
                return 'short'

            return None

    # Grid Search durchführen
    backtester = Backtester(initial_capital=10000)

    # Synchrones Laden für Grid Search
    import aiohttp

    async def load_and_search():
        await backtester.load_data(['BTC/USDT', 'ETH/USDT'], days=60)

        param_grid = {
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'volume_threshold': [1.5, 2.0, 2.5]
        }

        results = backtester.grid_search(
            ParameterizedScalper,
            param_grid,
            symbols=['BTC/USDT', 'ETH/USDT']
        )

        print("\nTop 5 Parameter-Kombinationen:")
        print("-" * 70)

        for i, r in enumerate(results[:5], 1):
            print(f"{i}. {r.strategy_name}")
            print(f"   Return: {r.total_return_pct:.1%} | Sharpe: {r.sharpe_ratio:.2f} | "
                  f"Win Rate: {r.win_rate:.1%}")

    asyncio.run(load_and_search())


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--grid':
        run_grid_search()
    else:
        asyncio.run(run_backtests())
