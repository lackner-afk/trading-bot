"""
Multi-Factor Strategy Demo (New 2026 Architecture)

This script demonstrates how to use the new ConfluenceStrategy with:
- Regime Detection
- Dynamic Asset Selection
- Multiple Technical Factors
- Basic Macro/News Filter
- Sentiment (Fear & Greed)

Run it with real candle data to see the new system in action.
"""

import asyncio
import pandas as pd
from datetime import datetime

# Import from our new system
from strategies import (
    ConfluenceStrategy,
    RegimeDetector,
    AssetSelector,
)

from strategies.factors.technical import (
    MultiTimeframeTrendFactor,
    MomentumFactor,
    VolatilityFilter,
    BreakoutFactor,
    VolumeConfirmationFactor,
)
from strategies.factors.macro_news import MacroNewsFilter
from strategies.factors.sentiment import SentimentFactor


async def fetch_sample_candles(symbol: str) -> pd.DataFrame:
    """
    Placeholder for fetching candles.
    In real usage, replace this with your CCXT feed (Kraken / OneTrading).
    """
    # For demo purposes, we generate fake data
    import numpy as np

    np.random.seed(42)
    n = 120
    base = 60000 if "BTC" in symbol else (2500 if "ETH" in symbol else 140)

    close = base + np.cumsum(np.random.randn(n) * (base * 0.008))
    high = close + np.abs(np.random.randn(n)) * (base * 0.003)
    low = close - np.abs(np.random.randn(n)) * (base * 0.003)
    volume = np.random.randint(1000, 50000, n)

    df = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="5min"),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

    return df


async def main():
    print("=== Multi-Factor Confluence Strategy Demo ===\n")

    # 1. Create the strategy with the new architecture
    strategy = ConfluenceStrategy()

    # 2. Add a strong set of factors
    strategy.add_common_factors(include_sentiment=True, include_macro=True)

    # You can also add custom factors manually:
    # strategy.add_factor(YourCustomFactor())

    # 3. Define which symbols we want to analyze
    symbols = ["BTC_EUR", "ETH_EUR", "SOL_EUR"]

    # 4. Fetch data (in real bot this comes from your CCXT feed)
    candles_dict = {}
    for sym in symbols:
        candles_dict[sym] = await fetch_sample_candles(sym)

    # 5. Get recommended universe + regime
    universe = strategy.get_recommended_assets(candles_dict)
    print(f"Current Regime: {universe['regime'].name}")
    print(f"Recommended Assets: {universe['symbols']}")
    print(f"Preferred Styles: {universe['preferred_styles']}\n")

    # 6. Analyze each recommended asset
    for symbol in universe["symbols"]:
        candles = candles_dict.get(symbol)
        if candles is None or len(candles) == 0:
            continue

        current_price = candles["close"].iloc[-1]
        signal = strategy.analyze(symbol, candles, current_price)

        if signal:
            print(f"→ SIGNAL for {symbol}")
            print(f"   Direction: {signal.direction.upper()}")
            print(f"   Confluence Score: {signal.confluence_score:.1f}/10")
            print(f"   Confidence: {signal.confidence:.0%}")
            print(f"   Suggested Leverage: {signal.suggested_leverage:.1f}x")
            print(f"   Reason: {signal.reason}")
            print(f"   TP: {signal.take_profit:.2f} | SL: {signal.stop_loss:.2f}")
            print()
        else:
            print(f"→ No high-confluence signal for {symbol}\n")

    print("=== Demo finished ===")


if __name__ == "__main__":
    asyncio.run(main())
