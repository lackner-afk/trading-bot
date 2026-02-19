# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the paper-trading bot
python main.py

# Run backtester
python backtest.py

# Run parameter grid search
python backtest.py --grid
```

## Architecture Overview

This is an async Python paper-trading bot for crypto spot markets on One Trading (EUR pairs).

### Core Flow
1. **main.py** orchestrates everything via `asyncio` event loops
2. **OneTradingFeed** provides real-time prices via WebSocket + historical candles via REST
3. **Strategies** analyze data and generate signals
4. **OrderEngine** simulates execution with slippage/fees
5. **Portfolio** tracks positions, PNL, and persists to SQLite
6. **RiskManager** enforces hard limits on leverage, drawdown, position sizing

### Key Components

| Module | Purpose |
|--------|---------|
| `core/portfolio.py` | Fake portfolio with SQLite persistence |
| `core/risk_manager.py` | Kelly criterion, stop-loss, drawdown limits |
| `core/order_engine.py` | Simulated market/limit orders with slippage |
| `data/onetrading_feed.py` | One Trading WebSocket (PRICE_TICKS) + REST candlesticks |
| `data/crypto_feed.py` | Shared dataclasses (CandleData, MarketData) + CCXT fallback |
| `strategies/momentum.py` | EMA 9/21 crossover with RSI filter |
| `strategies/crypto_scalper.py` | RSI+BB+Volume mean-reversion and breakout |
| `strategies/ml_predictor.py` | Gradient Boosting price direction prediction |

### Async Loop Structure (main.py)

- **Main Loop** (1s): Price updates, pending orders, exit conditions
- **Momentum Loop** (30s): EMA crossover signals
- **Scalper Loop** (15s): RSI/BB/Breakout signals
- **ML Loop** (5min): Model retraining and predictions
- **Risk Loop** (5min): Drawdown and exposure checks
- **Report Loop** (1h): PNL summaries

### Data Feed: One Trading

- **WebSocket**: `wss://streams.fast.onetrading.com` — PRICE_TICKS channel für Live-Preise
- **REST**: `https://api.onetrading.com/fast/v1/candlesticks/{symbol}` — historische OHLCV-Daten
- **Pairs**: BTC_EUR, ETH_EUR, SOL_EUR, XRP_EUR
- Kein API Key benötigt für öffentliche Marktdaten

## Code Conventions

- All comments in German (as per original spec)
- Async/await for all I/O operations
- Dataclasses for structured data
- Type hints throughout
- Rich library for console output

## Configuration

All settings in `config/settings.yaml`. Environment variables for secrets in `config/secrets.env`.

## Testing Strategies

Run backtester on historical data:
```bash
python backtest.py
```

The backtester loads 90 days of CoinGecko data and simulates all strategies with metrics: Return, Sharpe, Win Rate, Max Drawdown.
