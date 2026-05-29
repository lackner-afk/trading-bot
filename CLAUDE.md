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
| `core/portfolio.py` | Portfolio with SQLite persistence (works for both paper & live) |
| `core/risk_manager.py` | Kelly criterion, stop-loss, drawdown limits |
| `core/order_engine.py` | Paper trading execution |
| `core/live_order_engine.py` | Real One Trading execution via CCXT (with Shadow Mode) |
| `core/reconciliation.py` | Startup sync between local state and exchange (critical for live) |
| `data/onetrading_ccxt_feed.py` | Recommended live data feed (CCXT onetrading) |
| `data/kraken_feed.py` | Good public EUR feed for paper mode |
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

**Live Mode** is heavily guarded (see `LIVE_TRADING.md` and `tools/paper_to_live_checklist.py`). The bot supports both Paper and Live mode with proper branching in `main.py`.

## Testing Strategies

Run backtester on historical data:
```bash
python backtest.py
```

The backtester loads 90 days of CoinGecko data and simulates all strategies with metrics: Return, Sharpe, Win Rate, Max Drawdown.
