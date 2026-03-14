# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the paper-trading bot
python main.py

# Run backtester (90 days historical data)
python backtest.py

# Run parameter grid search for strategy optimization
python backtest.py --grid

# Monitor logs in real-time
tail -f bot.log
```

## Architecture Overview

This is an **async Python paper-trading bot** for crypto spot markets, trading EUR pairs. It uses real-time data from Kraken (via CCXT), multiple concurrent trading strategies, simulated order execution, and a SQLite-backed portfolio.

### Core Flow

1. **main.py** orchestrates everything via `asyncio` event loops
2. **KrakenFeed** (primary) provides real-time prices via polling + historical candles via CCXT
3. **OneTradingFeed** (alternative) provides prices via WebSocket + REST candlesticks
4. **Strategies** analyze data and generate directional signals
5. **OrderEngine** simulates execution with realistic slippage and fees
6. **Portfolio** tracks positions, PNL, and persists to SQLite
7. **RiskManager** enforces hard limits on leverage, drawdown, and position sizing

### Directory Structure

```
trading-bot/
├── config/
│   ├── settings.yaml           # All configuration parameters
│   └── secrets.env.example     # Template for API keys / Telegram credentials
├── core/
│   ├── __init__.py
│   ├── portfolio.py            # SQLite-backed paper portfolio (Position, Trade, PortfolioState)
│   ├── risk_manager.py         # Kelly criterion, position sizing, drawdown guards
│   └── order_engine.py         # Simulated market/limit orders with slippage & fees
├── data/
│   ├── __init__.py
│   ├── crypto_feed.py          # Shared dataclasses (CandleData, MarketData) + CCXT generic feed
│   ├── onetrading_feed.py      # One Trading WebSocket (PRICE_TICKS) + REST candlesticks
│   ├── kraken_feed.py          # Kraken feed via CCXT (EUR pairs, no API key required)
│   └── backtester.py           # Historical simulation engine with grid-search support
├── strategies/
│   ├── __init__.py
│   ├── momentum.py             # EMA 9/21 crossover + RSI filter + 1h trend filter
│   ├── crypto_scalper.py       # RSI+BB+Volume mean-reversion & breakout (currently disabled)
│   └── ml_predictor.py         # GradientBoosting price direction predictor (optional LSTM)
├── notifications/
│   ├── __init__.py
│   └── reporter.py             # Rich console UI + Telegram/Discord notifications
├── main.py                     # Bot orchestrator with 7 async loops
├── backtest.py                 # Standalone backtester entry point
├── requirements.txt
└── trades.db                   # SQLite database (auto-created at runtime)
```

### Key Components

| Module | Purpose |
|--------|---------|
| `core/portfolio.py` | Paper portfolio with SQLite persistence; tracks open/closed positions and PNL |
| `core/risk_manager.py` | Half-Kelly criterion, stop-loss calculation, hard drawdown limits |
| `core/order_engine.py` | Simulated market/limit orders with realistic slippage (0.01-0.05%) and fees |
| `data/kraken_feed.py` | **Primary feed** — Kraken prices via CCXT, no API key needed |
| `data/onetrading_feed.py` | **Alternative feed** — One Trading WebSocket + REST (EUR pairs) |
| `data/crypto_feed.py` | Shared dataclasses + generic CCXT feed (Bybit/Binance testnet) |
| `data/backtester.py` | Historical backtest engine using Binance 90-day OHLCV data |
| `strategies/momentum.py` | **Primary strategy** — EMA 9/21 crossover, enabled in config |
| `strategies/crypto_scalper.py` | High-frequency RSI+BB scalper, **disabled** (too noisy on 1m) |
| `strategies/ml_predictor.py` | GradientBoosting predictor, enabled, retrains every 6h |
| `notifications/reporter.py` | Rich tables + Telegram bot (Austrian dialect style) |

### Async Loop Structure (main.py)

The `TradingBot` class runs 7 concurrent async loops:

| Loop | Interval | Responsibility |
|------|----------|----------------|
| `_main_loop` | 1s | Price updates, pending order checks, exit conditions (TP/SL/trailing) |
| `_momentum_loop` | 30s | EMA crossover signals on 5m candles with 1h trend filter |
| `_scalper_loop` | 15s | RSI/BB/breakout signals on 1m candles (disabled in config) |
| `_ml_loop` | 5min | Model retraining and ML-based signal generation |
| `_risk_check_loop` | 5min | Drawdown and exposure checks; may pause trading |
| `_reporting_loop` | 1h | Portfolio summaries in console |
| `_telegram_hourly_loop` | 1h | Telegram notifications (rate-limited) |

## Data Feeds

### Primary: Kraken (kraken_feed.py)

- Symbol mapping: `BTC_EUR` → `BTC/EUR` internally
- Prices: `fetch_tickers` polled every 5s
- Candles: `fetch_ohlcv` polled every 60s
- No API key required for public market data

### Alternative: One Trading (onetrading_feed.py)

- **WebSocket**: `wss://streams.fast.onetrading.com` — PRICE_TICKS channel
- **REST**: `https://api.onetrading.com/fast/v1/candlesticks/{symbol}` — historical OHLCV
- **Pairs**: BTC_EUR, ETH_EUR, SOL_EUR, XRP_EUR
- Auto-reconnects on disconnect; falls back to simulated data if unavailable

### Fallback: CryptoFeed (crypto_feed.py)

- Generic CCXT-based feed supporting Bybit/Binance testnet
- Provides simulated candle data when live APIs are unavailable

### Technical Indicators (calculated in feed)

All feeds compute these on candle data:
- **RSI(14)**: Momentum oscillator
- **Bollinger Bands(20, 2)**: Volatility envelope
- **EMA(9) and EMA(21)**: Trend direction
- **VWAP**: Volume-weighted average price
- **Volume Delta**: Buy vs. sell volume estimate

## Strategies

### Momentum (strategies/momentum.py) — ENABLED

- **Signal**: EMA9 crosses EMA21 on 5m candles + RSI filter
  - LONG: EMA9 > EMA21 + RSI in [35, 55]
  - SHORT: EMA9 < EMA21 + RSI in [45, 65]
- **1h trend filter**: Blocks longs in downtrends, blocks shorts in uptrends
- **ATR-based sizing**: SL = ATR×2.0, TP = ATR×4.0, trailing = ATR×1.2
- **Cooldown**: 300s per symbol after signal
- **Leverage**: 10–20x depending on confidence (RSI + EMA spread)
- **Backtest results**: +4.2% return, 60% win rate, 6.3% max DD, Sharpe 1.14

### Scalper (strategies/crypto_scalper.py) — DISABLED

- **Strategy 1 (Mean Reversion)**: RSI < 30/> 70 + price at BB band + volume spike (>2x)
- **Strategy 2 (Breakout)**: Price breaks 15m high/low + volume spike
- **Leverage**: Base 20x, max 50x, scaled by confidence
- Disabled via `scalper.enabled: false` in settings.yaml (too noisy on 1m timeframe)

### ML Predictor (strategies/ml_predictor.py) — ENABLED

- **Model**: `GradientBoostingClassifier` (100 estimators, depth=5, lr=0.1)
- **Features**: RSI, RSI-change, BB position, volume ratio, 5m/15m price change, EMA cross, momentum, volatility, sentiment
- **Retraining**: Every 6 hours automatically
- **Minimum confidence**: 0.65 (configurable in settings.yaml)
- **Optional LSTM**: Uses `torch` if available (`LSTMPredictor` / `AdvancedMLPredictor`)

## Risk Management (core/risk_manager.py)

### Hard Limits (enforced regardless of config)

| Limit | Value |
|-------|-------|
| Max risk per trade | 2% of equity |
| Max daily drawdown | 10% → trading paused until next day |
| Max position size | 20% of equity |
| Max leverage | 50x |
| Max concurrent positions | 5 |
| Cooldown after 3 consecutive losses | 5 minutes |

### Risk Actions

`RiskManager.check_trade()` returns one of: `ALLOW`, `REDUCE_SIZE`, `BLOCK`, `CLOSE_ALL`, `COOLDOWN`

### Position Sizing

- `calculate_position_size()`: Half-Kelly criterion based on historical win rate and P/L ratios
- `size_from_risk()`: SL-distance based sizing to target exactly 2% risk

## Order Execution (core/order_engine.py)

### Realistic Simulation

| Parameter | Detail |
|-----------|--------|
| Slippage | 0.01–0.05% random, scales with order size |
| Maker fee | 0.04% |
| Taker fee | 0.06% |
| Latency | 50–200ms simulated |
| Partial fills | Orders >$50k have 20–30% partial fill probability |

### Order Types

- `MARKET`: Filled immediately with slippage
- `LIMIT`: Queued until price reaches limit
- `STOP_MARKET`: Triggered at stop price
- `STOP_LIMIT`: Triggered at stop, executed as limit

## Portfolio (core/portfolio.py)

### SQLite Tables

| Table | Contents |
|-------|----------|
| `positions` | Open positions (symbol, side, size, entry_price, leverage, SL/TP) |
| `trades` | Closed trades with PNL, fees, entry/exit timestamps |
| `portfolio_state` | Current balance, equity, unrealized/realized PNL snapshots |

### Key Metrics

- `get_sharpe_ratio()`: Annualized Sharpe from hourly returns
- `get_max_drawdown()`: Peak-to-trough drawdown percentage
- `get_daily_drawdown()`: Intraday drawdown from today's peak

## Configuration (config/settings.yaml)

```yaml
general:
  mode: paper          # NEVER change to 'live' without explicit user confirmation
  start_capital: 100   # EUR
  base_currency: EUR

strategies:
  momentum:
    enabled: true
    leverage: 10        # Base leverage (scales 10–20x by confidence)
    pairs: [BTC_EUR, ETH_EUR, SOL_EUR]
    take_profit: 0.015  # 1.5%
    stop_loss: 0.008    # 0.8%
    trailing_stop: 0.005
    cooldown_seconds: 900
    rsi_long_threshold: 55
    rsi_short_threshold: 45
    sl_atr_multiplier: 2.0
    tp_atr_multiplier: 4.0

  scalper:
    enabled: false      # Disabled — too noisy on 1m candles

  ml:
    enabled: true
    retrain_hours: 6
    min_confidence: 0.65

risk:
  max_risk_per_trade: 0.02
  max_daily_drawdown: 0.10
  max_position_size: 0.20
  max_leverage: 50
  max_concurrent_positions: 2
  cooldown_after_losses: 300

fees:
  crypto_maker: 0.0004
  crypto_taker: 0.0006

notifications:
  console: true
  telegram:
    enabled: true
    token: ${TELEGRAM_BOT_TOKEN}    # Set in config/secrets.env
    chat_id: ${TELEGRAM_CHAT_ID}
```

### Secrets

Copy `config/secrets.env.example` to `config/secrets.env` and set:
- `TELEGRAM_BOT_TOKEN`: Bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Target chat/user ID

## Code Conventions

- **Language**: All code comments and log messages in **German**
- **Async**: `async/await` for all I/O — network, database, and file operations
- **Data structures**: `@dataclass` for structured data (CandleData, MarketData, Position, Trade, etc.)
- **Type hints**: Throughout all functions and class attributes
- **Console output**: Use `rich` library for tables, panels, and colored output
- **Logging**: Standard `logging` module to `bot.log` file + console; use `INFO` level by default
- **PID lock**: `main.py` uses a PID file to prevent multiple simultaneous instances

## Backtesting (data/backtester.py + backtest.py)

- Loads 90 days of hourly OHLCV from Binance via CCXT
- Simulates positions with leverage, realistic fees (0.04%/0.06%)
- Hard stops: 0.8% SL, 1.5% TP
- Max 3 concurrent positions, 10% max per position, 5-period cooldown
- Generates: Return%, Sharpe, Max DD, Win rate, Profit factor, Alpha vs. Buy&Hold

### Strategies Tested in Backtest

1. Scalper (RSI+BB+Volume)
2. Momentum (EMA Cross)
3. Mean Reversion (RSI+BB)
4. Breakout (20-period high/low)

## Notifications (notifications/reporter.py)

- **Rich console**: Portfolio tables, position tables, trade history
- **Telegram**: "Moneyboy-style" messages (Austrian dialect, emoji-heavy)
  - Trade alerts sent on every open/close
  - Hourly summary rate-limited to max 1/hour
- **Webhooks**: Discord/Slack embed format supported

## Important Warnings for AI Assistants

1. **Never switch `mode: paper` to `mode: live`** without explicit user confirmation
2. **Never remove hard risk limits** in `core/risk_manager.py` — the 2% risk cap and 10% daily drawdown limits are safety-critical
3. **Never commit `config/secrets.env`** — it contains API credentials
4. The **scalper strategy is intentionally disabled** — do not re-enable without testing
5. All Telegram messages intentionally use casual Austrian dialect — do not "fix" the style
6. The backtester uses **Binance data**, not CoinGecko (the existing CLAUDE.md was outdated on this point)
