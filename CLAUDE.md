# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the paper-trading bot
python main.py

# Run backtester (defaults: Kraken EUR data, 5m candles, 90 days)
python backtest.py

# Backtest against a specific source / timeframe
python backtest.py --data-exchange onetrading --timeframe 5m --days 90

# Run parameter grid search for strategy optimization
python backtest.py --grid

# Check whether the bot is provably profitable (go-live gate)
python tools/profitability_gate.py

# Full paper -> live checklist
python tools/paper_to_live_checklist.py

# Run tests
pip install -r requirements-dev.txt
pytest -q

# Monitor logs in real-time
tail -f bot.log
```

Note: on Debian-based systems the `ta` package can fail to build against the
patched system setuptools. A venv resolves it: `python -m venv .venv &&
.venv/bin/pip install -r requirements.txt`.

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
| `core/portfolio.py` | Portfolio with SQLite persistence (works for both paper & live) |
| `core/risk_manager.py` | Kelly criterion, stop-loss, drawdown limits |
| `core/order_engine.py` | Paper trading execution (simulated) |
| `core/live_order_engine.py` | **Real** One Trading execution via CCXT (supports Shadow Mode) |
| `core/reconciliation.py` | Startup reconciliation between local state and exchange (critical for live) |
| `data/onetrading_ccxt_feed.py` | Recommended live data feed (CCXT `onetrading`, supports auth) |
| `data/kraken_feed.py` | Good public EUR feed for paper mode |
| `strategies/momentum.py` | EMA 9/21 crossover with RSI + 1h trend filter (main strategy) |
| `strategies/crypto_scalper.py` | RSI+BB+Volume mean-reversion & breakout |
| `strategies/ml_predictor.py` | Gradient Boosting price direction predictor |
| `notifications/reporter.py` | Rich console + Telegram (Money Boy / "i bims" style) |

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

## Spot Mode (core/market_constraints.py)

The target venue (Bitpanda Fusion) is **spot-only**: no leverage, no shorts.
These constraints apply in paper mode too — otherwise the test phase would
measure a strategy that cannot be executed live, and the numbers would be
useless for a go-live decision.

Enforced at four independent points (defense in depth):

1. `SignalAggregator` — leverage fixed at 1.0, SHORT becomes `is_exit_signal`
2. `main.py::_handle_exit_signal` — a SHORT signal closes an open long
   ("sell what you hold"); no-op otherwise
3. `RiskManager` — BLOCK on leverage > 1, as check 0 before everything else
4. `Portfolio.open_position` — refuses to book a short or leveraged position

Configured via the `trading:` block in `settings.yaml`.

## Go-Live Gate (tools/profitability_gate.py)

Eleven criteria evaluated against the real trade history in `trades.db`.
Enforced hard in `main.py` as the third live hurdle (after
`LIVE_TRADING_ENABLED` and `live_explicit_confirmation`), checked in
`paper_to_live_checklist.py`, and shown advisory in the daily report.

Thresholds live in the `go_live_gate:` block in `settings.yaml`. All
metrics come from `core/performance.py` — one implementation for every
consumer.

## Strategies

### Confluence (strategies/confluence_strategy.py) — ENABLED, primary

The only active strategy. `main.py` starts its loop exclusively; when
`confluence.enabled: true`, the momentum/scalper/ML loops do not run at all,
regardless of their own flags.

- **Factors**: `multi_timeframe_trend`, `momentum`, `volatility_filter`,
  `breakout`, `volume_confirmation`, `sentiment` (Fear & Greed, contrarian).
  `macro_news_filter` is off by default — the `EconomicCalendar` is never
  populated, so it would contribute a constant score of 1.0.
- **Scoring**: weighted mean of factor scores, renormalized over the
  categories actually present. **The score is in [0, 1]** — thresholds must
  be on that scale. `min_confluence_score > 1.0` raises at startup.
- **Direction**: weighted by `score * confidence`, requires a margin over the
  opposing direction, plus `min_directional_score` so that directionless
  filters cannot carry a signal on their own.
- **Regimes**: `RegimeDetector` yields `trending` / `ranging` / `low_vol_chop` /
  `high_vol_event`, steering factor weights and asset selection.
- **Exits**: `strategies/confluence_exit.py` (`ConfluenceExitManager`) with its
  own high/low tracking fed from the main loop, ATR from the feed.
- **Backtest**: via `data/confluence_backtest_adapter.py`.

### Momentum (strategies/momentum.py) — DISABLED

- **Signal**: EMA9 crosses EMA21 on 5m candles + RSI filter
  - LONG: EMA9 > EMA21 + RSI in [35, 55]
  - SHORT: EMA9 < EMA21 + RSI in [45, 65]
- **1h trend filter**: Blocks longs in downtrends, blocks shorts in uptrends
- **ATR-based sizing**: SL = ATR×2.0, TP = ATR×4.0, trailing = ATR×1.2
- **Old backtest results**: +4.2% return, 60% win rate, Sharpe 1.14 — note
  these came from `generate_signal()`, which checks EMA *state* rather than
  the crossover the live path used. They do not describe the live strategy.

### Scalper (strategies/crypto_scalper.py) — DISABLED

- **Strategy 1 (Mean Reversion)**: RSI < 30/> 70 + price at BB band + volume spike (>2x)
- **Strategy 2 (Breakout)**: Price breaks 15m high/low + volume spike
- **Leverage**: Base 20x, max 50x, scaled by confidence
- Disabled via `scalper.enabled: false` in settings.yaml (too noisy on 1m timeframe)

### ML Predictor (strategies/ml_predictor.py) — DISABLED

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
| `portfolio_state` | Balance, realized PNL, win/loss counts, daily start balance |
| `equity_snapshots` | Persisted equity curve (throttled, default every 60s) |

Trades **and** the equity curve are reloaded on startup — long-term
performance would otherwise be unmeasurable across restarts.

### Key Metrics

- `get_sharpe_ratio()`: Annualized Sharpe from **daily** returns (√365)
- `get_max_drawdown()`: Peak-to-trough over the full persisted curve
- `get_daily_drawdown()`: Intraday drawdown from today's peak
- `get_profit_factor()`: gross profit / gross loss — **not** the payoff ratio
- `get_expectancy()`, `get_net_pnl()`, `get_total_fees()`

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
- Loads OHLCV via CCXT; source selectable (`--data-exchange kraken|onetrading|binance`)
- Timeframe configurable (`--timeframe`, default **5m** — Confluence runs on 5m live)
- Uses capital, fees and risk limits from `config/settings.yaml`
- Honours signal-provided SL/TP and the `close` signal
- **Never falls back to synthetic data silently** — pass `--allow-synthetic` to permit it.
  `BacktestResult.data_source` records where the data came from.
- Generates: Return%, Sharpe, Max DD, Win rate, Profit factor, Alpha vs. Buy&Hold

**Live Mode** is heavily guarded (see `LIVE_TRADING.md` and `tools/paper_to_live_checklist.py`). The bot supports both Paper and Live mode with proper branching in `main.py`.

### Strategies Tested in Backtest

1. **Confluence (Multi-Factor)** — the active strategy, via `data/confluence_backtest_adapter.py`
2. Scalper (RSI+BB+Volume)
3. Momentum (EMA Cross)
4. Mean Reversion (RSI+BB)
5. Breakout (20-period high/low)

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
7. **`min_confluence_score` lives on a 0–1 scale.** The score is a weighted mean
   of factor scores from [0, 1] with weights summing to 1.0, so it can never
   exceed 1.0. A value like `3.5` is unreachable and silently blocks every
   trade — that bug cost this project its entire runtime. `SignalAggregator`
   now raises at startup on such a value; do not "fix" that by removing the check.
8. **Exits must go through the OrderEngine.** `_close_position` books into the
   portfolio only after `result.success`. Never bypass it — a locally closed but
   really open position is the worst possible state.
9. **Spot constraints apply in paper mode too.** Do not relax them to get more
   trades in testing; that would make the test data unusable for a go-live decision.
10. **Do not weaken the go-live gate to make it pass.** It is the only check
    that looks at actual performance.
