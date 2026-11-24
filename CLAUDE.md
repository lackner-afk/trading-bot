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

# Tests (no network needed)
pip install -r requirements-dev.txt && python -m pytest tests

# Daily-trend backtest and its robustness check (random baseline, walk-forward)
POSITION_FRACTION=0.2 MAX_POSITIONS=2 python tools/backtest_daily.py
POSITION_FRACTION=0.2 MAX_POSITIONS=2 python tools/robustness_daily.py
```

> **Status 24.11.2026:** the only strategy that trades is `daily_trend` — three paper bots, all
> with 1000 €: Panda-9 (Donchian 20/10 on daily candles), Trend-1 (majority vote, overlay
> `config/daily_trend.yaml`) and Swing-1 (EMA 20/50 on 4h candles, long AND simulated short,
> overlay `config/ema_4h.yaml` — the only one with `spot_only: false`). Momentum, scalper, ML and confluence are all
> `enabled: false`. Measured evidence and its limits: `docs/BACKTEST_BEFUNDE.md`,
> section "Unabhängige Prüfung (17.11.2026)".

## Architecture Overview

This is an **async Python paper-trading bot** for crypto spot markets, trading EUR pairs. It uses real-time data from Bitpanda Fusion (paper mode default, `general.data_feed`), multiple concurrent trading strategies, simulated order execution, and a SQLite-backed portfolio.

### Core Flow

1. **main.py** orchestrates everything via `asyncio` event loops
2. **FusionFeed** (paper default) provides Bitpanda Fusion prices, orderbook bid/ask and candles via REST; **KrakenFeed** is the keyless alternative (`data_feed: kraken`)
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
├── dashboard/
│   ├── server.py               # aiohttp web server (Panda Pro UI), /api/state + /api/pause
│   ├── store.py                # SQLite tables: equity_snapshots, dashboard_events, bot_flags
│   └── static/index.html       # Single-file dashboard UI (vanilla JS + SVG, no CDN)
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
| `dashboard.run` | 5min | Web dashboard on 127.0.0.1:8080 + equity snapshots (if `dashboard.enabled`) |

## Data Feeds

### Primary (paper): Bitpanda Fusion (fusion_feed.py)

- Selected by `general.data_feed: bitpanda` (default); requires `BITPANDA_API_KEY` even for market data — startup fails loudly without it
- Base URL `https://api.fusion.bitpanda.com`, pair format `BTC-EUR` mapped to `BTC_EUR`
- Prices: `/v1/tickers` every 5s (mid price only) + `/v1/orderbook` for real bid/ask
- Candles: `/v1/candles` every 60s (Unix seconds), `/v1/pairs` checked at startup (min order amount)

### Alternative: Kraken (kraken_feed.py)

- Selected by `general.data_feed: kraken`

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

### Daily trend (strategies/daily_trend.py) — ENABLED (the only active strategy)

- Signals from **closed Kraken EUR daily candles**, execution at the feed price on the next 15-minute check
- `rule: donchian` (Panda-9): buy when the close is above the prior 20-day high, sell below the prior 10-day low
- `rule: vote` (Trend-1 overlay `config/daily_trend.yaml`): hold while ≥3 of 5 rules (SMA 50/100/200, momentum 30/90) agree
- `rule: ema` (Swing-1 overlay `config/ema_4h.yaml`): `timeframe: 4h`, always in the market — long while EMA 20 > EMA 50,
  otherwise short; a direction change closes and reopens in the same step. Shorts only pass `_execute_signal`
  because that overlay sets `spot_only: false` (Fusion cannot short — paper comparison only, no borrow costs)
- `TrendVote.target` (+1/−1/0) drives `_daily_trend_step`; the open candle is dropped in `closed_candles()`
- No TP/SL; 1x. Same functions are used by `tools/backtest_daily.py` and `tools/backtest_longshort.py`
- Below ~150 € the 30 € minimum order exceeds the 20 % position limit (only via `allow_min_order_exception`, logged as `[RISIKO]`)

### Momentum (strategies/momentum.py) — DISABLED

- **Signal**: EMA9 crosses EMA21 on 5m candles + RSI filter
  - LONG: EMA9 > EMA21 + RSI in [35, 55]
  - SHORT: EMA9 < EMA21 + RSI in [45, 65]
- **1h trend filter**: Blocks longs in downtrends, blocks shorts in uptrends
- **ATR-based sizing**: SL = ATR×2.0, TP = ATR×4.0, trailing = ATR×1.2
- **Cooldown**: 300s per symbol after signal
- **Leverage**: suggests 10x (confidence is effectively constant); forced to 1 by `spot_only`
- The old "+4.2% / Sharpe 1.14" claim is **not reproducible**: `backtest.py` on 19.08.–17.11.2026 gives −74 %
  (10x, shorts, 1h candles — a different rule than the live 5m path)

### Scalper (strategies/crypto_scalper.py) — DISABLED

- **Strategy 1 (Mean Reversion)**: RSI < 30/> 70 + price at BB band + volume spike (>2x)
- **Strategy 2 (Breakout)**: Price breaks 15m high/low + volume spike
- **Leverage**: Base 20x, max 50x, scaled by confidence
- Disabled via `scalper.enabled: false` in settings.yaml (too noisy on 1m timeframe)

### ML Predictor (strategies/ml_predictor.py) — DISABLED

- Never placed a trade (`_ml_loop` only logs) and cannot train in paper mode: `train()` needs 500 candles, the feeds keep 300

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

### Known gaps (verified 17.11.2026)

- `check_trade()` now gets the real order size; a `REDUCE_SIZE` below the exchange minimum means no trade
- The minimum order may exceed the 20 % position limit only via `exchange_rules.allow_min_order_exception: true`
  (exactly the minimum order, never more than 50 % of equity; decided by the user 17.11.2026)
- Strategies without a stop-loss (daily trend) have no real 2 % risk bound — `check_trade` assumes a 1 % stop
- `calculate_position_size()` (Kelly) is never called
- Live mode with `shadow_mode: false` refuses to start: `_close_position` sends no sell order

### Position Sizing

- `calculate_position_size()`: Half-Kelly criterion based on historical win rate and P/L ratios
- `size_from_risk()`: SL-distance based sizing to target exactly 2% risk

## Order Execution (core/order_engine.py)

### Realistic Simulation

| Parameter | Detail |
|-----------|--------|
| Slippage | 0.01–0.05% random, scales with order size |
| Maker fee | 0.25% (`fees.crypto_maker`, Bitpanda Fusion tier 1) |
| Taker fee | 0.25% (`fees.crypto_taker`) — also used for exit fees in `_close_position` |
| Latency | 50–200ms simulated |
| Partial fills | Orders >$50k have 20–30% partial fill probability |

### Exchange Rules (`exchange_rules` in settings.yaml)

Applied in `_execute_signal` in every mode, so paper results match Bitpanda Fusion:
- `spot_only: true` → SHORT signals are skipped, leverage forced to 1
- Minimum order per pair comes from FusionFeed `/v1/pairs` (`minOrderAmount`); `min_order_eur` is only the fallback.
  Orders below the minimum are raised to it if that stays within the 25% margin cap, otherwise skipped

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
  start_capital: 1000  # EUR (all three bots since 24.11.2026)
  data_feed: bitpanda

strategies:
  momentum:   {enabled: false}   # pairs: here define the traded pairs for ALL strategies
  scalper:    {enabled: false}
  ml:         {enabled: false}
  confluence: {enabled: false}
  daily_trend:
    enabled: true
    rule: donchian     # vote in config/daily_trend.yaml (Trend-1)
    entry_days: 20
    exit_days: 10

risk:
  max_risk_per_trade: 0.02
  max_daily_drawdown: 0.10
  max_position_size: 0.20        # NOT read by the code (class constant applies)
  max_leverage: 1
  max_concurrent_positions: 2
  cooldown_after_losses: 300     # NOT read by the code (it reads cooldown_seconds)

fees:    {crypto_maker: 0.0025, crypto_taker: 0.0025}
slippage: {min: 0.0001, max: 0.0005}   # entry and exit in paper mode
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
- Simulates positions with leverage, fees default to 0.25% (Bitpanda Fusion tier 1) unless configured
- Hard stops: 0.8% SL, 1.5% TP
- Max 3 concurrent positions, 10% max per position, 5-period cooldown
- Generates: Return%, Sharpe, Max DD, Win rate, Profit factor, Alpha vs. Buy&Hold

**Live Mode** is heavily guarded (see `LIVE_TRADING.md` and `tools/paper_to_live_checklist.py`). The bot supports both Paper and Live mode with proper branching in `main.py`.

### Strategies Tested in Backtest

1. Scalper (RSI+BB+Volume)
2. Momentum (EMA Cross)
3. Mean Reversion (RSI+BB)
4. Breakout (20-period high/low)

## Bitpanda Trading Limits (docs/BITPANDA_API_LIMITS.md)

Exchange-side safety net, independent of `core/risk_manager.py`:

- `POST https://api.public.bitpanda.com/v1/trading-limits` with header `x-api-key`
- Body: `buy_limit`, `sell_limit` (numbers), `currency_id` (UUID) — all required
- `200` returns the limits plus `buy_budget_remaining` / `sell_budget_remaining`; `409` = limits already set for this key (POST cannot overwrite)
- Limits are per API key, per day
- Helper: `python tools/bitpanda_trading_limit.py --buy X --sell Y --currency-id <UUID> [--confirm]` (dry run without `--confirm`)
- Keep `sell_limit` above `buy_limit` — otherwise the bot may be unable to close positions

## Dashboard (dashboard/)

- Runs inside the bot process; reads live state from `TradingBot` (portfolio, risk manager, feed)
- `TradingBot.trading_paused` blocks NEW entries only (checked in `_execute_signal`); exits keep running.
  Persisted in `bot_flags` so a restart doesn't silently resume trading
- Open/close/pause events are recorded via `TradingBot._record_event()` — failures there must never block trading
- Binding to a non-loopback host requires `DASHBOARD_TOKEN`; POSTs check Origin + JSON content type
- On first start the store backfills equity curve + activity from the existing `trades` table
- With `DASHBOARD_PASSWORD` set, the whole site (not only `/api/`) requires Basic-Auth (`DASHBOARD_USER`, default `nici`); `/health` stays open
- Extra panels: signal proximity per coin (`SignalAggregator.last_eval` — also rejected scores), heartbeat (`TradingBot._last_cycle_at`), data recorder progress (`market_data.db`, read-only)

## VPS deployment (deploy/vps/)

- `./deploy/vps/deploy.sh` rsyncs to `deploy@91.98.234.212:/home/deploy/trading-bot` and runs `docker compose up -d --build --remove-orphans`
- Containers: `trading-bot` (Panda-9, dashboard on 127.0.0.1:3400, `DASHBOARD_HOST=0.0.0.0` inside), `trading-trend` (Trend-1, :3401),
  `trading-swing` (Swing-1, :3402) and `trading-recorder`; each bot has its own `/state` volume (`BOT_STATE_DIR`);
  the Panda-9 dashboard switches to the others via `DASHBOARD_PEERS`
- nginx + Let's Encrypt in front: https://bot.silvertoast.at
- Secrets only on the server: `config/secrets.env` (BITPANDA_API_KEY — a Fusion key, not a Bitpanda app key) and `config/dashboard.env`
- If Fusion delivers no prices at startup (rejected key still "starts"), the bot falls back to Kraken and logs an ERROR

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
