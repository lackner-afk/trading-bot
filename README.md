# Paper-Trading-Bot — One Trading

Async Python Paper-Trading-Bot für Crypto-Spot-Märkte auf [One Trading](https://www.onetrading.com/) (EUR-Pairs).

**⚠️ ACHTUNG: Dies ist eine SIMULATION mit Fake-Money. Kein echtes Trading!**

## Features

- **Live-Datenfeed**: One Trading WebSocket (PRICE_TICKS) + REST API für Candlesticks
- **EMA-Momentum**: EMA 9/21 Crossover-Strategie mit RSI-Filter
- **Crypto-Scalper**: RSI+Bollinger Mean-Reversion und Breakout-Trading
- **ML-Predictor**: Gradient Boosting für Preis-Richtungsprognosen
- **Risk-Management**: Kelly-Criterion, Drawdown-Limits, Trailing-Stop

## Installation

```bash
# Dependencies installieren
pip install -r requirements.txt

# Konfiguration vorbereiten (API Key optional)
cp config/secrets.env.example config/secrets.env
```

## Schnellstart

```bash
# Bot im Paper-Trading-Modus starten
python main.py

# Backtester ausführen
python backtest.py

# Parameter-Optimierung (Grid Search)
python backtest.py --grid
```

## Projektstruktur

```
trading-bot/
├── config/
│   ├── settings.yaml          # Alle konfigurierbaren Parameter
│   └── secrets.env.example    # Template für API-Keys
├── core/
│   ├── portfolio.py           # Fake-Portfolio mit SQLite-Persistenz
│   ├── risk_manager.py        # Stop-Loss, Drawdown-Limits, Kelly-Criterion
│   └── order_engine.py        # Simulierte Order-Execution mit Slippage
├── data/
│   ├── onetrading_feed.py     # One Trading WebSocket + REST Datenfeed
│   ├── crypto_feed.py         # Shared Dataclasses + CCXT-Fallback
│   └── backtester.py          # Historische Simulation
├── strategies/
│   ├── momentum.py            # EMA 9/21 Crossover + RSI-Filter
│   ├── crypto_scalper.py      # RSI+BB Mean-Reversion und Breakout
│   └── ml_predictor.py        # Gradient Boosting Preis-Prediktor
├── notifications/
│   └── reporter.py            # Rich Console + optionale Webhook-Notifications
├── deploy/
│   ├── setup-server.sh        # Einmalig auf dem VPS ausführen
│   └── deploy.sh              # Code deployen: Laptop → VPS
├── main.py                    # Haupt-Orchestrator (asyncio)
└── backtest.py                # Standalone Backtest-Runner
```

## Datenfeed: One Trading

| Kanal | Protokoll | Zweck |
|-------|-----------|-------|
| PRICE_TICKS | WebSocket | Echtzeit Bid/Ask/Last |
| candlesticks | REST | Historische OHLCV-Kerzen |

Unterstützte Pairs: **BTC_EUR · ETH_EUR · SOL_EUR · XRP_EUR**

API Key ist für öffentliche Marktdaten **nicht erforderlich**.

## Strategien

### EMA-Momentum (aktiv, Hauptstrategie)
- EMA9 kreuzt über EMA21 + RSI < 40 → **LONG**
- EMA9 kreuzt unter EMA21 + RSI > 60 → **SHORT**
- Take-Profit: 1,5% · Stop-Loss: 0,8% · Trailing-Stop: 0,5%
- Leverage: 10× (bis 20× bei hoher Confidence)
- Backtest: +4,2% Return, 60% Win Rate, 6,3% Max DD, Sharpe 1,14

### Crypto-Scalper (deaktiviert)
- RSI < 30 + Bollinger Lower Touch + Volume Spike → LONG
- RSI > 70 + Bollinger Upper Touch + Volume Spike → SHORT
- Breakout über 15m-High/Low mit Volume → Momentum
- Leverage: 20–50× (konfigurierbar)

### ML-Predictor (informell)
- Gradient Boosting auf RSI, BB, Volume, EMA-Cross, Momentum
- Prognose-Horizont: 15 Minuten
- Retrain alle 6 Stunden
- Gibt Signale aus, führt selbst keine Trades aus

## Risk-Management

Harte Limits (nicht überschreibbar):
- Max **2%** Risiko pro Trade
- Max **10%** täglicher Drawdown → Pause bis nächster Tag
- Max **20%** pro Position
- Max **50×** Leverage
- Max **5** gleichzeitige Positionen
- **5 Min** Cooldown nach 3 Verlusten in Folge

## Konfiguration

Alle Parameter in `config/settings.yaml`:

```yaml
general:
  mode: paper       # NIEMALS 'live' ohne explizite Bestätigung!
  start_capital: 10000
  base_currency: USDT

strategies:
  momentum:
    enabled: true
    leverage: 10
    pairs: [BTC_EUR, ETH_EUR, SOL_EUR, XRP_EUR]
    take_profit: 0.015
    stop_loss: 0.008

risk:
  max_daily_drawdown: 0.10
  max_leverage: 50
```

## VPS-Deployment

```bash
# Einmalig: Server einrichten
ssh root@<IP> 'bash -s' < deploy/setup-server.sh

# Code deployen + Bot neustarten
./deploy/deploy.sh <SERVER-IP>

# Logs live verfolgen
ssh deploy@<IP> 'journalctl -u trading-bot -f -q'
```

## Disclaimer

Dieser Bot ist ausschließlich für Bildungszwecke und Paper-Trading gedacht. Crypto-Trading ist extrem riskant. Nutze niemals echtes Geld ohne umfassende Kenntnisse der Risiken.
