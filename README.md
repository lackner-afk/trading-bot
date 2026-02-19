# Paper-Trading-Bot

Ein hochspekulativer Paper-Trading-Bot für Crypto-Perpetuals und Polymarket-Prediction-Märkte.

**⚠️ ACHTUNG: Dies ist eine SIMULATION mit Fake-Money. Kein echtes Trading!**

## Features

- **Crypto-Scalping**: High-Leverage Momentum/Breakout-Trading auf 1m-Kerzen
- **Polymarket-Arbitrage**: Scannt YES/NO-Shares für Arbitrage-Opportunities
- **Hybrid-Sentiment**: Nutzt Polymarket-Events als Leading Indicator für Crypto
- **ML-Predictor**: Gradient Boosting für Preis-Prediktion

## Installation

```bash
# Dependencies installieren
pip install -r requirements.txt

# Konfiguration vorbereiten
cp config/secrets.env.example config/secrets.env
# API-Keys eintragen (optional für Testnet)
```

## Schnellstart

```bash
# Bot im Paper-Trading-Modus starten
python main.py

# Backtester ausführen
python backtest.py

# Parameter-Optimierung
python backtest.py --grid
```

## Projektstruktur

```
trading-bot/
├── config/
│   ├── settings.yaml          # Alle konfigurierbaren Parameter
│   └── secrets.env.example    # Template für API-Keys
├── core/
│   ├── portfolio.py           # Fake-Portfolio-Management
│   ├── risk_manager.py        # Stop-Loss, Drawdown-Limits
│   └── order_engine.py        # Simulated Order Execution
├── strategies/
│   ├── polymarket_arbitrage.py
│   ├── crypto_scalper.py
│   ├── hybrid_sentiment.py
│   └── ml_predictor.py
├── data/
│   ├── crypto_feed.py         # CCXT-basierter Datenfeed
│   ├── polymarket_feed.py     # Polymarket CLOB API
│   └── backtester.py          # Historische Simulation
├── notifications/
│   └── reporter.py            # Console + Webhook Reports
├── main.py                    # Haupt-Orchestrator
└── backtest.py                # Standalone Backtest-Runner
```

## Strategien

### Crypto-Scalper
- RSI < 30 + Bollinger Lower Touch + Volume Spike → LONG
- RSI > 70 + Bollinger Upper Touch + Volume Spike → SHORT
- Breakout über 15m-High/Low mit Volume → Momentum-Trade
- Leverage: 20-50x (konfigurierbar)

### Polymarket-Arbitrage
- Scannt Binary-Markets nach YES_ask + NO_ask < 1.00
- Kauft beide Seiten → Garantierter Profit nach Resolution
- Min-Profit-Threshold: 0.5% nach Fees

### Hybrid-Sentiment
- Überwacht crypto-relevante Polymarket-Events
- Bei >5% Preisänderung in 1h → Trade-Signal
- Korrelation zu BTC/ETH/SOL basierend auf Event-Typ

### ML-Predictor
- Gradient Boosting auf RSI, BB, Volume, Sentiment
- Target: Preis-Richtung in 5-15 Minuten
- Retrain alle 6 Stunden
- Min-Confidence: 65%

## Risk-Management

Harte Limits (nicht überschreibbar):
- Max 2% Risiko pro Trade
- Max 10% täglicher Drawdown → Pause bis nächster Tag
- Max 20% pro Position
- Max 50x Leverage
- Max 5 gleichzeitige Positionen
- 5 Min Cooldown nach 3 Verlusten in Folge

## Konfiguration

Alle Parameter in `config/settings.yaml`:

```yaml
general:
  mode: paper  # NIEMALS 'live' ohne explizite Bestätigung!
  start_capital: 10000

strategies:
  scalper:
    leverage: 20
    pairs: [BTC/USDT, ETH/USDT, SOL/USDT]

risk:
  max_daily_drawdown: 0.10
  max_leverage: 50
```

## Disclaimer

Dieser Bot ist ausschließlich für Bildungszwecke und Paper-Trading gedacht. Crypto-Trading mit hohem Leverage ist extrem riskant. Nutze niemals echtes Geld ohne umfassende Kenntnisse der Risiken.
