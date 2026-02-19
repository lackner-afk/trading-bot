# Trading-Bot — Projektspezifikation

Async Python Paper-Trading-Bot für Crypto-Spot-Märkte auf One Trading (EUR-Pairs).

## Ziel

Paper-Trading-Bot der auf One Trading live Marktdaten abruft (WebSocket + REST),
technische Strategien anwendet und ein simuliertes Portfolio trackt — ohne echtes Geld.

## Datenfeed: One Trading

- **WebSocket**: `wss://streams.fast.onetrading.com`
  - Channel: `PRICE_TICKS` für Echtzeit Bid/Ask/Last
- **REST**: `https://api.onetrading.com/fast/v1/candlesticks/{symbol}`
  - Parameter: `unit` (MINUTES/HOURS), `period`, `from`, `to`
- **Pairs**: BTC_EUR, ETH_EUR, SOL_EUR, XRP_EUR
- Kein API Key für öffentliche Daten nötig

## Technische Indikatoren (pro Pair, pro Timeframe)

Berechnet auf jeder Kerze via `ta`-Library:
- RSI (14)
- Bollinger Bands (20, 2σ)
- EMA 9 und EMA 21
- VWAP (vereinfacht)
- Volume Delta

## Strategien

### 1. EMA-Momentum (Hauptstrategie, aktiv)

```
LONG:  EMA9 kreuzt über EMA21  AND RSI < 40
SHORT: EMA9 kreuzt unter EMA21 AND RSI > 60

Take-Profit:   +1,5%
Stop-Loss:     -0,8%
Trailing-Stop:  0,5%
Leverage:      10x (bis 20x je nach Confidence)
Cooldown:      5 Minuten pro Symbol nach Signal
```

Backtest-Ergebnis (90 Tage): +4,2% Return, 60% Win Rate, 6,3% Max DD, Sharpe 1,14

### 2. Crypto-Scalper (deaktiviert in config)

```
Mean-Reversion: RSI < 30 + BB Lower Touch + Volume Spike → LONG
Mean-Reversion: RSI > 70 + BB Upper Touch + Volume Spike → SHORT
Breakout:       Preis über 15m-High + Volume Spike → LONG
Breakout:       Preis unter 15m-Low  + Volume Spike → SHORT

Leverage: 20–50x
```

### 3. ML-Predictor (informell, führt selbst keine Trades aus)

- Gradient Boosting Classifier pro Symbol
- Features: RSI, BB-Position, Volume-Ratio, EMA-Cross, Momentum, Volatilität
- Target: Preis-Richtung in 15 Minuten (up/down)
- Retrain alle 6 Stunden auf den letzten ~500 1m-Kerzen
- Gibt Prognose + Probability aus, wird im Log angezeigt

## Async Loop-Struktur (main.py)

```
Haupt-Loop     (1s):  Preise aktualisieren, Pending Orders, Exit-Checks
Momentum-Loop  (30s): EMA-Crossover analysieren
Scalper-Loop   (15s): RSI/BB/Breakout analysieren
ML-Loop        (5min): Modell retrainieren + Prognose ausgeben
Risk-Loop      (5min): Drawdown und Exposure prüfen
Report-Loop    (1h):  Portfolio-Summary ausgeben
```

## Risk-Management

```
Max Risiko pro Trade:      2% des Portfolios
Max täglicher Drawdown:   10% → alle Positionen schließen
Max Positionsgröße:       20% des Portfolios
Max Leverage:             50x (absolut)
Max gleichzeitige Pos.:    5
Cooldown nach 3 Verlusten: 5 Minuten
```

Position Sizing: Kelly-Criterion (Halb-Kelly), skaliert mit Drawdown

## Portfolio & Execution

- SQLite-Datenbank für Persistenz (trades.db)
- Simulierte Slippage: 0,01%–0,05%
- Fees: Maker 0,04% / Taker 0,06%
- Paper-Money: $10.000 Startkapital

## Projektstruktur

```
trading-bot/
├── config/
│   ├── settings.yaml
│   └── secrets.env.example
├── core/
│   ├── portfolio.py        # Fake-Portfolio, SQLite, PNL-Tracking
│   ├── risk_manager.py     # Alle Risiko-Checks
│   └── order_engine.py     # Simulated Execution
├── data/
│   ├── onetrading_feed.py  # WebSocket + REST Datenfeed
│   ├── crypto_feed.py      # CandleData/MarketData Dataclasses
│   └── backtester.py       # Offline Backtest
├── strategies/
│   ├── momentum.py         # EMA 9/21 Crossover
│   ├── crypto_scalper.py   # RSI+BB+Volume
│   └── ml_predictor.py     # Gradient Boosting
├── notifications/
│   └── reporter.py         # Rich Console + Webhooks
├── deploy/
│   ├── setup-server.sh
│   └── deploy.sh
├── main.py
└── backtest.py
```

## Code-Konventionen

- Kommentare auf Deutsch
- Async/await für alle I/O-Operationen
- Dataclasses für strukturierte Daten
- Type Hints überall
- Rich Library für Console-Output
