# Claude Code Prompt: Hochspekulativer Paper-Trading-Bot (Crypto + Polymarket)

---

## Prompt (Copy & Paste in Claude Code):

```
Erstelle einen vollständigen Paper-Trading-Bot in Python, der Crypto-Perpetuals und Polymarket-Prediction-Märkte kombiniert, um aggressive tägliche Gewinne zu simulieren. Der Bot läuft im Fake-Money-Modus mit 10.000 USD Startkapital.

## Projektstruktur

Erstelle folgende Struktur:

```
trading-bot/
├── config/
│   ├── settings.yaml          # Alle konfigurierbaren Parameter
│   └── secrets.env.example    # Template für API-Keys
├── core/
│   ├── portfolio.py           # Fake-Portfolio-Management (Balances, PNL-Tracking)
│   ├── risk_manager.py        # Stop-Loss, Drawdown-Limits, Position-Sizing
│   └── order_engine.py        # Simulated Order Execution mit Slippage-Modell
├── strategies/
│   ├── polymarket_arbitrage.py # Scannt YES/NO-Shares wo Summe < $1
│   ├── crypto_scalper.py      # High-Leverage Momentum/Breakout-Scalping
│   ├── hybrid_sentiment.py    # Polymarket-Sentiment → Crypto-Trade-Signals
│   └── ml_predictor.py        # ML-basierte Preis-/Outcome-Prediktion
├── data/
│   ├── crypto_feed.py         # Live-Daten via ccxt (Binance, Bybit, Hyperliquid)
│   ├── polymarket_feed.py     # Polymarket CLOB API für Orderbooks & Events
│   └── backtester.py          # Historische Simulation mit CoinGecko + Polymarket
├── notifications/
│   └── reporter.py            # PNL-Reports (Console + optional Webhook/WhatsApp)
├── main.py                    # Haupt-Orchestrator mit async Event-Loop
├── backtest.py                # Standalone Backtest-Runner
├── requirements.txt
└── README.md
```

## Technische Anforderungen

### 1. Dependencies installieren
```bash
pip install ccxt pandas numpy ta scikit-learn torch requests websockets pyyaml python-dotenv aiohttp rich
```
- Verwende `ta` (nicht ta-lib) um C-Library-Probleme zu vermeiden
- `rich` für schöne Console-Outputs
- `aiohttp` + `asyncio` für parallele API-Calls

### 2. Daten-Layer (`data/`)

**crypto_feed.py:**
- Nutze `ccxt` im Sandbox/Testnet-Modus für Bybit und Binance Futures
- Fetche OHLCV-Daten (1m, 5m, 15m, 1h Kerzen) für: BTC/USDT, ETH/USDT, SOL/USDT, DOGE/USDT, PEPE/USDT
- Streaming via WebSocket wo möglich, Fallback auf REST-Polling alle 5 Sekunden
- Berechne live: RSI(14), Bollinger Bands(20,2), VWAP, Volume-Delta, EMA(9,21)

**polymarket_feed.py:**
- Nutze die Polymarket CLOB API (https://docs.polymarket.com)
- Endpunkte: GET /markets, GET /book (Orderbook), GET /prices
- Scanne alle aktiven Märkte mit Volume > $10k
- Berechne für jedes Binary-Market: `spread = 1 - (best_yes_ask + best_no_ask)`
- Wenn spread > 0: Arbitrage-Opportunity gefunden
- Tracke auch zeitbasierte Märkte (z.B. "BTC über $X in 4h") für Sentiment

### 3. Strategien (`strategies/`)

**polymarket_arbitrage.py:**
```python
# Pseudocode-Logik:
# 1. Scanne alle Binary-Markets
# 2. Für jedes Market: hole best YES ask + best NO ask
# 3. Wenn YES_ask + NO_ask < 1.00: 
#    → Kaufe beide Seiten → Garantierter Profit = 1.00 - Kosten - Fees
# 4. Auch: Vergleiche gleiche Events über verschiedene Zeitfenster
# 5. Min-Profit-Threshold: 0.5% nach Fees
# Execution: Sofort, da Arbitrage-Fenster kurz sind (Sekunden)
```

**crypto_scalper.py:**
```python
# Logik:
# 1. Scanne 1m-Kerzen für Momentum-Signale:
#    - RSI < 30 + Bollinger Lower Touch + Volume Spike → LONG
#    - RSI > 70 + Bollinger Upper Touch + Volume Spike → SHORT
# 2. Breakout-Detection: Preis durchbricht 15m-High/Low mit >2x Avg-Volume
# 3. Leverage: Simuliere 20x-50x (konfigurierbar)
# 4. Entry/Exit:
#    - Take-Profit: 0.3%-1% (je nach Volatilität)
#    - Stop-Loss: 0.1%-0.5% 
#    - Trailing-Stop nach 0.2% Profit
# 5. Max 3 gleichzeitige Positionen
```

**hybrid_sentiment.py:**
```python
# Logik:
# 1. Lese Polymarket-Preise für crypto-relevante Events
#    (z.B. "Fed Rate Cut", "BTC ATH in Q1", "ETF Approval")
# 2. Wenn Event-Wahrscheinlichkeit sich >5% in 1h ändert:
#    → Generiere Crypto-Signal basierend auf erwarteter Korrelation
# 3. Beispiel: "BTC > 100k" YES-Preis steigt von 0.40→0.55
#    → Long BTC/USDT mit 10x Leverage
# 4. Gewichte Signal mit ML-Confidence-Score
```

**ml_predictor.py:**
- Trainiere ein Gradient Boosting Modell (scikit-learn) auf:
  - Features: RSI, BB-Position, Volume-Ratio, Funding-Rate, Polymarket-Sentiment-Scores
  - Target: Preis-Richtung in nächsten 5-15 Minuten
- Retrain alle 6 Stunden mit neuen Daten
- Mindest-Confidence von 65% für Trade-Signal
- Optional: Einfaches LSTM (PyTorch) für Sequence-Prediction

### 4. Risk Management (`core/risk_manager.py`)

```python
# Harte Regeln (NICHT überschreibbar):
MAX_RISK_PER_TRADE = 0.02      # 2% des Portfolios
MAX_DAILY_DRAWDOWN = 0.10       # 10% → Pause bis nächster Tag
MAX_POSITION_SIZE = 0.20        # 20% des Portfolios pro Position
MAX_LEVERAGE = 50               # Absolutes Leverage-Limit
MAX_CONCURRENT_POSITIONS = 5
COOLDOWN_AFTER_3_LOSSES = 300   # 5 Min Pause nach 3 Verlusten in Folge

# Position Sizing: Kelly-Criterion (halb-Kelly für Sicherheit)
# Dynamisch: Reduziere Größe bei steigendem Drawdown
# Warnung bei: Daily-PNL < -5%, Sharpe < 0.5 über 24h
```

### 5. Portfolio & Order Engine (`core/`)

**portfolio.py:**
- Starte mit 10.000 USD (fake)
- Tracke: Balance, Equity, Unrealized PNL, Realized PNL, Win-Rate, Sharpe-Ratio
- Speichere alle Trades in SQLite-DB für Analyse
- Berechne stündlich und täglich: PNL, Drawdown, Win/Loss-Ratio, Avg-Win vs Avg-Loss

**order_engine.py:**
- Simuliere realistische Execution:
  - Slippage: 0.01%-0.05% (zufällig)
  - Fees: 0.04% Maker / 0.06% Taker (Crypto), 2% auf Polymarket
  - Latenz: 50-200ms simuliert
  - Partial Fills bei großen Orders

### 6. Haupt-Loop (`main.py`)

```python
# Async Event-Loop:
# 1. Starte alle Data-Feeds (Crypto + Polymarket) parallel
# 2. Alle 1 Sekunde: 
#    - Polymarket-Arbitrage-Scanner
# 3. Alle 5 Sekunden:
#    - Crypto-Scalper-Signale prüfen
#    - Hybrid-Sentiment-Check
# 4. Alle 5 Minuten:
#    - ML-Predictor Re-Score
#    - Risk-Check (Drawdown, Exposure)
# 5. Alle 1 Stunde:
#    - PNL-Report ausgeben
#    - Performance-Metriken updaten
# 6. Alle 6 Stunden:
#    - ML-Modell Retrain
#    - Strategy-Parameter Auto-Tuning
# 
# Nutze asyncio.gather() für Parallelismus
# Graceful Shutdown mit Signal-Handling
```

### 7. Reporting (`notifications/reporter.py`)

- Console: Rich-Table mit Live-Portfolio, offenen Positionen, letzte Trades
- Stündlich: Kompakter PNL-Summary (Balance, PNL%, Top-Trade, Worst-Trade)
- Täglich: Ausführlicher Report (alle Metriken, Strategy-Breakdown, Empfehlungen)
- Format: Klar strukturiert, farbcodiert (grün/rot)
- Optional: Webhook-URL für externe Notifications (konfigurierbar in settings.yaml)

### 8. Backtester (`backtest.py`)

- Lade historische Daten: CoinGecko API (30-90 Tage, stündlich)
- Simuliere alle Strategien auf historischen Daten
- Output: Total Return, Max Drawdown, Sharpe Ratio, Win-Rate pro Strategie
- Vergleiche: Buy & Hold vs. Bot-Performance
- Optimiere Parameter via Grid-Search

### 9. Konfiguration (`config/settings.yaml`)

```yaml
general:
  mode: paper  # paper | live (NIEMALS live ohne explizite Bestätigung)
  start_capital: 10000
  base_currency: USDT

exchanges:
  bybit:
    testnet: true
    api_key: ${BYBIT_TESTNET_KEY}
    secret: ${BYBIT_TESTNET_SECRET}
  binance:
    testnet: true

polymarket:
  enabled: true
  min_volume: 10000
  arbitrage_min_profit: 0.005  # 0.5%

strategies:
  scalper:
    enabled: true
    leverage: 20
    pairs: [BTC/USDT, ETH/USDT, SOL/USDT]
    timeframe: 1m
  arbitrage:
    enabled: true
    scan_interval: 1  # Sekunden
  hybrid:
    enabled: true
  ml:
    enabled: true
    retrain_hours: 6

risk:
  max_risk_per_trade: 0.02
  max_daily_drawdown: 0.10
  max_leverage: 50
  cooldown_seconds: 300
```

## Ausführungsreihenfolge

1. Erstelle alle Dateien mit vollständigem, lauffähigem Code
2. Installiere alle Dependencies
3. Führe den Backtester aus, um die Strategien zu validieren
4. Starte den Bot im Paper-Trading-Modus
5. Zeige den ersten PNL-Report nach 5 Minuten Laufzeit
6. Gib eine Zusammenfassung: Was funktioniert, was muss optimiert werden

## Wichtige Hinweise

- ALLES ist Paper-Trading / Simulation — kein echtes Geld
- Code muss TATSÄCHLICH laufen, nicht nur Pseudocode
- Fehlerbehandlung: Try/Except überall, API-Fehler graceful handeln
- Logging: Jeder Trade, jede Entscheidung wird geloggt
- Wenn eine API nicht erreichbar ist: Fallback auf simulierte Daten
- Polymarket-API kann Geo-Restrictions haben → baue Mock-Fallback ein
- Kommentare auf Deutsch im Code
```

---

## Nutzung

Kopiere den gesamten Text zwischen den äußeren ``` Blöcken und füge ihn als einzelnen Prompt in Claude Code ein.
