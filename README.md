# Paper-Trading-Bot — One Trading

Async Python Paper-Trading-Bot für Crypto-Spot-Märkte auf [One Trading](https://www.onetrading.com/) (EUR-Pairs).

**⚠️ ACHTUNG: Dies ist eine SIMULATION mit Fake-Money. Kein echtes Trading!**

**→ Bevor du jemals echtes Geld einsetzt: Lies `LIVE_TRADING.md` (im Projekt-Root)!**

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

# Konfiguration vorbereiten — BITPANDA_API_KEY eintragen (für die Kurse nötig)
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

## Dashboard ("Panda Pro")

Läuft automatisch mit dem Bot. Im Browser öffnen: **http://127.0.0.1:8080**

- **Gesamtwert** mit Verlauf (1T / 1W / 1M / 1J / Max) — alle 5 Min. gespeichert
- **Bot-Karte** mit Schalter: aus = keine neuen Einstiege, Stop-Loss/Take-Profit laufen weiter.
  Die Pause bleibt auch nach einem Neustart bestehen.
- **Allokation**, **Strategien** (Ergebnis 30 Tage), **Risiko-Tacho** (0–100), **Letzte Aktivität**
- Aktualisiert sich alle 5 Sekunden, funktioniert auch am Handy

Einstellungen in `config/settings.yaml` unter `dashboard:`. Standardmäßig nur auf dem eigenen
Rechner erreichbar. Für den Zugriff vom iPhone im WLAN `host: 0.0.0.0` setzen **und**
`DASHBOARD_TOKEN` in `config/secrets.env` — ohne Token startet das Dashboard dann nicht.
Aufruf: `http://<IP-des-Macs>:8080/?token=<DASHBOARD_TOKEN>`

**Risiko-Wert:** 50 % Ausnutzung des Tages-Drawdown-Limits, 20 % belegte Positions-Slots,
20 % gebundene Margin, 10 % Verlustserie. Unter 25 niedrig, ab 75 hoch.

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

## Datenfeed im Paper-Modus: Bitpanda Fusion

Standard ist `general.data_feed: bitpanda` in `config/settings.yaml`. Der Bot holt Kurse,
Kerzen und Geld-/Briefkurse (Orderbuch) direkt von Bitpanda Fusion — also von der Börse,
auf der später gehandelt werden soll. Orders bleiben im Paper-Modus simuliert.

- Braucht `BITPANDA_API_KEY` in `config/secrets.env`, auch für reine Marktdaten
- Fehlt der Key, bricht der Bot beim Start mit einer klaren Meldung ab
- Alternative ohne Key: `data_feed: kraken`

Die Simulation hält sich an die Regeln von Bitpanda Fusion (`exchange_rules` in `config/settings.yaml`):
nur Kaufen/Verkaufen (keine Shorts, kein Hebel), 0,25 % Gebühr pro Order und die Mindestordergröße,
die Bitpanda beim Start pro Paar meldet (steht im `bot.log`: `BTC_EUR: min. Ordervolumen … EUR`).

## Datenfeed: One Trading (Live-Modus)

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

**Neues Multi-Factor Confluence System (Phase 1–6 Überarbeitung)**:
- 5+ Faktoren (Trend, Momentum, Breakout, Volume, Volatility, Sentiment/Fear&Greed, Macro/News)
- Regime Detection (trending / ranging / high_vol_event / low_vol_chop / event_driven)
- Adaptive Faktor-Gewichtung je Regime
- **Phase 6 Monitoring**: Detaillierte Factor-Attribution in Logs + Console + Telegram, Regime-Change-Alerts, Macro-Event-Alerts, täglicher Attribution-Report

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

**Live-Status Check (empfohlen):**
```bash
./deploy/status.sh <SERVER-IP>
```

Dieses Script (Phase 6) zeigt explizit, ob der Bot im **LIVE MODE** läuft, letzte Reconciliation-Ergebnisse und kritische Errors.

**Vor dem ersten Live-Einsatz unbedingt ausführen:**
```bash
python tools/paper_to_live_checklist.py
```
Siehe auch `LIVE_TRADING.md` für die vollständige Cutover-Checklist.
```

## Bitpanda / Real Portfolio Oversight (Phase 6)

Wenn du echtes Geld auf Bitpanda hast, nutze in dieser Grok-Session die integrierten `bitpanda-broker` MCP Tools für tägliche Checks:

- `get_portfolio` → Dein reales Exposure in EUR
- `list_trades` → Echte Trades abgleichen mit Bot-Logs
- `list_wallets` / `list_transactions`

Siehe [tools/bitpanda_mcp_oversight.md](tools/bitpanda_mcp_oversight.md) für den empfohlenen Workflow.

## Disclaimer

Dieser Bot ist **ausschließlich für Bildungszwecke und Paper-Trading** gedacht.

**Live-Trading mit echtem Kapital ist extrem riskant** und kann zum vollständigen Verlust führen.

Bevor du den Live-Modus aktivierst:
- Lies und verstehe `LIVE_TRADING.md`
- Setze mehrere explizite Sicherheits-Flags (siehe dort)
- Teste gründlich im Shadow-Modus + mit sehr kleinem Kapital

Crypto-Trading ist hochspekulativ. Nutze **niemals** Geld, dessen Verlust du dir nicht leisten kannst.
