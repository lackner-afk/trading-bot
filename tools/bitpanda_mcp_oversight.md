# Bitpanda MCP Tools – Oversight & Monitoring (Phase 6)

Dieses Dokument zeigt dir, wie du die integrierten **bitpanda-broker** MCP Tools in dieser Grok-Session nutzen kannst, um deinen echten Bitpanda-Bestand zu überwachen – perfekt als Ergänzung zum Trading-Bot.

## Warum das wichtig ist (für Profitabilität)

- Du siehst **dein echtes Exposure** in EUR (nicht nur den simulierten Bot-Balance).
- Du kannst vor/nach Live-Sessions schnell prüfen, was wirklich passiert ist.
- Du kannst echte Trade-Historie importieren und damit bessere Backtests / ML-Modelle trainieren.
- Früherkennung von Problemen (z.B. wenn der Bot und dein reales Wallet auseinanderdriften).

## Verfügbare Tools (in dieser Grok Session)

- `get_portfolio` → Gesamter Bestand mit EUR-Werten (sortierbar)
- `get_price` / `list_prices` → Aktuelle Kurse
- `list_trades` → Deine echten Buy/Sell Trades (mit Filtern)
- `list_wallets` → Alle Wallets mit Balances
- `list_transactions` → Ein- und Auszahlungen, Transfers etc.
- `get_asset` → Metadaten zu einem Asset

## Typische tägliche / wöchentliche Checks (empfohlen)

### 1. Schneller Portfolio-Überblick (vor/nach Live-Session)
```text
Use the bitpanda-broker MCP tool: get_portfolio with sort_by="value"
```

### 2. Letzte Trades anschauen (um Bot-Performance zu validieren)
```text
Use list_trades with limit=50, trade_type="sell" or operation="sell"
```

### 3. Aktuelle Preise deiner gehaltenen Assets
```text
Use list_prices with all_assets=true (oder spezifische Symbole)
```

### 4. Volle Transaktionshistorie für Analyse exportieren
Kombiniere `list_trades` + `list_transactions` und kopiere die Ausgabe in eine Datei für dein Backtesting/ML.

## Wichtiger Hinweis zu Credentials

Falls du aktuell die Fehlermeldung „Credentials / Access token wrong“ siehst:
- Die MCP-Verbindung ist in dieser Grok-Umgebung vorhanden, aber der Token ist nicht (mehr) gültig.
- Du musst den Bitpanda API-Zugriff in deinen Grok / MCP Einstellungen erneuern.
- Sobald das erledigt ist, funktionieren alle oben genannten Checks sofort.

## Empfohlener Workflow mit dem Bot

1. Vor Live-Session starten:
   - `get_portfolio` → Notiere dein aktuelles reales Exposure
2. Bot im Shadow-Mode laufen lassen (Phase 5)
3. Nach ein paar Tagen:
   - `list_trades` + Bot-Logs vergleichen
4. Bei Abweichungen → Reconciliation-Probleme oder manuelle Trades untersuchen

## Nächste Schritte (optional)

- Wir können später ein kleines Python-Skript `tools/fetch_bitpanda_data.py` bauen, das die echten Trades automatisch in dein `trades.db` oder ein separates Analyse-Format importiert.
- Oder Alerts bauen, die bei großen Abweichungen zwischen Bot und echtem Portfolio warnen.

---

**Stand:** Phase 6 – MCP-Integration für menschliche Oversight dokumentiert und vorbereitet.
