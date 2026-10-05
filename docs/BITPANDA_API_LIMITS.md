# Bitpanda API — Tägliche Handelslimits pro API-Key

Quelle: <https://docs.public.bitpanda.com/set-daily-trading-volume-limits-for-this-api-key-4461013e0>
(OpenAPI-Spezifikation, Stand Oktober 2026)

## Worum geht's?

Bitpanda erlaubt es, **pro API-Key** ein tägliches Volumen-Limit für Käufe und
Verkäufe festzulegen. Die Börse selbst lehnt dann Orders ab, sobald das
Tagesbudget aufgebraucht ist.

Für den Bot ist das ein **zweites, unabhängiges Sicherheitsnetz** auf
Börsenseite. Unsere Risk-Limits in `core/risk_manager.py` (2 % Risiko pro Trade,
10 % Daily Drawdown) greifen nur, solange der Code korrekt läuft. Das
Börsen-Limit greift auch dann, wenn der Bot spinnt (Endlosschleife, doppelte
Orders, Bug in der Positionsgröße, geleakter Key).

## Endpoint

| | |
|---|---|
| Methode | `POST` |
| URL | `https://api.public.bitpanda.com/v1/trading-limits` |
| Auth | Header `x-api-key: <dein Key>` |
| Content-Type | `application/json` |
| Wirkung | Limit gilt für **genau den Key**, mit dem der Request geschickt wird |

> ⚠️ Achtung: Das ist der Host `api.public.bitpanda.com` — **nicht**
> `api.fusion.bitpanda.com`, den `data/fusion_feed.py` für Marktdaten nutzt.
> Ob derselbe Key für beide APIs gilt, ist noch nicht geprüft.

### Request-Body (alle Felder Pflicht)

| Feld | Typ | Bedeutung |
|------|-----|-----------|
| `buy_limit` | number | Maximales Kaufvolumen pro Tag |
| `sell_limit` | number | Maximales Verkaufsvolumen pro Tag |
| `currency_id` | string (UUID) | Währung, in der die Limits angegeben sind |

Beispiel aus der Doku:

```json
{
  "buy_limit": 1000,
  "sell_limit": 500,
  "currency_id": "b88b8466-efe3-11eb-b56f-0691764446a7"
}
```

Die UUID im Beispiel ist **nicht dokumentiert** — welche Währung sie meint
(vermutlich EUR), muss vor dem echten Setzen geprüft werden.

### Antworten

| Status | Bedeutung |
|--------|-----------|
| `200` | Limits gesetzt. Antwort enthält `data` mit `buy_limit`, `sell_limit`, `currency_id`, **`buy_budget_remaining`**, **`sell_budget_remaining`** (Restbudget für heute) |
| `400` | Ungültige Parameter → `{"error": {"code": "..."}}` |
| `409` | **Für diesen Key sind schon Limits gesetzt** → `{"error": {"code": "..."}}` |
| `500` | Serverfehler → `{"error": {"code": "..."}}` |

Beispiel `200`:

```json
{
  "data": {
    "buy_limit": 1000,
    "sell_limit": 500,
    "currency_id": "b88b8466-efe3-11eb-b56f-0691764446a7",
    "buy_budget_remaining": 1000,
    "sell_budget_remaining": 500
  }
}
```

## Wichtige Eigenheiten

1. **Einmal setzen, nicht überschreiben.** `POST` auf einen Key mit
   bestehenden Limits liefert `409`. Ändern/Löschen läuft über andere
   Endpoints im Ordner „Trading Limits“ der Doku — die sind hier noch **nicht**
   dokumentiert. Also: Werte vorher gut überlegen.
2. **Nicht dokumentiert** sind: Zeitpunkt des Tages-Resets (UTC? Ortszeit?),
   ob Gebühren mitzählen und welcher Fehlercode bei einer Order kommt, die das
   Limit überschreitet. Bitte beim ersten Live-Test beobachten und hier
   nachtragen.
3. Die Limits hängen am **Key**, nicht am Konto. Ein zweiter Key (z. B. für
   manuelles Handeln) hat eigene Limits — oder gar keine.

## Empfohlene Werte für den Bot

> **Faustregel: `sell_limit` großzügiger als `buy_limit`.**
> Wenn das Verkaufsbudget aufgebraucht ist, kann der Bot offene Positionen
> **nicht mehr schließen** — Stop-Loss und Take-Profit laufen dann ins Leere.
> Ein zu knappes Kauflimit kostet nur entgangene Trades, ein zu knappes
> Verkaufslimit kann echtes Geld kosten.

Grobe Rechnung (Spot, kein Hebel):

```
buy_limit  ≈ Kapital × max_position_size × erwartete Einstiege pro Tag × Puffer
sell_limit ≈ buy_limit + Wert aller gehaltenen Coins
```

Beispiel mit den aktuellen Settings (`start_capital: 100`,
`max_position_size: 0.20`, `max_concurrent_positions: 2`, Momentum-Cooldown
900 s auf 3 Paaren):

| | Wert | Begründung |
|---|---|---|
| `buy_limit` | **150 EUR** | ca. 7 volle Einstiege à 20 EUR + Puffer; ein Bot, der mehr kauft, hat einen Bug |
| `sell_limit` | **300 EUR** | doppelt so viel — Schließen muss immer gehen |

Bei mehr Kapital die Werte proportional hochskalieren. Lieber klein starten:
Ein Limit zu erhöhen ist harmlos, ein Bug ohne Limit nicht.

## Setzen per Skript

```bash
# Trockenlauf — zeigt nur, was geschickt würde
python tools/bitpanda_trading_limit.py --buy 150 --sell 300 --currency-id <UUID>

# Wirklich setzen (braucht BITPANDA_API_KEY in config/secrets.env)
python tools/bitpanda_trading_limit.py --buy 150 --sell 300 --currency-id <UUID> --confirm
```

Oder direkt mit curl:

```bash
curl -X POST https://api.public.bitpanda.com/v1/trading-limits \
  -H "x-api-key: $BITPANDA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"buy_limit": 150, "sell_limit": 300, "currency_id": "<UUID>"}'
```

## Offene Punkte

- [ ] UUID für EUR herausfinden (Currencies-Endpoint der Bitpanda-API)
- [ ] Endpoints zum Abfragen/Ändern/Löschen der Limits dokumentieren
- [ ] Reset-Zeitpunkt und Fehlercode bei Überschreitung beobachten
- [ ] Prüfen, ob der Fusion-Key (`BITPANDA_API_KEY`) auch auf `api.public.bitpanda.com` gilt
- [ ] Live-Order-Engine: Ablehnung wegen Limit erkennen, Telegram-Alarm schicken, Kaufen pausieren
