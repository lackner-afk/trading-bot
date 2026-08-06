# Testphase: kalibrieren, messen, dann live

Ablauf vom aktuellen Stand bis zum Live-Trading auf Bitpanda Fusion.

## Warum die Reihenfolge so ist

Die Richtung des Bots hängt faktisch am **Fear-&-Greed-Index**. Von den fünf
technischen Faktoren sind `volatility_filter` und `volume_confirmation`
richtungslos, `momentum` liefert nur bei RSI > 55 bzw. < 45 eine Richtung,
`breakout` nur bei echtem Ausbruch — und `multi_timeframe_trend` liegt auf 5m
fast immer am Score-Floor 0.08. Der `min_directional_score` von 0.35 wird
damit nur so passiert:

```
Fear (F&G < 45):  (0.08 Trend + 0.65 Sentiment) / 2 = 0.365  →  knapp über 0.35
Neutral (45–55):  Sentiment liefert direction=None → nur 0.08 →  blockiert
Greed (> 55):     Sentiment short → auf Spot nur Exit          →  keine Entries
```

Drei Konsequenzen:

1. **Die Trade-Rate ist ein An/Aus-Schalter.** 3–7 Trades/Tag in Fear-Phasen,
   0–1 sonst.
2. **Alle offenen Positionen sind perfekt korreliert** — derselbe Index kippt
   für alle Symbole gleichzeitig. Deshalb `max_position_size: 0.15` statt 0.20
   bei drei Slots.
3. **Gemessen wird im Kern „kaufe wenn F&G < 45"** mit technischen Filtern.

Deshalb steht die Kalibrierung vorne: sie beantwortet in Stunden, ob das nach
Gebühren bei Leverage 1 trägt — statt in sechs Wochen.

---

## Stufe 0 — Kalibrierung

### 1. Fear-&-Greed-Historie holen

```bash
python tools/fetch_fng_history.py
```

Einmalig. Ohne das würde der Backtest den **heutigen** F&G-Wert auf den
gesamten Zeitraum anwenden und damit genau den Faktor falsch messen, der die
Strategie steuert — äußerlich völlig unauffällig. `backtest.py` bricht
deshalb ab, wenn keine echte Historie vorliegt.

Die Ausgabe zeigt die Verteilung Fear/Neutral/Greed. **Der Fear-Anteil ist der
beste Frühindikator dafür, wie lange die Testphase bis 100 Trades braucht** —
bei 40 % Fear-Tagen sind rund 3–4 Trades/Tag realistisch, bei 15 % dauert es
entsprechend länger.

### 2. Backtest fahren

```bash
python backtest.py --data-exchange kraken --timeframe 5m --days 90
```

Prüfen: `Datenquelle: kraken` (nicht `simulated`) und dass die Confluence-
Strategie als erste getestet wird.

### 3. Kalibrieren

Grid über die vier Parameter, die Durchsatz und Qualität steuern:

| Parameter | Kandidaten | Wirkung |
|---|---|---|
| `min_confluence_score` | 0.50 / 0.55 / 0.60 / 0.65 | Gesamtschwelle |
| `min_directional_score` | 0.25 / 0.30 / 0.35 / 0.40 | Wie stark die Richtung sein muss |
| `trailing_arm_profit_pct` | 0.003 / 0.005 / 0.008 | Ab wann der Trailing-Stop scharf wird |
| `max_hold_hours` | 0 (aus) / 12 / 24 | Zeit-Stop |

`max_hold_hours` gehört dazu, weil ein seitwärts laufender Trade sonst einen
der drei Slots **unbegrenzt** blockiert — der wichtigste Turnover-Risikofaktor.

### Entscheidungspunkt

**Profit Factor nach Gebühren unter 1.0 → nicht in den sechswöchigen Lauf.**
Dann ist die Strategie das Problem, nicht die Messdauer. Das zu wissen ist das
eigentliche Ergebnis dieser Stufe.

---

## Stufe 1 — Paper-Dauerlauf (≥ 30 Tage)

### Deployment

```bash
ssh root@<IP> 'bash -s' < deploy/setup-server.sh    # einmalig
# config/secrets.env manuell auf dem Server anlegen — rsync schließt sie aus
./deploy/deploy.sh <SERVER-IP>
ssh root@<IP> 'systemctl status trading-bot'
```

`mode: paper` bleibt. Kraken liefert die Kurse, es fließt kein echtes Geld.

### Tag 1 — verifizieren, bevor die Uhr läuft

```bash
ssh botuser@<IP> 'tail -f ~/trading-bot/bot.log'
# oder seit S0.3 auch:
ssh root@<IP> 'journalctl -u trading-bot -f'
```

| Prüfung | Erwartung |
|---|---|
| `grep "\[CONFLUENCE CYCLE\]" bot.log \| tail` | alle 45 s, `Analyzed 5 symbols` |
| `grep "Starvation" bot.log` | leer |
| `grep -E "\[CONFLUENCE\] \|\[EXIT\]" bot.log` | erste Einträge nach einigen Stunden |

Bleibt der dritte Punkt über 48 h leer, obwohl F&G < 45 steht: nicht warten,
sondern `[AGGREGATOR REJECT]` ansehen — dort steht der genaue Grund.

### Laufender Betrieb

```bash
ssh botuser@<IP> 'cd ~/trading-bot && venv/bin/python tools/profitability_gate.py'
```

Wöchentlich. Zeigt „X von 11 Kriterien erfüllt" mit Ist/Soll je Kriterium.
Derselbe Block steht beratend im Tagesreport. Passiv laufen Telegram-
Stundenreport und Trade-Alerts.

**Wichtigste Regel: Parameter während des Laufs nicht anfassen.** Jede Änderung
an Schwellwerten, Positionsgröße oder Symbolen macht die bisher gesammelten
Trades unvergleichbar und setzt die Messung faktisch zurück. Dafür ist
Stufe 0 da.

### Abbruchkriterien

| Signal | Reaktion |
|---|---|
| `grep "KILL-SWITCH" bot.log` | Bot hat sich selbst gestoppt (5 fehlgeschlagene Exits). systemd startet **nicht** neu |
| `grep "EXIT FEHLGESCHLAGEN" bot.log` | Position ließ sich nicht schließen — höchste Priorität |
| `grep "KAPITAL ZU KLEIN" bot.log` | Equity unter ~67 € → jede Order unter dem Venue-Minimum, der Bot handelt dauerhaft nicht mehr |
| Max Drawdown > 10 % | Lauf beenden, Strategie überarbeiten |
| 7 Tage ohne Trade | F&G-Phase oder Konfigurationsproblem |

---

## Stufe 2 — Shadow Mode auf Fusion (2 Wochen, ≥ 50 Shadow-Trades)

Erst wenn das Gate **grün** ist:

```yaml
general:
  mode: live
  live_explicit_confirmation: true
live:
  venue: fusion
  shadow_mode: true      # bleibt true
```
```bash
export LIVE_TRADING_ENABLED=1
```

Beim Start läuft der **Preflight** gegen Pairs, Tickers und Balances. Bei
**401/403** liegt es am Key: `BITPANDA_API_KEY` braucht den `trade`-Scope und
muss ein v2-Key sein. Bei **404** stimmen die aus dem CLI abgeleiteten
URL-Pfade nicht — in `settings.yaml` korrigierbar, ohne Code anzufassen.
Danach die **Spot-Reconciliation**: Base-Asset-Bestand gegen lokale Positionen.

Ziel: echte Fusion-Kurse gegen Kraken-Kurse halten, Slippage aus den
`[EXEC-QUALITY]`-Zeilen gegen die Backtest-Annahme prüfen, Ordergrößen gegen
die echten `min_amount`/`min_notional`, und Reconciliation nach bewusstem
Neustart mehrfach testen.

Zusätzlich zu messen, wenn `live.quote_asset` auf EURCV steht: die
`[EXEC-QUALITY]`-Zeilen der EURCV-Paare gegen die Backtest-Annahme. Der
Backtest läuft auf Kraken-EUR-Daten; Stablecoin-Bücher sind dünner. Ist die
Slippage deutlich höher, kippt die Rechnung — die Strategie braucht 51,4 %
Trefferquote zum Break-even.

## Stufe 3 — Kleines Kapital (2–4 Wochen)

`shadow_mode: false`, 100–200 € echtes Kapital. Tägliche Fill-Kontrolle über
den read-only Bitpanda-MCP (`get_portfolio`, `list_trades`).

**Kapital trennen, bevor echtes Geld fließt.** Der Bot nimmt sonst den
gesamten EUR-Bestand des Kontos als Handelskapital — bei 5.000 € auf dem Konto
wären das 750 € pro Position statt der gedachten 15 €. Mit
`live.quote_asset: EURCV` handelt er `*-EURCV`-Paare und sieht ausschließlich
den EURCV-Bestand:

```yaml
live:
  quote_asset: EURCV
```

100 € in der Bitpanda-App in EURCV tauschen, fertig. Gewinne fließen wieder in
EURCV zurück, der Topf wächst also mit. Details und die Slippage-Frage in
`LIVE_TRADING.md`.

---

## Gegenprobe (durchgehend)

```sql
-- muss immer 0 sein: die Testphase misst nur, was live ausführbar ist
SELECT COUNT(*) FROM trades WHERE side='short' OR leverage != 1.0;
```

Das Go-Live-Gate prüft das ohnehin als elftes Kriterium.
