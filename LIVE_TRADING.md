# ⚠️ LIVE TRADING — KRITISCHE SICHERHEITSANLEITUNG ⚠️

**DIESER BOT HANDELT MIT ECHTEM GELD, WENN DU IHM DAS ERMÖGLICHST.**

Der Trading-Bot wurde ursprünglich **ausschließlich als Paper-Trading-System** entwickelt.  
Es gibt **keine Garantie** auf Profit. Krypto-Trading ist extrem risikoreich und kann zu **vollständigem Kapitalverlust** führen.

---

## Aktueller Status (nach Phase 7)

Der Bot hat jetzt eine **komplette, sichere Live-Architektur**:

- ✅ One Trading CCXT Feed (Phase 1)
- ✅ LiveOrderEngine mit Shadow Mode (Phase 2 + 5)
- ✅ Reconciliation & Startup-Sync (Phase 3)
- ✅ Sauberes Live-Mode Wiring in main.py mit starken Guardrails (Phase 4)
- ✅ Data Parity + Execution Quality Logging (Phase 5)
- ✅ Verbesserte Deploy & Status-Tools + MCP Oversight (Phase 6)
- ✅ **Faktor-Attribution & Monitoring** (Logs, Rich-Console-Tabellen, Telegram "Warum dieser Trade?", Regime-Change-Alerts, Macro-Event-Alerts, täglicher Factor-Performance-Report)
- ✅ Vollständige Dokumentation & Cutover-Checklist (Phase 7)

Trotzdem gilt weiterhin: **Krypto-Trading mit Hebel ist extrem riskant.** Es gibt keine Garantie auf Gewinne.

---

## Wie du den Live-Modus aktivierst (Mehrstufige Zwangssicherung)

Der Code erzwingt **mehrere unabhängige Hürden**. Siehe auch das Tool `tools/paper_to_live_checklist.py`.

### Erforderliche Flags
```yaml
# config/settings.yaml
general:
  mode: live
  live_explicit_confirmation: true
```

```bash
export LIVE_TRADING_ENABLED=1
```

### Die Hürden im Einzelnen (in dieser Reihenfolge geprüft)

| # | Hürde | Wo |
|---|-------|-----|
| 1 | `LIVE_TRADING_ENABLED` gesetzt | `main.py::start` |
| 2 | `live_explicit_confirmation: true` | `main.py::start` |
| 3 | **Profitabilitäts-Gate grün** | `main.py::_check_profitability_gate` |
| 4 | 10-Sekunden-Countdown mit CRITICAL-Logs | `main.py::start` |
| 5 | Startup-Reconciliation erfolgreich | `core/reconciliation.py` |

Hürde 3 ist neu und der einzige Check, der die **tatsächliche Performance**
ansieht: elf Kriterien gegen die echte Trade-Historie in `trades.db` (siehe
`tools/profitability_gate.py`, Schwellen im `go_live_gate:`-Block). Jederzeit
manuell prüfbar mit:

```bash
python tools/profitability_gate.py
```

Übergehbar ist das Gate nur explizit über `general.skip_profitability_gate: true`
— und dann mit lautem CRITICAL-Log. Das ist bewusst unbequem.

Zusätzlich steht `live.shadow_mode` per Default auf `true`: der Bot loggt
im Live-Modus exakt, was er tun würde, platziert aber keine echten Orders.
Erst nach einer sauberen Shadow-Phase auf `false` setzen.

**Nur wenn alle Hürden genommen sind, darf der Bot echte Orders platzieren.**

---

## Kapital-Risiko-Policy (verbindlich)

Bevor du echtes Geld einsetzt, musst du folgendes **schriftlich** für dich klären:

- Wie viel EUR bist du maximal bereit, **komplett zu verlieren**? (z.B. 500 €, 2000 €)
- Welcher Prozentsatz deines Gesamtvermögens ist das?
- Hast du den Bot mindestens **4–6 Wochen im Shadow-Modus** (keine echten Orders, aber Live-Daten + LiveOrderEngine simuliert) laufen lassen?
- Hast du Reconciliation nach Crashes / Restarts gründlich getestet?
- Hast du die tatsächlichen Fees + Slippage von One Trading mit deinen Strategie-Annahmen verglichen?

**Empfohlener Rollout-Pfad:**

1. **Paper-Modus** (aktuell) — beliebig lange
2. **Shadow / Dry-Run Live** — Live-Daten + LiveOrderEngine, aber Orders werden nur geloggt, nicht ausgeführt (wird implementiert)
3. **Small-Capital Validation** — max. 1–5 % deiner geplanten Live-Capital (z.B. 200–500 € Risiko) für mindestens 2–4 Wochen mit täglicher manueller Kontrolle
4. **Skalierung** — nur nach positiver Validierung und nur schrittweise

---

## Notfall-Stopp

Sollte etwas schiefgehen:

- Bot-Prozess sofort killen (Ctrl+C oder `kill` des PID)
- Auf der Exchange (One Trading) manuell alle offenen Orders stornieren
- Positionen ggf. manuell schließen
- Logs + trades.db sichern

Später wird es zusätzliche automatische Kill-Switches geben (max daily DD, API-Error-Rate, etc.).

---

## Was aktuell noch fehlt

- **Bitpanda Fusion ist nicht angebunden.** Der einzige implementierte
  Exchange-Zugang ist `ccxt.onetrading` (One Trading, ehemals Bitpanda Pro —
  seit 2023 ein eigenständiges Unternehmen, *nicht* Bitpanda). CCXT
  unterstützt Fusion nicht (Issue #25354, offen seit Feb 2025, kein PR), die
  Anbindung braucht also einen eigenen REST-Client gegen die Fusion-API.
- **Es gibt zwei verschiedene Bitpanda-MCPs — nicht verwechseln:**
  - `bitpanda-labs/bitpanda-mcp` (Public-/Broker-API) ist **read-only**:
    `get_portfolio`, `list_wallets`, `get_price`, `list_prices`, `get_asset`,
    `list_transactions`, `list_trades`. Auth über `BITPANDA_API_KEY`.
    Gut für Oversight, kann keine Orders platzieren.
  - **Fusion MCP** kann traden. Bitpanda hat am 16.07.2026 API *und* MCP für
    automatisiertes Trading auf Fusion gelauncht; laut Ankündigung lassen sich
    darüber Orders platzieren, Positionen abrufen und das Buch verwalten.
    Auth über einen eigenen Fusion-API-Key (`FUSION_API_KEY`).
- **Order-Reconciliation ist ein Platzhalter.** `_reconcile_open_orders` zählt
  offene Orders und loggt sie; der Abgleich offline gefüllter Orders fehlt.
  `_reconcile_positions` warnt nur, statt wirklich zu vergleichen. Im
  Spot-Modell *ist* die Base-Asset-Balance die Position — damit wäre ein
  echter Abgleich implementierbar.
- **Keine automatischen Kill-Switches** ausser dem Exit-Fehlschlag-Zähler
  (`MAX_EXIT_FAILURES`) und dem Tagesdrawdown-Limit.

---

## Bitpanda Fusion: API-Oberfläche

Abgeleitet aus dem offiziellen CLI [`bitpanda-labs/bitpanda-fusion-cli`](https://github.com/bitpanda-labs/bitpanda-fusion-cli)
(Go, Apache 2.0). Die Doku unter `docs.fusion.bitpanda.com` antwortet auf
automatisierte Abrufe mit HTTP 403 — das CLI-README ist die belastbarste
öffentlich zugängliche Quelle.

| Punkt | Wert |
|-------|------|
| Base-URL | `https://api.fusion.bitpanda.com` |
| Auth | `FUSION_API_KEY` (eigener Key, **getrennt** vom read-only `BITPANDA_API_KEY`) |
| **Paar-Format** | `BTC-EUR` (Bindestrich!) — nicht `BTC/EUR`, nicht `BTC_EUR` |
| Ordertypen | `limit`, `market` |
| Ordergröße | `quantity` (Base) **oder** `amount` (Quote, z.B. 30 EUR) — exklusiv |
| Order-Status | `open`, `closed`, `new`, `partially-filled`, `filled`, `canceled`, `filled-and-canceled`, `done-for-day`, `rejected` |
| Candles | OHLCV vorhanden; Intervalle `1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d`, `limit` max 1440, `from`/`to` als RFC3339 |
| Instrument-Metadaten | Endpoint für Trading-Pairs liefert min/max Ordergröße, Tick-Size, Increments |
| Weiteres | Orderbook (Tiefe 1–100), Tickers (Mid + 24h), Balances, Gebührenstaffel/30d-Volumen, Trades-Historie |

Damit sind drei zuvor offene Fragen beantwortet:

- **Fusion liefert OHLCV** inklusive 5m — der Feed braucht keine Kraken-Daten
  und es entsteht kein Preisbasis-Risiko.
- **Das Symbol-Mapping** ist `BTC_EUR` → `BTC-EUR`.
- **Mindestordergröße** ist pro Paar über den Pairs-Endpoint abrufbar; die
  Rundung auf Tick-Size/Increments muss vor jedem Order-Versand passieren
  (häufigste Ursache für Live-Rejects).

Offen bleibt: Zeigt das read-only Broker-Konto denselben Bestandstopf wie
Fusion? Davon hängt ab, ob der Broker-MCP als Reconciliation-Quelle taugt.

### Spot-only

Kein Leverage, keine Shorts (Margin ist bei Bitpanda als "coming soon"
angekündigt); weder CLI noch API-Oberfläche kennen entsprechende Parameter.
Der Bot läuft deshalb auch im Paper-Modus long-only mit Leverage 1 — siehe
`trading:`-Block in `settings.yaml`.

### Ausführungsweg: REST direkt

Umgesetzt ist der direkte REST-Weg (`core/fusion_client.py` +
`core/bitpanda_fusion_engine.py`). Für einen Dauerläufer-Bot ist das die
native Schnittstelle: kein Zusatzprozess, deterministisches Fehlerverhalten,
volle Kontrolle über Retries und Idempotenz — wichtig in einem Pfad, der
Stop-Loss-Orders zuverlässig absetzen muss.

Der **Fusion-MCP** bleibt parallel nutzbar, um aus Claude oder Cursor heraus
manuell auf dasselbe Konto zuzugreifen. Beide Wege enden bei derselben
Execution-Engine von Fusion.

### Was implementiert ist

| Komponente | Datei |
|---|---|
| REST-Client (Auth, Backoff, Antwort-Normalisierung) | `core/fusion_client.py` |
| Order-Engine (Market/Limit, Präzision, Idempotenz, Shadow) | `core/bitpanda_fusion_engine.py` |
| Marktdaten-Feed (Tickers 5s, Candles 60s) | `data/bitpanda_fusion_feed.py` |
| Spot-Reconciliation (Bestand = Position) | `core/spot_reconciliation.py` |
| Gemeinsames Engine-Interface | `core/execution_base.py` |
| Zentrales Symbol-Mapping | `data/symbols.py` |

Aktiviert über `live.venue: fusion` in `settings.yaml`. Key kommt aus
`FUSION_API_KEY`.

### Preflight — bevor scharf geschaltet wird

Die URL-Pfade in `live.endpoints` sind aus dem CLI abgeleitet, **nicht** gegen
die Doku verifiziert (die blockt automatisierte Abrufe). Beim Live-Start läuft
deshalb zuerst ein Preflight gegen Pairs, Tickers und Balances. Schlägt er
fehl, startet der Bot nicht — dann sind entweder die Pfade oder der
`auth_header` in `settings.yaml` zu korrigieren. Beides ist reine Konfiguration,
ohne Codeänderung.

`live.shadow_mode` steht per Default auf `true`: die Engine loggt exakt, was
sie tun würde, schickt aber nichts los. Erst nach grünem Preflight und einer
sauberen Shadow-Phase auf `false` setzen.

---

## Zusammenfassung für dich

> **"Ich will den Bot live mit echtem Geld laufen lassen."**
>
> → Dann musst du **bewusst und mehrfach** mehrere Sicherheitsmechanismen umgehen.
> → Der Code wird dich **absichtlich** so stark wie möglich davon abhalten.
> → Das ist kein Bug, das ist gewollt.

Wenn du das verstanden hast und trotzdem weitermachen willst: gut.  
Wir bauen die Schutzmaßnahmen jetzt schrittweise ein.

---

## ✅ Paper → Live Cutover Checklist (Phase 7)

**Führe diese Checkliste vollständig ab, bevor du echtes Geld riskierst.**

### Stufe 0: Vorbereitung (Paper Mode)
- [ ] Alle Strategien mindestens 30–60 Tage stabil im Paper-Mode gelaufen
- [ ] Backtests mit `data_exchange: onetrading` oder `kraken` durchgeführt (Data Parity)
- [ ] Execution Quality Logs analysiert (Slippage, Rejection Rate, Latency)
- [ ] `tools/paper_to_live_checklist.py` ohne Fehler durchgelaufen

### Stufe 1: Shadow Mode (empfohlen 2–4 Wochen)
- [ ] Bot im Live-Modus gestartet **aber** mit `shadow_mode: true`
- [ ] Mindestens 100 simulierte Trades mit realen Marktdaten
- [ ] Reconciliation nach Neustarts / Crashes mehrfach erfolgreich getestet
- [ ] Telegram + Logs regelmäßig auf Abweichungen geprüft
- [ ] `get_portfolio` (via bitpanda-broker MCP) regelmäßig mit Bot-Logs verglichen

### Stufe 2: Small Capital Validation (1–5 % deines geplanten Risikokapitals)
- [ ] `live_explicit_confirmation: true` + `LIVE_TRADING_ENABLED` gesetzt
- [ ] Max. 200–500 € echtes Risiko (je nach deinem Gesamtvermögen)
- [ ] Tägliche manuelle Überprüfung der Fills + Reconciliation
- [ ] Stop-Loss / Drawdown-Limits in der Praxis beobachtet
- [ ] Mindestens 2–4 Wochen ohne größere negative Überraschungen

### Stufe 3: Skalierung (nur nach erfolgreicher Validation)
- [ ] `paper_to_live_checklist.py` besteht vollständig
- [ ] Alle offenen Fragen aus dem Plan geklärt (Kapital, Order-Typen, Spot vs. Futures)
- [ ] Notfallprozedur (manuelles Schließen aller Positionen) geübt
- [ ] Langfristiges Monitoring & Alerting eingerichtet

---

## Notfallprozedur (Live Mode)

1. Bot sofort stoppen: `systemctl stop trading-bot`
2. Auf One Trading manuell alle offenen Orders stornieren
3. Bei Bedarf Positionen manuell schließen
4. Logs + `trades.db` sichern
5. `get_portfolio` via MCP ziehen und mit Bot-Logs abgleichen

---

**Stand dieser Datei:** Phase 7 – Vollständige Cutover-Checklist + Dokumentation abgeschlossen.

**Du bist jetzt offiziell bereit für verantwortungsvolles Live-Trading – aber nur, wenn du die Checkliste wirklich durchläufst.**