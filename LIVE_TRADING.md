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
- **Der offizielle Bitpanda MCP-Server kann nicht traden.** Er ist strikt
  read-only (`get_portfolio`, `list_wallets`, `get_price`, `list_prices`,
  `get_asset`, `list_transactions`, `list_trades`); die Doku sagt ausdrücklich,
  er könne keine Orders platzieren. Er taugt als Kontrollinstanz für
  Reconciliation, nicht als Ausführungsweg.
- **Order-Reconciliation ist ein Platzhalter.** `_reconcile_open_orders` zählt
  offene Orders und loggt sie; der Abgleich offline gefüllter Orders fehlt.
  `_reconcile_positions` warnt nur, statt wirklich zu vergleichen. Im
  Spot-Modell *ist* die Base-Asset-Balance die Position — damit wäre ein
  echter Abgleich implementierbar.
- **Keine automatischen Kill-Switches** ausser dem Exit-Fehlschlag-Zähler
  (`MAX_EXIT_FAILURES`) und dem Tagesdrawdown-Limit.

---

## Bitpanda Fusion: bekannte Randbedingungen

- **Spot-only.** Kein Leverage, keine Shorts (Margin ist bei Bitpanda als
  "coming soon" angekündigt). Der Bot läuft deshalb auch im Paper-Modus
  long-only mit Leverage 1 — siehe `trading:`-Block in `settings.yaml`.
- **Eigene REST-API**, dokumentiert unter `docs.fusion.bitpanda.com` bzw.
  `techsolutions.bitpanda.com`. Beide antworten auf automatisierte Abrufe mit
  HTTP 403; für die Implementierung wird die OpenAPI-Spezifikation aus dem
  eingeloggten Browser gebraucht.
- **Getrennte Keys.** Ein künftiger `BITPANDA_FUSION_API_KEY` (Trading) muss
  strikt vom read-only `BITPANDA_API_KEY` (MCP/Broker-API) getrennt bleiben.
- **Offene Fragen vor der Anbindung:** Liefert Fusion OHLCV, oder muss der
  Feed Marktdaten von Kraken beziehen (mit Preisbasis-Risiko)? Zeigt das
  read-only Broker-Konto denselben Bestandstopf wie Fusion? Liegt eine
  20-EUR-Position (20 % von 100 EUR) über der Mindestordergröße?

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