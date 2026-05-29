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
export LIVE_TRADING_ENABLED=1          # oder mit Datum: 2026-06-02
```

Beim Start erscheint ein **lauter 10-Sekunden-Countdown** mit CRITICAL-Logs. Reconciliation muss erfolgreich sein.

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

## Was aktuell (Phase 0) noch fehlt (wichtig!)

- Kein `LiveOrderEngine` (echte Orders via CCXT onetrading)
- Keine Reconciliation-Logik (was passiert nach Bot-Crash mit offenen Positionen?)
- Keine echte Balance-Sync mit der Exchange
- Backtest verwendet teilweise andere Datenquellen als Live (Parity-Problem)
- Keine Shadow-Trading-Funktion

**Solange diese Punkte nicht implementiert und getestet sind, ist Live-Trading fahrlässig.**

---

## Nächste Schritte (laut Plan)

Siehe [plan.md](../.grok/sessions/%2FUsers%2Fnici/019e74ea-2dc2-7e03-98fb-3ef5a3215e9d/plan.md) → Phase 1–4 für die eigentliche Live-Execution-Implementierung.

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