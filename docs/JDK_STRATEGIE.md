# JDK-Orderflow-Strategie

Nachbau des Trading-Ansatzes von **JDK Analysis** (X: [@The_JDK99](https://x.com/The_JDK99),
TradingView: [JDK-Analysis](https://www.tradingview.com/u/JDK-Analysis/)).
Code: `strategies/jdk_orderflow.py`, Config: `strategies.jdk` in `config/settings.yaml`.

## Was JDK macht (aus seinen öffentlichen Posts)

- **Key Levels statt Indikator-Signale.** Er markiert Levels aus Volumenprofil und
  VWAPs und wartet, bis der Preis dort ankommt. Beispiel: *„Watching a 66K test for a
  potential long if OrderFlow confirms — 50% level, AVWAP uptrend, rVAL and nPOC."*
- **Orderflow als Auslöser.** Am Level wird nicht blind gekauft: *„Now watching
  OrderFlow very closely for potential signs of strength at this key level."*
  Er liest Footprint-Charts, CVD und Absorption (Exocharts).
- **Auktionslogik / Value Area.** *„Price is still trading below the overall range VAL.
  Last week's reclaim attempt failed and ended in a smaller local failed auction. If
  bulls can reclaim this level and accept back above it, we could see a larger move."*
- **Session-VWAP als Kontext.** *„With price holding above session VWAP, price action
  hasn't confirmed weakness."*
- **HTF-Level:** Halving-anchored VWAP, Uptrend-AVWAP, LVNs aus dem Tages-Volumenprofil.
- Er handelt selten und nur mit Bestätigung: *„For now, no new trades triggered on my side."*

## Umsetzung im Bot

Zeitrahmen **1h**, Range = die letzten **240 Kerzen (10 Tage)**.

| Level | Berechnung |
|---|---|
| VAL / POC / VAH | Volumenprofil der Range (ohne die letzten 12 Kerzen), Value Area 70 % |
| nPOC | POC jedes abgeschlossenen UTC-Tages, der seither nicht mehr gehandelt wurde |
| LVN | lokale Minima im Profil, < 35 % des POC-Volumens |
| AVWAP↑ | VWAP ab dem tiefsten Tief der Range (Beginn des Aufwärtstrends) |
| Session-VWAP | VWAP seit 00:00 UTC |
| Range-50 % | Mitte zwischen Range-Hoch und -Tief |

**Orderflow-Näherung:** CCXT liefert keine Footprint-/CVD-Daten. Das Delta pro Kerze wird
aus OHLCV geschätzt (`Volumen × Lage des Schlusskurses in der Kerze`), das CVD ist die
Summe davon. Das ist deutlich gröber als sein Footprint. Echte Trade-Daten (Taker-Seite)
wären der nächste Ausbauschritt.

### Setup 1: Level-Test (`level_test`)

1. Preis schließt **nicht** unter der Range-VAL (sonst bärisch, siehe Setup 2).
2. Die Kerze testet eine Zone, in der **mindestens 2 verschiedene Level-Typen** innerhalb
   von `max(0,35×ATR; 0,2 %)` liegen.
3. Orderflow-Bestätigung, mindestens eins davon:
   - **Ablehnung:** Docht unter die Zone, Schluss darüber, Docht ≥ 40 % der Kerze, Kaufdelta
   - **Absorption:** neues Tief, aber das CVD macht ein höheres Tief
4. Stop 0,25×ATR unter Zone/Docht, Ziel = das erste Level darüber mit **CRV ≥ 2 nach Gebühren**.

### Setup 2: Failed Auction / VAL-Reclaim (`failed_auction`)

1. Der Preis schloss zuletzt unter der Range-VAL (höchstens 24 Kerzen lang, sonst
   Akzeptanz darunter).
2. Jetzt **2 Schlusskurse in Folge** wieder über der VAL (Akzeptanz), mit Kaufdelta.
3. Der Ausflug war höchstens 4×ATR tief (tiefer = Range-Bruch).
4. Stop unter dem Deviation-Tief, Ziel POC → VAH → Range-Hoch (erstes mit CRV ≥ 2).

### Trade-Management

- **Break-Even:** Ab +1R wandert der Stop auf Einstieg + Round-Trip-Gebühren.
- **Cooldown:** 6 Kerzen pro Symbol nach einem Signal.
- **Spot:** `long_only: true`, `leverage: 1` (Bitpanda Fusion kann weder Shorts noch Hebel).
  Shorts sind implementiert (gespiegelte Logik: VAH-Ablehnung, Failed Auction über VAH)
  und lassen sich per `long_only: false` für Paper-Tests aktivieren.

## Backtest

    python tools/backtest_jdk.py                       # 90 Tage, Binance EUR-Paare, 0,25 % Gebühr
    DAYS=90 BACKTEST_END=2026-05-20 python tools/backtest_jdk.py
    JDK_MIN_RR=2.5 JDK_MIN_CONFLUENCE_LEVELS=3 python tools/backtest_jdk.py
    CSV_DIR=daten/ python tools/backtest_jdk.py        # eigene Kerzen (BTC_EUR.csv …)

**Status:** Bisher nur auf synthetischen Daten als Funktionstest gelaufen, weil die
Cloud-Umgebung, in der die Strategie gebaut wurde, keinen Zugriff auf Börsen-APIs hatte.
Vor jedem Einsatz über die drei Fenster aus `docs/BACKTEST_BEFUNDE.md` laufen lassen.

## Bekannte Grenze: Positionsgröße vs. Mindestorder

Mit 100 € Kapital, 2 % Risiko und der 20-%-Grenze aus dem RiskManager wird eine
Position ~20 € groß. Bitpanda Fusion verlangt **25 €** je Order. Im Paper-Modus fällt
das nicht auf. Der Backtest weist darauf hin. Für den Live-Betrieb muss entweder mehr
Kapital her oder die Positionsgrenze angepasst werden.
