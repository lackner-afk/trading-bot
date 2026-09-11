# Trendfolge auf Tagesbasis — Strategiewechsel vom 11.09.2026

Die Confluence-Strategie (5-Minuten-Kerzen, Multi-Faktor) ist abgeschaltet. An
ihre Stelle tritt eine langsame Trendfolge auf Tageskerzen: `strategies/daily_trend.py`.

## Warum

Die Messung vom 24.08.2026 (`BACKTEST_BEFUNDE.md`) hat gezeigt: Unter den echten
Bedingungen von Bitpanda Fusion verliert jede schnelle Strategie.

| Bedingung auf Fusion | Folge |
|---|---|
| 0,25 % Gebühr je Seite, kein Maker-Rabatt | 0,50 % je Round-Trip |
| Spot: kein Short, kein Hebel | Nur Long/Flat möglich |
| 25 € Mindestorder | Kleine Konten können nur wenige Positionen halten |
| 5m-Strategie mit 100–500 Trades je Quartal | Gebühr frisst 50–250 % des Kapitals pro Jahr |

Was in der Literatur über viele Marktphasen hält, ist langsame Trendfolge:
im Aufwärtstrend investiert sein, sonst Cash. Der robuste Befund ist nicht
„mehr Rendite als Halten", sondern „ähnliche Rendite bei deutlich kleinerem
Drawdown". Der eigene Backtest zeigte genau das: im Abwärtsfenster
Nov 2025–Feb 2026 verlor der Trendansatz 7 %, Halten 28 %.

Belege, kritisch gelesen:

- Zeitreihen-Momentum auf Krypto (akademisch, Monats-/Wochenhorizont): Sharpe
  um 1,5 gegen 0,8 für den Markt. Stichproben meist vor 2021.
- Grayscale, 50-Tage-Durchschnitt long/flat 2012–2023: Sharpe 1,9 gegen 1,3.
  Ohne Gebühren gerechnet, Anbieter mit Bitcoin-Interesse, 2012–2017 dominiert.
- Kurzfristige Technik-Regeln (EMA-Cross, RSI, Bollinger) halten nach Korrektur
  für Data-Snooping in neueren Stichproben nicht (Wei 2024; Finance Research
  Letters 2020; BTC/ETH 2022–2023).

Kurz: Die Strategie kann Verluste begrenzen. Ob sie Halten schlägt, hängt vom
Zeitraum ab. Wer mehr verspricht, verkauft etwas.

## Regeln

Alles auf **abgeschlossenen Tageskerzen (UTC)**. Die laufende Kerze wird
abgeschnitten, damit Bot und Backtester dieselben Daten sehen.

| Schritt | Regel | Parameter |
|---|---|---|
| Einstieg | Tagesschluss > SMA(n) × (1 + Puffer) und SMA steigt | `ma_days: 100`, `entry_buffer_pct: 0.01`, `slope_days: 10` |
| Ausstieg | Tagesschluss < SMA(n) × (1 − Puffer) | `exit_buffer_pct: 0.02` |
| Notstopp | Live-Preis ≤ Einstieg × (1 − max_loss) | `max_loss_pct: 0.08` |
| Größe | Wunschanteil, dann harte RiskManager-Grenzen, dann Cash | `allocation_pct: 0.20` |
| Ausführung | Signal am Tagesschluss, Order zur nächsten Eröffnung | `interval_seconds: 3600` |

Die Puffer (Hysterese) verhindern das Hin und Her um die SMA, das bei 0,50 %
Round-Trip die Rendite auffrisst. Kein Take-Profit: der Trend läuft, bis er bricht.

## Die Kapitalgrenze

Der RiskManager kappt jede Position hart bei 20 % des Eigenkapitals und 2 %
Risiko je Trade. Das ist Absicht und bleibt.

    Mindestorder 25 € / 20 % = 125 € Eigenkapital

Unter 125 € kann der Bot **keine Order platzieren**. Er startet, loggt die
Warnung und beobachtet. Mit den aktuellen 100 € Startkapital passiert also
nichts, bis entweder Kapital nachkommt oder die Kappung bewusst geändert wird.
Letzteres ist eine Entscheidung des Kontoinhabers, nicht der Software.

Mit zwei Paaren und 20 % je Paar sind maximal 40 % des Kapitals investiert.
Die Rendite gegenüber vollem Halten ist entsprechend gedämpft, der Drawdown auch.

## Backtest — vor jedem Live-Schritt

    python tools/backtest_daily_trend.py                       # Binance ab 2018, 100 €
    python tools/backtest_daily_trend.py --capital 100,250,500 # Effekt der Mindestorder
    python tools/backtest_daily_trend.py --ma 50,100,150,200   # Robustheit je SMA
    FUSION_API_KEY=... python tools/backtest_daily_trend.py --source fusion

Der Backtester nutzt **dieselbe Funktion** `compute_state` wie der Bot, rechnet
Gebühr, Slippage, Mindestorder, die harten Risikogrenzen und den Notstopp gegen
das Tagestief. Er gibt je SMA-Länge und Kalenderjahr aus: Return, Buy & Hold,
Max-Drawdown, Sharpe, Trades, Gebühren, investierter Anteil, übersprungene Orders.

`--max-loss` ist der erste Parameter, den man auf echten Daten prüfen sollte:
Ein enger Notstopp (8 %) erlaubt die volle 20 %-Größe, wird aber öfter
ausgelöst. Ein weiter Stopp (15–25 %) wird selten getroffen, schrumpft die
Position aber über die 2 %-Risikoregel auf 13 % bzw. 8 % des Kapitals.

Regeln zum Lesen der Zahlen:

1. Ein Parameter zählt nur, wenn er **in allen Jahresfenstern** nicht deutlich
   schlechter ist als Halten und im schlechtesten Jahr den Verlust begrenzt.
   Das beste Einzeljahr ist irrelevant.
2. Springt die optimale SMA-Länge je Fenster stark, ist der Parameter angepasst,
   nicht kalibriert. Dann die Default-Werte behalten.
3. Live-Ergebnisse liegen typischerweise unter dem Backtest. Alles, was im
   Backtest nur knapp positiv ist, ist live negativ.

## Tests

    python -m pytest tests/ -q

`tests/test_daily_trend.py` prüft Signal-Kausalität (kein Blick in die Zukunft),
Gebühren-Arithmetik, Mindestorder und Notstopp im Backtester.
`tests/test_bot_daily_trend_integration.py` fährt den Bot mit einem Fake-Feed
durch Kauf, Trendbruch und Notstopp.

## Was noch fehlt für Fusion live

- **Order-Engine für Fusion**: `core/live_order_engine.py` spricht One Trading
  über CCXT. Fusion hat keinen CCXT-Adapter; die REST-API (Market/Limit, Buy/Sell,
  `x-api-key`) müsste direkt angebunden werden, inklusive Reconciliation.
- **Fusion MCP** ist ein Aufsichts- und Eingriffskanal für ein KI-Modell im
  Gespräch, keine Ausführungsschicht für eine autonome Schleife. Der Bot spricht
  die REST-API; der MCP bleibt für Kontrolle und manuelles Eingreifen.
- **Shadow-Phase**: Mindestens einen vollen Trendzyklus (Einstieg und Ausstieg)
  im Paper-Modus mit `data_feed: fusion` beobachten, bevor echtes Geld fließt.
