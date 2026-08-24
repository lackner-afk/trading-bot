# Backtest-Befunde — 24.08.2026

Vermessung der Confluence-Strategie unter realistischen Bedingungen, nachdem die
bisherige Kalibrierung (68 % Trefferquote, 30-Tage-Fenster) mit zu günstigen
Annahmen gerechnet hatte.

## Kurzfassung

**Die Strategie ist unter realen Gebühren über drei unabhängige Zeiträume
verlustbringend.** Kein echtes Kapital auf dieser Basis.

Die Umkehr des Risiko/Ertrag-Verhältnisses (TP größer als SL) verbessert das
Ergebnis erheblich — von summiert −55,5 % auf −19,1 % — reicht aber nicht bis in
den positiven Bereich. Das verbleibende Problem ist die Signalqualität, nicht
mehr das Money Management.

## Was die Annahmen verzerrt hatte

| Annahme bisher | Realität |
|---|---|
| Taker-Gebühr 0,06 % | Bitpanda Fusion Stufe 1: **0,25 %** (Round-Trip 0,50 %) |
| Shorts möglich | Fusion ist Spot: *„Short selling is not supported by this API"* |
| Hebel 6× | Kein Leverage-Feld in der Fusion-Order-API |
| Startkapital 10.000 € im Harness | Reales Konto: 100 € |
| Mindestordergröße egal | Fusion: **25 €** je Order (BTC-EUR) |
| 200-EMA-Trendfilter | Live nur 99er-EMA — `.tail(100)` kappte die Historie |

## Der Kern: das Risiko/Ertrag-Verhältnis

Die Konfiguration fuhr `tp_atr_multiplier: 5` und `sl_atr_multiplier: 7` — der
Stop lag **weiter weg als das Ziel**. Das treibt die Trefferquote nach oben, macht
aber jeden Verlust 1,4-mal so schwer wie jeden Gewinn.

Bei typischen 0,3 % ATR (Gewinn 1,5 %, Verlust 2,1 %) und 0,5 % Round-Trip-Kosten:

    nötige Trefferquote = (2,1 + 0,5) / (1,5 + 2,1) = 72 %

Real erreicht wurden 55–65 %. Das erklärt, warum Konfigurationen mit 64 %
Trefferquote trotzdem −27,9 % lieferten.

## Messreihe: TP/SL-Profile über drei Marktphasen

Jeweils bester Return aus 48 Kombinationen, Taker 0,25 %, 90-Tage-Fenster:

| Profil | Mai–Aug 2026 | Feb–Mai 2026 | Nov 2025–Feb 2026 | Summe |
|---|---|---|---|---|
| TP 5 / SL 7 (bisherige Config) | +2,8 % | −27,9 % | −30,4 % | **−55,5 %** |
| **TP 12 / SL 5** (robustester) | +7,4 % | −19,3 % | −7,2 % | **−19,1 %** |
| TP 16 / SL 6 | +14,1 % | −25,2 % | −14,6 % | −25,7 % |
| Buy & Hold (Ø 3 Coins) | +12,2 % | +8,5 % | −28,0 % | — |

`16:6` gewinnt im ersten Fenster, verliert aber in den anderen stärker. Es
auszuwählen wäre derselbe Fehler wie die ursprüngliche 30-Tage-Kalibrierung.

Bei 12:5 liegt die nötige Trefferquote bei ~41 %. Erreicht: 40,2 % (Fenster 1,
knapp positiv), je ~35 % (Fenster 2 und 3, negativ). Die Lücke ist der Verlust.

## Overfitting-Nachweis

Die optimale Schwelle wandert je Zeitraum erheblich:

| Fenster | optimale Schwelle |
|---|---|
| 30 Tage (ursprüngliche Kalibrierung) | 0,646 |
| Mai–Aug 2026 | 0,827 |
| Feb–Mai 2026 | 0,857 |
| Nov 2025–Feb 2026 | 0,868 |

Die Config fährt 0,647. Ein Parameter, der je nach Fenster um 30 % springt, ist an
ein Fenster angepasst, nicht kalibriert.

## Wo der Bot tatsächlich Wert hat

Im Abwärtsfenster (Nov 2025–Feb 2026) hielt er mit 12:5 den Verlust bei −7,2 %,
während Halten −28 % kostete: gut **20 Punkte Vorsprung**. Als defensives
Instrument funktioniert er, als Renditequelle bislang nicht.

## Der Trend-Faktor war tot

`multi_timeframe_trend` rechnete `score = min(spread * 8, 1.0)`. Für Score 1,0
hätte es 12,5 % EMA-Spread gebraucht; real kommen auf 5m-Kerzen höchstens 0,83 %
vor (gemessen über 1971 Kerzen BTC/ETH/SOL-EUR). Der Faktor erreichte nie mehr als
**0,066** und senkte als toter Ballast den Mittelwert aller Technik-Faktoren.

Jetzt ATR-normalisiert (`spread / atr_pct`), Median 0,483 statt 0,008. Die
Backtest-Performance verbessert das allerdings **nicht** — der Faktor ist nur
nicht mehr kaputt.

## Nächste Schritte, falls weiterverfolgt

1. Die Confluence-Faktoren selbst prüfen — bei ~35 % Trefferquote gegen ~41 %
   nötige liegt das Problem in der Signalqualität.
2. Handelsfrequenz senken (100–500 Trades je Fenster; die Gebühr skaliert mit der
   Anzahl). Größere Zeitrahmen als 5m prüfen.
3. Erst wenn ein Profil über alle Fenster positiv ist, über Kapitaleinsatz reden.

## Werkzeug

`tools/backtest_confluence.py` ist jetzt über die Umgebung steuerbar:

    TAKER_FEE=0.0025      # echte Börsengebühr
    DAYS=90               # Fensterlänge
    BACKTEST_END=2026-05-20   # Fensterende (leer = bis jetzt)
    TPSL="12:5,16:6"      # TP/SL als ATR-Vielfache
    LONG_ONLY=1           # Spot-Börse ohne Shorts
    MIN_ORDER_AMOUNT=25   # Mindestordergröße der Börse
    MIN_TP_PCT=0.010      # TP-Floor über den Round-Trip-Kosten
    START_CAPITAL=100     # reales Kapital statt 10.000
    GRID_CACHE=1 GRID_CACHE_PATH=/tmp/grid_w1.pkl   # Pass 1 überspringen
