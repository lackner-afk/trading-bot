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

## Faktor-Analyse (25.08.2026): es fehlt das Signal, nicht die Kalibrierung

Gemessen mit `tools/factor_analysis.py`: Für jedes Signal wird der reine Ausgang
bestimmt (läuft der Kurs zuerst ins TP oder ins SL, ohne Portfolio-Effekte), dann
je Faktor die Rangkorrelation zwischen Score und Ausgang — der Information
Coefficient. Über dieselben drei 90-Tage-Fenster, je ~46.000 auswertbare Signale.

### Einzelfaktoren: keiner ist über Marktphasen stabil

| Faktor | Mai–Aug | Feb–Mai | Nov–Feb | stabil? |
|---|---|---|---|---|
| multi_timeframe_trend | −0,053 | −0,004 | −0,008 | nein |
| momentum | −0,023 | +0,003 | −0,001 | nein |
| sentiment | +0,022 | +0,032 | **−0,063** | Vorzeichen dreht |
| mean_reversion | +0,021 | +0,010 | +0,008 | ja, aber ≈ 0 |
| volatility_filter | −0,012 | −0,052 | −0,042 | nein |
| volume_confirmation | −0,002 | −0,004 | +0,022 | nein |
| macro_news_filter | konstant | konstant | konstant | reiner Ballast |

Wechselnde Vorzeichen zwischen Marktphasen heißt: kein Signal, sondern
angepasstes Rauschen.

### Der Gesamtscore hat ebenfalls keine Vorhersagekraft

| Fenster | IC | Trefferquote Q1→Q5 |
|---|---|---|
| Mai–Aug 2026 | −0,009 | 27 % 26 % 26 % 25 % 26 % |
| Feb–Mai 2026 | +0,016 | 29 % 25 % 29 % 30 % 29 % |
| Nov 2025–Feb 2026 | **−0,055** | 29 % 29 % 24 % 24 % 23 % |

In zwei Fenstern ist die Trefferquote über alle Quintile praktisch identisch, im
dritten fällt sie monoton. **`min_confluence_score` filtert damit nicht nach
Qualität, sondern nur nach Menge** — die gesamte Kalibrierhistorie an dieser
Schwelle (0,66 → 0,647 → Perzentil-Sweeps) optimierte einen Parameter, der nichts
sortiert.

### Größenordnung

Trefferquote über alle Signale: 30,6 % / 28,6 % / 25,7 %. Nötig bei TP 12× / SL 5×
sind ~39 %. Für die fehlenden ~10 Punkte bräuchte es einen IC um 0,15; vorhanden
sind 0,01–0,06 mit wechselndem Vorzeichen. Das ist keine Lücke, die Gewichtung,
Schwellen oder das Aussortieren einzelner Faktoren schließen können.

**Konsequenz: Nicht weiter an Gewichten, Schwellen oder TP/SL drehen.** Die sieben
Faktoren sind OHLCV-Ableitungen auf 5m-Kerzen plus ein täglicher Stimmungsindex —
auf diesem Zeitraster ist kein verwertbarer Vorsprung in den Daten.

### Methodischer Merkposten

Ein erster Lauf über nur 10 Tage zeigte scheinbar starke Werte (sentiment
IC −0,375, volatility_filter +0,260). Beides war ein Artefakt: Langsam variierende
Faktoren — Fear & Greed liefert einen Wert pro Tag — haben über kurze Strecken zu
wenige Ausprägungen, die Quintile trennen dann nach Kalendertagen statt nach
Signalstärke. Erkennbar am nicht-monotonen Verlauf (Einbruch im obersten
Quintil). Faktor-Analysen deshalb nie unter 90 Tagen.

## Zeitrahmen 1h statt 5m (Schritt 2 des Datenquellen-Plans): negativ

Dieselben Faktoren, dieselben drei Fenster, 1h-Kerzen, Horizont 120 Bars = 5 Tage:

| Fenster | Trefferquote 5m | Trefferquote 1h |
|---|---|---|
| Mai–Aug 2026 | 30,6 % | **19,3 %** |
| Feb–Mai 2026 | 28,6 % | **20,3 %** |
| Nov 2025–Feb 2026 | 25,7 % | **33,8 %** |

In zwei von drei Fenstern deutlich schlechter. Die ICs sind betragsmäßig größer
(bis ±0,14), wechseln aber weiter die Vorzeichen, und die Quintil-Verläufe werden
erratisch — bei ~2.400 statt 46.000 Signalen ist das grösstenteils Rauschen.
**Der Zeitrahmen ist nicht die Ursache.**

## Null-Modell: der Vorsprung gegenüber Zufall

`tools/null_model.py` vergleicht dieselben Kerzen, TP/SL-Regeln, Horizonte und
die ATR-Verteilung — nur Einstiegszeitpunkt und Richtung werden gewürfelt.

| Fenster | Confluence | Zufall | Differenz | |
|---|---|---|---|---|
| Mai–Aug 2026 | 26,1 % | 25,6 % | +0,6 % (1,9 σ) | nicht unterscheidbar |
| Feb–Mai 2026 | 28,6 % | 26,9 % | +1,7 % (5,5 σ) | besser als Zufall |
| Nov 2025–Feb 2026 | 25,7 % | 26,8 % | −1,0 % (−3,4 σ) | schlechter als Zufall |

Im Mittel **+0,4 Prozentpunkte** gegenüber Münzwurf-Einstiegen. Der Vorsprung ist
im mittleren Fenster mit 5,5 Standardfehlern real, aber nicht stabil — im dritten
kehrt er sich signifikant um. Und er ist um eine Größenordnung zu klein: bis zum
Break-even fehlen ~11 Punkte, geliefert wird im besten Fall 1,7.

Auf beiden Zeitrastern landen die Trefferquoten also fast genau dort, wo ein
Zufallsprozess mit diesem Chance-Risiko-Verhältnis landet. Merke: Eine
Trefferquote ist ohne diesen Vergleich nicht interpretierbar — 30 % klingt
schlecht, 60 % gut; was zählt, ist der Abstand zum Zufall bei gleichem TP/SL.

## Nächste Schritte, falls weiterverfolgt

1. **Andere Datenquellen.** Der Bot sieht nur Kerzen. Orderbuchtiefe und
   -ungleichgewicht, Funding Rates, Open Interest, Liquidationen bleiben
   ungenutzt — daher kommen kurzfristige Krypto-Signale üblicherweise. Der
   Fusion-Feed liefert bereits Orderbuchdaten.
2. **Größerer Zeitrahmen.** 5m ist für Kerzen-Indikatoren stark verrauscht; auf
   1h/1d ist mehr Struktur vorhanden, und die Gebühr fällt seltener an.
3. **Jeden neuen Faktor zuerst durch `tools/factor_analysis.py` schicken.** Ein
   IC unter 0,03 oder ein Vorzeichenwechsel zwischen Fenstern heißt: nicht
   einbauen. Das kostet Minuten statt Wochen Papierbetrieb.
4. Erst wenn ein Profil über alle Fenster positiv ist, über Kapitaleinsatz reden.

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
