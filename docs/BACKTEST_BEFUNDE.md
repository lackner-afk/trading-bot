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
| Mindestordergröße egal | Fusion: **25 €** je Order (BTC-EUR, gemessen über `/v1/pairs` → `minOrderAmount` am 24.08.2026; kann je Paar und über die Zeit abweichen — der Bot liest den Wert seit Okt. 2026 bei jedem Start neu) |
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

## Funding Rate (Schritt 1 des Datenquellen-Plans): negativ

Getestet mit `tools/funding_factor_test.py`. These: Hohe positive Funding Rates
zeigen überhitzte Long-Positionierung und gehen Rücksetzern voraus — dann müsste
ein Long-Einstieg bei hoher Rate schlechter laufen, also ein negativer IC.

Methodik: Funding wird alle 8 h fixiert, über 90 Tage sind das 270 Werte je
Symbol. Ein Test pro 5m-Kerze hätte 45.000 stark autokorrelierte Beobachtungen
und viel zu optimistische p-Werte ergeben — deshalb genau **eine Beobachtung je
Funding-Periode**.

| Fenster | IC | p | Beobachtungen |
|---|---|---|---|
| Mai–Aug 2026 | +0,025 | 0,52 | 670 |
| Feb–Mai 2026 | −0,042 | 0,27 | 696 |
| Nov 2025–Feb 2026 | +0,021 | 0,59 | 635 |

Nichts davon ist signifikant, die Vorzeichen wechseln, kein Quintil-Gradient.

**Einschränkung, fairerweise:** Die Raten lagen durchgehend zwischen −0,009 % und
+0,010 % pro 8 h, also um den Binance-Normalwert. Extreme Positionierung, bei der
dieser Faktor erst aussagekräftig wird, kam in den 270 Tagen nicht vor. Der Test
schliesst nicht aus, dass Funding in einer Euphorie- oder Panikphase etwas taugt —
für den Alltagsbetrieb liefert es nichts. Mit ~670 Beobachtungen liessen sich zudem
nur Effekte ab etwa |IC| 0,08 nachweisen; ein kleinerer echter Effekt wäre
untergegangen, würde aber ohnehin nicht reichen (nötig wären ~0,15).

## Abbruchkriterium erreicht (26.08.2026)

Der Plan legte vorab fest: Zeigt nach Schritt 1 und 2 kein Faktor über drei
Fenster einen stabilen IC ≥ 0,03, ist die Strategie-Idee auf diesem Zeitraster
erschöpft. Schritt 2 (Zeitrahmen 1h): negativ. Schritt 1 (Funding Rate): negativ.

**Konsequenz: keine weitere Optimierung an Gewichten, Schwellen, TP/SL oder
weiteren Faktoren aus Kerzendaten.** Offen bleibt allein Schritt 0 — Orderbuch-
und Orderflow-Daten selbst sammeln, weil sie historisch nicht beschaffbar sind
und als einzige Quelle noch ungeprüft ist.

## Regime-Erkennung (06.10.2026): misst nicht, was sie behauptet

Der Bot meldete praktisch immer `low_vol_chop`. Über 30 Tage BTC/ETH/SOL-EUR auf
5m: 91–94 % `low_vol_chop`, Rest `ranging`, **nie** `trending` oder
`high_vol_event`. Zwei Rechenfehler in `RegimeDetector`:

1. `short_vol = std(12 Renditen) · √12` ist eine Stunden-Vola (Median 0,3–0,5 %,
   Maximum 1,9–3,6 %). Die Schwellen 0,16 und 0,85 liegen 45–60× darüber — die
   Bedingung für „ruhig“ ist immer wahr, „hohe Volatilität“ unerreichbar.
2. `vol_ratio = (std12 · √12) / (std30 · √30)` enthält den Faktor √(12/30) = 0,63.
   Bei gleicher Schwankung ergibt sich 0,63 < 0,85 → „ruhig“.

Folgen: Gewichte stehen immer im chop-Profil, das Sentiment deutet Gier als
**Short**-Signal (auf Spot verworfen), `volatility_level < 0.2` ist immer wahr.

Korrekte Messung (`MarketConditions`: Schwankung relativ zum eigenen Fenster,
Kaufman-Effizienz, Schwellen aus 90-Tage-Perzentilen) ergibt ~70 % Spanne,
16–20 % ruhig, 7 % Trend, 4–6 % hohe Volatilität. **Im Handel getestet war sie
schlechter** — real: 0,25 %, nur Long, 12:5, 30 € Mindestorder, 100 € Kapital,
Schwelle 0,647, Cooldown 6 Kerzen:

| 90-Tage-Fenster bis | alte Erkennung | korrekte Erkennung | Buy & Hold |
|---|---|---|---|
| 20.02.2026 | −65,7 % | −80,6 % | −28,0 % |
| 20.05.2026 | −43,7 % | −71,2 % | +8,5 % |
| 20.08.2026 | −43,5 % | −72,7 % | +0,3 % |
| 06.10.2026 | +5,7 % | −47,0 % | +48,0 % |

Die Strategie ist auf das falsche Etikett eingestellt; die Profile für Trend und
hohe Volatilität waren toter Code und nie kalibriert. **Entscheidung:** Handel
weiter mit `RegimeDetector`, `MarketConditions` nur im Dashboard.

Zweiter Befund aus derselben Messreihe: Mit **100 €** Kapital und 30 €
Mindestorder (jede Position ~30 % des Kapitals) verliert die Live-Konfiguration
44–66 % in drei von vier Fenstern. Die milderen Zahlen oben (−7 bis −19 % bei
12:5) galten für 10.000 € Kapital. Trefferquote 28–35 %, nötig ~41 %.

> **Korrektur 12.10.2026:** Diese Messreihe lief mit dem Signal-Hebel (~6×),
> nicht mit dem Hebel 1 des Spot-Betriebs — siehe nächster Abschnitt. Die
> Beträge sind dadurch etwa doppelt so groß; die Rangfolge alt/neu bleibt.

## Shorts und Perpetuals (12.10.2026): die Gebühr zählt, nicht die Short-Seite

Frage: Würde der Bot mit Shorts (über Perpetual Futures) besser abschneiden?
Gemessen in denselben vier 90-Tage-Fenstern, 100 € Kapital, 30 € Mindestorder,
TP/SL 12:5, Floor `min_tp_pct` 0,8 %, Verlust-Sperre 6 Kerzen.

**Werkzeug-Fehler vorab:** `tools/backtest_confluence.py` nahm bisher immer den
Hebel aus dem Signal (`base_leverage: 6` → ~4–7×), auch mit `LONG_ONLY=1`. Live
handelt der Bot im Spot-Modus mit 1×. Alle Spot-Zahlen vor dem 12.10. sind
deshalb etwa doppelt so schlecht wie die Wirklichkeit (Zeile „Signal-Hebel“
unten). Neu: `LEVERAGE=1` erzwingt den Hebel.

Annahmen für Perps: 0,05 % Taker, Funding 0,01 % je 8 h (Longs zahlen, Shorts
bekommen) — Werte aus `docs/KRYPTO_BOTS_IM_VERGLEICH.md`, nicht an einer
konkreten Börse geprüft.

Rendite je Fenster bei der Live-Schwelle 0,647:

| Variante | bis 20.02. | bis 20.05. | bis 20.08. | bis 12.10. |
|---|---|---|---|---|
| Buy & Hold | −28,0 % | +8,5 % | +0,3 % | +31,2 % |
| **Spot nur Long, 0,25 %, 1× (live)** | −31,9 % | −24,7 % | −26,3 % | −13,9 % |
| Spot Long + Short, 0,25 %, 1× (gibt es nicht) | −24,4 % | −29,3 % | −27,8 % | −14,6 % |
| Perp nur Long, 0,05 %, 1× | −14,2 % | −3,1 % | −6,6 % | +8,0 % |
| Perp Long + Short, 0,05 %, 1× | −4,6 % | −1,9 % | −1,8 % | +2,5 % |
| Perp Long + Short, 0,05 %, 2× | −7,8 % | −4,1 % | −3,9 % | +2,4 % |
| Spot nur Long, Signal-Hebel (alte Rechnung) | −68,3 % | −43,9 % | −44,6 % | −12,1 % |

Ergebnis der Short-Trades allein (Perp, 1×): +7,7 / −2,4 / +2,6 / −5,3 € —
summiert **+2,6 € über 360 Tage**. Mit Spot-Gebühren −43,5 €.

- **Die Gebühr ist der größte Hebel:** 0,25 % → 0,05 % bringt 10–22
  Prozentpunkte je Fenster. Bei 0,25 % kostet ein Fenster 24–29 € Gebühren auf
  100 € Kapital.
- **Shorts glätten, bringen aber keinen Ertrag:** Gewinn im Bärenfenster, Verlust
  im Bullenfenster. Passt zur Literatur (Short-Seite in Krypto-Trendfolge schwach).
- **Hebel 2× ist schlechter als 1×.**
- Keine Variante schlägt Buy & Hold über alle Fenster (summiert −5,8 % gegen +12 %).

Nicht belastbar: Bei Schwelle 0,85 sah Long + Short besser aus (+13,7 / −7,7 /
+6,7 / −1,0 %). Das ist eine von 18 gerechneten Varianten, nachträglich
gewählt — kein Beleg. **Entscheidung:** kein Konto für Shorts eröffnen; der Bot
bleibt Spot-only.

## Tageskerzen-Trendfilter (12.10.2026): der erste Ansatz mit positivem Ergebnis

Test T1 aus `docs/KRYPTO_BOTS_IM_VERGLEICH.md`, Werkzeug `tools/backtest_daily.py`.
Sechs vorab festgelegte Varianten, alle berichtet: Halten (je Coin gleicher
Anteil), Trendfilter (halten, solange Schlusskurs > gleitender Durchschnitt 50 /
100 / 200 Tage), Momentum (halten, solange die Coin über 30 / 90 Tage gestiegen
ist). Entscheidung am Tagesschluss, Ausführung zum Eröffnungskurs des
Folgetags, 0,25 % je Order, Mindestorder 30 € mit Anhebung wie live.

Gesamtrendite (größter Rückgang in Klammern):

| Variante | BTC+ETH+SOL ab 02/2021, 100 € | BTC ab 03/2018, 100 € | BTC+ETH ab 03/2018, 1000 € |
|---|---|---|---|
| Halten | +299 % (92 %) | +629 % (77 %) | +414 % (81 %) |
| SMA 50 | +1147 % (56 %) | +1857 % (59 %) | +2424 % (58 %) |
| SMA 100 | +380 % (64 %) | +1797 % (39 %) | +1386 % (60 %) |
| SMA 200 | +1483 % (50 %) | +615 % (65 %) | +977 % (64 %) |
| Momentum 30 | +1307 % (43 %) | +748 % (66 %) | +1907 % (58 %) |
| Momentum 90 | +358 % (64 %) | +647 % (67 %) | +516 % (72 %) |

In den vier 90-Tage-Fenstern des Confluence-Backtests (BTC/ETH/SOL): SMA 100
−1 / −7 / +10 / +32 %, SMA 200 ±0 / ±0 / +4 / +21 % gegen Confluence live
−32 / −25 / −26 / −14 %.

- **Der Wert ist Crash-Schutz:** 2022 verlor Halten 65–76 %, die Trendfilter
  9–40 %; 2018 ähnlich.
- Risikobereinigt (Sharpe 0,8–1,2 gegen 0,6–0,7) fast überall besser als Halten;
  nur bei BTC allein liegen SMA 200 und Momentum etwa gleichauf.
- **Keine Variante ist überall die beste** — robust ist die Kombination mehrerer
  Rückblick-Längen, nicht die Auswahl der besten.
- 15–85 Trades pro Jahr statt ~190 in 90 Tagen; die Gebühr verliert ihr Gewicht.
- **100-€-Falle:** BTC+ETH mit 100 € friert nach 2018 ein — unter 60 € Kapital
  ist eine 30-€-Order mehr als die Hälfte, die meisten Varianten kaufen nie
  wieder (Endstand ~−42 %). Mit 1000 € tritt das nicht auf.
- Vorbehalte: USDT- statt EUR-Kurse, kein Slippage, keine Steuer; SOL ist im
  Nachhinein gewählt (Gegenprobe ohne SOL bestätigt das Bild).

### Mehrheitsregel und Live-Größe (14.10.2026)

Nachträglich ergänzt, darum ausdrücklich als solche markiert: **vote3** hält
eine Coin, solange mindestens 3 der 5 Regeln oben halten wollen. Gewählt, weil
keine Einzelregel überall die beste war — nicht, weil vote3 irgendwo gewann.
1000 € Start, 0,25 %:

| Korb | je Coin | Halten | vote3 | größter Rückgang Halten → vote3 |
|---|---|---|---|---|
| BTC+ETH+SOL ab 02/2021 | ⅓ | +286 % | +502 % | 92 → 59 % |
| BTC ab 03/2018 | alles | +624 % | +1311 % | 77 → 47 % |
| BTC+ETH ab 03/2018 | ½ | +409 % | +1752 % | 81 → 54 % |
| BTC+ETH+SOL ab 02/2021 | **20 %** | +172 % | +313 % | 85 → 47 % |
| BTC+ETH ab 03/2018 | **20 %** | +163 % | +439 % | 61 → 37 % |

vote3 ist nie die beste und nie die schlechteste Einzelregel, schlägt Halten
aber überall. 2022: −13 bis −37 % statt −27 bis −76 %.

**20 % je Coin** ist die Live-Größe: Der RiskManager begrenzt jede Position
hart auf 20 % des Kapitals, höchstens 60 % sind investiert. Nicht simuliert:
das Tageslimit (10 % Verlust an einem Tag → alles schließen) — bei 60 %
Investitionsgrad bräuchte es dafür einen Tagesrutsch aller Coins um ~17 %.

Live läuft vote3 als zweiter Paper-Bot (`strategies/daily_trend.py`,
Container `trading-trend`, Overlay `config/daily_trend.yaml`). Der Backtest
lädt die Regeln aus derselben Datei. Unterschiede zum Backtest: Kraken-EUR-
Tageskerzen statt Binance-USDT, Ausführung zum Fusion-Kurs beim nächsten
Abgleich (alle 15 min) statt exakt zum Tageseröffnungskurs.

## Donchian-Ausbruch für Panda-9 (28.10.2026)

Anlass: Panda-9 (Confluence) lag live vom 11. bis 28.10. bei 16 Trades, einem
Gewinner und −6,6 %, wie von der Faktor-Analyse vorhergesagt. Statt weiter am
Score zu drehen, bekommt der 100-€-Bot eine zweite Trendfamilie, damit die
beiden Paper-Bots Verschiedenes testen: **Donchian-Ausbruch** auf Tageskerzen,
Kauf über dem Hoch der vorigen N Tage, Verkauf unter dem Tief der vorigen M
Tage. Vor dem Lauf festgelegt: don20 = 20/10 (Turtle-System 1), don55 = 55/20
(System 2); dazu `MAX_POSITIONS` im Backtest (wollen mehr Coins rein, gewinnt
die Reihenfolge BTC → ETH → SOL).

100 € Start, 0,25 %, je Coin 20 % (wie live auf die 30-€-Mindestorder
angehoben), Gesamtrendite (größter Rückgang):

| Korb, höchstens | Halten | vote3 | don20 | don55 |
|---|---|---|---|---|
| BTC+ETH+SOL ab 02/2021, 1 | +23 % (30 %) | +157 % (36 %) | +98 % (22 %) | +55 % (31 %) |
| BTC+ETH+SOL ab 02/2021, **2** | +41 % (60 %) | +190 % (49 %) | **+107 % (28 %)** | +97 % (39 %) |
| BTC+ETH+SOL ab 02/2021, 3 | +241 % (91 %) | +377 % (48 %) | +221 % (38 %) | +292 % (38 %) |
| BTC+ETH ab 07/2018, 1 | +300 % (61 %) | +211 % (32 %) | +179 % (16 %) | +87 % (34 %) |
| BTC+ETH ab 07/2018, 2 | +427 % (72 %) | +594 % (37 %) | +335 % (22 %) | +318 % (39 %) |

- **Positiv in jeder Variante**, mit dem kleinsten Rückgang aller Regeln
  (don20 16–38 %). 2022: −9 bis −33 % statt bis −68 % beim Halten.
- **Nicht besser als vote3:** Die Mehrheitsregel bringt fast überall mehr
  Rendite; don20 ist die vorsichtigere Variante, nicht die stärkere. In den
  letzten 90-Tage-Fenstern liegt don20 sogar meist hinter vote3
  (z. B. −8,5 / −9,1 / +2,5 / +4,9 % gegen −0,9 / −5,0 / +9,6 / +15,9 %).
- 2025 und 2026 bisher leicht negativ (−5 bis −10 %) — Seitwärtsmärkte sind
  für Ausbruchsregeln das schlechteste Umfeld.
- Mit 1 Position steckt bei 100 € nur die 30-€-Mindestorder im Markt; die
  Equity bewegt sich dann kaum. Darum live **2 Positionen**
  (`max_concurrent_positions` bleibt 2).
- Gewählt wurde don20 (bessere Sharpe und kleinerer Rückgang als don55 in
  allen fünf Läufen) — nachträglich zwischen zwei vorab festgelegten Varianten,
  also kein Beleg, dass 20/10 die richtige Länge ist.

Live: `strategies/daily_trend.py` mit `rule: donchian` (settings.yaml), selber
Abgleich-Loop wie Trend-1, dieselbe Funktion `donchian_hold` im Backtest. Weil
der Zustand vom Verlauf abhängt, kauft der Bot nach einem Neustart auch einen
Ausbruch, der schon Tage zurückliegt — im Backtest wäre er dort schon drin.
Neustart am 28.10. mit 100 €, alte Datenbank gesichert.

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
    MIN_ORDER_AMOUNT=30   # Mindestordergröße der Börse (Fusion: 30 €); wie live
                          # angehoben, solange ≤ 50 % des Kapitals
    THRESHOLDS=0.647      # feste Schwellen zusätzlich zu den Perzentilen
    MIN_TP_PCT=0.010      # TP-Floor über den Round-Trip-Kosten
    START_CAPITAL=100     # reales Kapital statt 10.000
    GRID_CACHE=1 GRID_CACHE_PATH=/tmp/grid_w1.pkl   # Pass 1 überspringen
    LEVERAGE=1            # fester Hebel (Spot live: 1); ohne Angabe Signal-Hebel ~6x
    FUNDING_8H=0.0001     # Perp-Funding je 8 h; Longs zahlen, Shorts bekommen

Je Lauf stehen im JSON-Ergebnis zusätzlich `long_pnl`, `short_pnl`,
`short_wins` und `fees` (Summe der Gebühren).
