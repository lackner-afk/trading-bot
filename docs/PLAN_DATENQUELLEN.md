# Plan: neue Datenquellen (Stand 25.08.2026)

Ausgangslage: Die sieben Confluence-Faktoren haben über drei 90-Tage-Fenster
keine stabile Vorhersagekraft (Details in `BACKTEST_BEFUNDE.md`). Auch der
Gesamtscore sagt den Ausgang nicht voraus. An Gewichten, Schwellen oder TP/SL
weiterzudrehen bringt nichts — es fehlt Signal, nicht Kalibrierung.

Alle Faktoren stammen aus OHLCV-Ableitungen auf 5m-Kerzen plus einem täglichen
Stimmungsindex. Dieser Plan prüft, ob andere Daten oder ein anderer Zeitrahmen
etwas beitragen.

## Was sich historisch überhaupt beschaffen lässt

Am 25.08.2026 direkt gegen die Binance-API geprüft:

| Quelle | Historie | Auflösung | über 3×90 Tage validierbar? |
|---|---|---|---|
| **Funding Rate** | **90 Tage** | 8 h | **ja** |
| Open Interest | 30 Tage (hart) | 5 m | nein — nur ein Fenster |
| Taker Buy/Sell-Volumen | 30 Tage (hart) | 5 m | nein |
| Long/Short-Ratio | 30 Tage (hart) | 5 m | nein |
| Orderbuchtiefe | **keine** | nur live | nein |

Ab 35 Tagen antwortet `/futures/data/*` mit HTTP 400. Orderbuchdaten gibt es
historisch gar nicht — die lassen sich ausschließlich vorwärts sammeln.

## Validierungsregel (gilt für jeden neuen Faktor)

Erst messen, dann einbauen. Ein Faktor kommt nur in die Strategie, wenn
`tools/factor_analysis.py` über **drei** 90-Tage-Fenster einen IC von **≥ 0,03
mit gleichem Vorzeichen** zeigt. Nie unter 90 Tagen messen — ein 10-Tage-Lauf
zeigte für `sentiment` einen IC von −0,375, der sich über 90 Tage vollständig
auflöste.

## Schritte

### Schritt 0 — Datensammler starten (zuerst, weil er Zeit braucht)
Recorder, der alle 5 Minuten Orderbuch-Snapshot (Tiefe, Bid/Ask-Ungleichgewicht),
Open Interest, Taker-Ratio und Long/Short-Ratio für BTC/ETH/SOL in eine eigene
SQLite schreibt. Nach 30 Tagen existiert eine Historie, die Binance nicht mehr
hergibt; nach 90 Tagen eine, die eine echte Validierung erlaubt. Läuft im
LaunchAgent mit.
*Aufwand ~2 h, Ertrag erst in Wochen — deshalb sofort starten.*

### Schritt 1 — Funding-Rate-Faktor
Einzige neue Quelle mit 90 Tagen Historie, also sofort über dieselben drei
Fenster messbar. These: Extreme Funding Rates zeigen überhitzte Positionierung —
hohe positive Raten heißen, Longs zahlen für ihre Position, was Rücksetzern
vorausgeht. Inhaltlich etwas anderes als Kerzen-Indikatoren.
*Aufwand ~3 h inklusive Messung.*

### Schritt 2 — Zeitrahmen prüfen  ← **hier fangen wir an**
Dieselben sieben Faktoren auf 1h-Kerzen statt 5m. Daten liegen vor, es ist im
Kern ein Parameter. 5 Minuten sind für Kerzen-Indikatoren stark verrauscht; auf
1h ist mehr Struktur vorhanden, und die Gebühr fällt seltener an — bei 0,5 % pro
Runde erheblich.
*Aufwand ~1 h. Bestes Verhältnis von Aufwand zu Erkenntnis, und es beantwortet
die Frage, die allen anderen Schritten vorausgeht: liegt es an den Daten oder am
Zeitraster?*

### Schritt 3 — Orderflow über 30 Tage als Vorab-Indikation
Taker-Ratio, OI-Veränderung, Long/Short-Ratio über das einzige verfügbare
Fenster. Ergebnis ausdrücklich als Indikation, nicht als Beweis — bei einem
Fenster droht genau der Artefakt-Effekt aus dem Merkposten oben. Zeigt sich
etwas, wird es mit den Daten aus Schritt 0 nachvalidiert.
*Aufwand ~3 h.*

## Abbruchkriterium

Zeigt nach Schritt 1 und 2 **kein** Faktor über drei Fenster einen stabilen
IC ≥ 0,03, ist die Strategie-Idee auf diesem Zeitraster erschöpft. Dann bleibt
der Bot ein Lernprojekt mit solider Infrastruktur, und wir hören auf, Rendite zu
erwarten.

Diese Grenze vorher festzulegen ist der Punkt — sonst findet man in jedem neuen
Datensatz irgendein Muster.
