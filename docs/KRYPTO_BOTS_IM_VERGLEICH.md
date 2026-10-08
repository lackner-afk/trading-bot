# Weniger handeln, langsamer entscheiden, Gebühren schlagen

*Forschungsbericht, Stand 6. Oktober 2026. Grundlage: vier Recherche-Notizen in `docs/recherche/krypto-bots-im-vergleich/`.*

**Kurz gesagt:** Unser Bot verliert nicht, weil die Idee „Trend plus Filter" falsch wäre. Er verliert vor allem an zwei Dingen: an den Gebühren und an zu vielen Stellschrauben. Strategien mit echten Belegen entscheiden auf Tages-Kerzen oder höchstens auf 6-Stunden-Kerzen. Sie nutzen wenige, robuste Regeln und handeln selten. Unser Bot entscheidet alle 45 Sekunden auf 5-Minuten-Kerzen. Er mischt acht Signale mit fein eingestellten Gewichten. Und er zahlt 0,50 % pro Hin- und Rückweg. Das ist ungefähr das Drei- bis Vierfache einer normalen 5-Minuten-Bewegung von Bitcoin (eigene Rechnung). Die gute Nachricht: Der Baustein mit den besten Belegen ist schon eingebaut, nämlich der 200-EMA-Trendfilter. Auch das Bärenmarkt-Ergebnis passt genau zu dem, was Trendfolge laut Forschung leistet: −7 % statt −28 % bei reinem Halten. Die Assets sind liquide, aber fast gleichläufig. 100 € Kapital reichen zum Lernen, nicht zum Geldverdienen. Margin-Handel ändert die Rechnung stark. Für Privatkunden in der EU sind bei Krypto-Derivaten wahrscheinlich nur 2:1 erlaubt. Bitpanda-Margin kostet rund 0,18 % Finanzierung pro Tag. Und die Belege dafür, dass Shorts Rendite bringen, sind schwach. Die wichtigste Änderung ist deshalb: Signale auf Tagesbasis treffen, die Formel vereinfachen und die Zahl der Trades stark senken.

---

## Wie belastbar die Quellen sind: drei von vier Zahlen stammen aus Zusammenfassungen

Eine wichtige Einschränkung vorweg. In der Recherche-Umgebung waren fast alle Volltexte gesperrt. Das betraf arXiv, SSRN, BIS, ESMA, Bitpanda, Kraken, das österreichische Finanzministerium und weitere. **Die meisten Zahlen stammen daher aus Suchergebnis-Zusammenfassungen und Abstracts, nicht aus den Originaltexten.** Bevor eine Zahl in Code, Konfiguration oder eine Entscheidung mit echtem Geld einfließt, sollte man sie auf der Originalseite prüfen. Das gilt besonders für Gebühren und Regulierung.

Jede Aussage im Bericht trägt deshalb eine Beweisstufe:

| Kennzeichnung | Bedeutung |
|---|---|
| **[Live/geprüft]** | Echtes Geld, von Dritten oder on-chain dokumentiert (Index, Fonds, Protokoll, Behörde) |
| **[Wiss. Backtest]** | Wissenschaftliche Studie mit historischer Simulation, teils mit Korrektur für Mehrfachtests. Bleibt trotzdem ein Backtest |
| **[Einzel-Backtest]** | Einzelne Studie oder Vorabveröffentlichung, nicht unabhängig nachgeprüft |
| **[Marketing/unbelegt]** | Anbieter, Blogs, Ranglisten ohne überprüfbare Methode |
| **[Eigene Rechnung]** | Rechnung aus den Recherche-Notizen, ohne externe Quelle |

Ein *Backtest* ist eine Simulation auf vergangenen Kursen. Er zeigt, was passiert *wäre*. Er zeigt nicht, was passieren *wird*.

---

## Nur zwei Strategie-Familien haben echte Belege, und beide handeln langsam

### Was wirklich funktioniert hat

Die stärksten Belege gibt es für **langsame Trendfolge auf Tages-Kerzen**. Trendfolge heißt: kaufen, wenn der Kurs über einem gleitenden Durchschnitt oder einem alten Hoch liegt, und aussteigen, wenn er darunter fällt. Zarattini, Pagani und Barbon kombinieren mehrere Donchian-Kanäle mit verschiedenen Rückblick-Längen. Ein Donchian-Kanal ist das höchste Hoch und das tiefste Tief der letzten N Tage. Die Position wird nach Schwankung skaliert. Damit erreichen sie auf den 20 liquidesten Coins **eine Sharpe-Ratio über 1,5 und 10,8 % Zusatzrendite pro Jahr gegenüber Bitcoin, nach Gebühren** [Wiss. Backtest] ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)). Die *Sharpe-Ratio* misst Rendite pro Einheit Schwankung. Über 1 gilt als gut, über 2 ist selten und verdächtig. Wichtig an dieser Studie: Sie nutzt alle Coins seit 2015, auch die später verschwundenen. Sie leidet also nicht unter Überlebens-Verzerrung.

Detzel und Kollegen zeigen im *Financial Management*: Das Verhältnis von Kurs zu gleitendem Durchschnitt sagt **tägliche** Bitcoin-Renditen voraus, auch außerhalb der Stichprobe [Wiss. Backtest] ([WUSTL](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)). Liu und Tsyvinski finden starkes Momentum bei Krypto, also eine Tendenz, dass ein Trend weiterläuft. Das gilt auf **Tages- bis Wochenebene** [Wiss. Backtest] ([RFS](https://doi.org/10.1093/rfs/hhaa113)). Die schnellste glaubwürdige Variante ist eine Vorabveröffentlichung von 2026 mit **6-Stunden-Kerzen**. Sie meldet eine Sharpe-Ratio von 2,41 bei maximal −12,7 % Rückgang [Einzel-Backtest] ([arXiv](https://arxiv.org/html/2602.11708v1)). Dieser Wert ist ungewöhnlich hoch und nicht nachgeprüft.

Die zweite belegte Familie ist **Carry-Handel**: Man kauft Spot und verkauft gleichzeitig einen Terminkontrakt. So kassiert man die Prämie, die gehebelte Käufer zahlen. Die BIS fand Carry-Renditen, die **zeitweise über 40 % pro Jahr** lagen [Wiss. Backtest] ([BIS](https://www.bis.org/publications/working-paper-1087-crypto-carry)). Das Ethena-Protokoll betreibt so eine Strategie mit echtem Geld. Laut Sekundärquellen lag die Rendite **2024 bei rund 18 %, 2025 nur noch bei 4–15 %** [Live, aber nicht geprüft] ([eco.com](https://eco.com/support/en/articles/15254002-ethena-usde-and-susde-2026-delta-neutral-yield)). Der Vorteil schrumpft, weil immer mehr Kapital ihn ausnutzt. Für uns ist das ohnehin nicht machbar. Carry braucht eine Short-Seite, und Bitpanda Fusion bietet nur Spot.

### Was nur behauptet wird

Für **Grid-Bots** gibt es nur Anbieterzahlen. Pionex-nahe Quellen versprechen „15–50 % pro Jahr" und „8–12 % pro Monat in Seitwärtsphasen" [Marketing/unbelegt] ([Pionex-Blog](https://www.pionex.com/blog/15-reddit-questions-about-crypto-trading-bots-and-pionex-answered-with-real-data/)). Pionex vergleicht seinen eigenen Backtester sogar mit „Goldman Sachs" [Marketing/unbelegt] ([Pionex](https://www.pionex.com/blog/botai2_en/)). Freqtrade-„Fallstudien" mit „2509 % Gewinn" sind reine Backtests auf Medium [Marketing/unbelegt] ([Medium](https://imbuedeskpicasso.medium.com/2509-profit-unlocked-a-case-study-on-algorithmic-trading-with-freqtrade-39b1051c0f1e)). Freqtrade selbst warnt in der Dokumentation: **Ein Backtest ersetzt nie den Probelauf**, und auch gute Probeläufe garantieren keinen Live-Gewinn ([Freqtrade-Doku](https://www.freqtrade.io/en/stable/backtesting/)). Für DCA (regelmäßiges Kaufen) und privates Market-Making fand die Recherche gar keine unabhängigen Zahlen.

Copy-Trading zeigt die Lücke zwischen Schaufenster und Realität. Eine kommerzielle Auswertung von über 100.000 Fällen meldet: **97 % der Vorbild-Trader waren auf dem eigenen Konto im Plus, aber nur 44 % brachten ihren Nachahmern Gewinn** [Marketing/unbelegt, Methode nicht geprüft] ([YieldFund](https://yieldfund.com/is-copy-trading-profitable-a-90-day-multi-exchange-study)).

### Wie viele private Trader verdienen wirklich Geld?

Es gibt keine saubere Studie speziell zu Krypto-Bots. Die beste harte Zahl kommt aus einem Nachbarmarkt. In Brasilien wurden alle Daytrader auf Index-Futures ausgewertet. **97 % derer, die länger als 300 Tage durchhielten, verloren Geld**, und es gab keinen Lerneffekt [Live/geprüft, Vollerhebung, Daten 2013–2015] ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101)). Die oft zitierte Zahl „27 % der Krypto-Bots sind nach sechs Monaten erfolgreich" hat keine nachvollziehbare Methode [Marketing/unbelegt] ([lenz.io](https://lenz.io/q/percentage-of-crypto-trading-bots-successful)). Eine vernünftige Grundannahme lautet daher: Die klare Mehrheit privater Algo-Trader verliert nach Gebühren. Am stärksten trifft es die, die viel handeln.

### Selbst echte Trendfolger haben schlechte Jahre

Auch bewährte Profis leiden in bestimmten Marktphasen. Der SG Trend Index bildet große Trendfolge-Fonds mit echtem Geld ab. Er lag **im April 2025 bei −9,3 % seit Jahresbeginn und schloss 2025 mit +2,4 %**. Seit Start erzielt er rund 5,3 % pro Jahr bei einem maximalen Drawdown von 20,6 % [Live/geprüft] ([Top Traders Unplugged](https://www.toptradersunplugged.com/trend-following-performance-report-december-2025/)). *Drawdown* ist der Rückgang vom letzten Höchststand. Die Lehre daraus: Trendfolge schützt vor allem in Abstürzen. In Seitwärtsphasen kostet sie Geld. **Genau dieses Muster zeigt unser Bot: −7 % statt −28 % im Bärenmarkt, Verluste in den anderen Zeitfenstern.** Das spricht für die Grundidee. Es spricht aber nicht für die jetzige Umsetzung.

---

## Die Confluence-Formel hat zu viele Stellschrauben, um ihr zu trauen

### Mehr Signale bedeuten mehr Gelegenheiten, sich selbst zu täuschen

*Confluence* heißt: Viele Signale werden zu einer Punktzahl zusammengefasst. Das klingt vernünftig. Aber jede Gewichtung und jeder Schwellenwert ist eine weitere Stellschraube. Die Forschung zum *Overfitting* zeigt das Problem. Overfitting heißt Überanpassung: Eine Strategie lernt das Rauschen der Vergangenheit statt eines echten Musters. Bailey, Borwein, López de Prado und Zhu zeigen: **Je mehr Varianten man testet, desto wahrscheinlicher ist die beste Variante nur Zufall** [Wiss. Methode] ([SSRN](https://papers.ssrn.com/abstract=2326253)). Sie bieten dafür einen Test an, die „Wahrscheinlichkeit von Backtest-Overfitting" (PBO). Dazu gibt es eine fertige Software ([CRAN pbo](https://packages.oit.ncsu.edu/cran/web/packages/pbo/readme/README.html)). Eine verwandte Arbeit, die „Deflated Sharpe Ratio", zeigt: **Die beste von N Varianten hat eine hohe Sharpe-Ratio, selbst wenn keine echte Stärke vorhanden ist** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).

Unser Score enthält mindestens drei gewählte Gewichte (0,55 / 0,28 / 0,17) und einen Schwellenwert (0,647). Dazu kommen die inneren Parameter von sechs technischen Teilsignalen, die Fear-&-Greed-Umrechnung, der Makro-Filter und der Regime-Detektor. Mit allen Parameter-Durchläufen sind das vermutlich Hunderte bis Tausende getestete Varianten [Eigene Rechnung]. Zwei Warnzeichen fallen auf. Erstens ist der Schwellenwert auf drei Nachkommastellen genau. Zweitens **wanderte der beste Schwellenwert je nach Zeitfenster von 0,65 auf 0,87**. Ein echter Vorteil sollte über Zeitfenster stabil sein. Ein wandernder Optimalwert ist ein typisches Zeichen für Überanpassung.

Kombinationen *können* helfen, aber nur in einer bestimmten Form. Neely und Kollegen fassten 14 technische Indikatoren zusammen und verbesserten damit Monatsprognosen für Aktien. Sie nutzten dafür eine Methode fast ohne frei gewählte Gewichte [Wiss. Backtest] ([Management Science](https://doi.org/10.1287/mnsc.2013.1838)). Zarattini und Kollegen mitteln **eine einzige Signalfamilie über viele Rückblick-Längen**. Sie stellen dabei keine Gewichte pro Baustein ein ([RePEc](https://ideas.repec.org/p/chf/rpseri/rp2580.html)). Unser Bot macht das Gegenteil. Er mischt verschiedenartige Signale mit fein eingestellten Gewichten. Für so einen gemischten Score aus Trend, Rückkehr zum Mittelwert, Stimmung und Makro auf Krypto-Intraday-Daten fand die Recherche **keine einzige wissenschaftliche Bestätigung**.

### Die Bausteine einzeln betrachtet

| Baustein | Was die Forschung zeigt | Gilt für welchen Zeitrahmen? | Urteil für unseren Bot |
|---|---|---|---|
| **200-EMA-Richtungsfilter / MTF-Trend** | Kurs-zu-Durchschnitt sagt BTC-Renditen voraus ([WUSTL](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)) [Wiss. Backtest] | Tag | **Behalten**, aber auf Tages-Daten |
| **Momentum** | Starkes Zeitreihen-Momentum ([RFS](https://doi.org/10.1093/rfs/hhaa113)) [Wiss. Backtest] | Tag bis Woche | Sinnvoll, aber langsam |
| **Breakout** | Nach Kosten schlagen Ausbruchsregeln Kaufen-und-Halten meist nicht. Die Profitabilität sinkt seit 2017 (Zuordnung unsicher) ([Bakker](https://thesis.eur.nl/pub/41546/Bakker.pdf)) | Intraday und Tag | Auf 5 Minuten schwach |
| **Volumen-Bestätigung** | Volumen wirkt als Zeitgeber, nicht als Filter pro Signal ([Birmingham](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/)) | Halbe Stunde | Kein Beleg für „Volumen > X" |
| **Connors RSI-2 (Mittelwert-Rückkehr)** | Bei BTC schlug Trend die Mittelwert-Rückkehr sogar auf Tagesbasis ([Quantpedia](https://quantpedia.com/trend-following-and-mean-reversion-in-bitcoin/)) [Praxis-Backtest] | Tag | **Streichen** oder trennen: arbeitet gegen das Trendsignal |
| **Fear & Greed Index** | Sagt Renditen über **1 Woche bis 1 Monat** voraus. Ob gegenläufig oder mitlaufend, ist unklar ([FSV Prag](https://journal.fsv.cuni.cz/storage/1548_attachment.pdf)) | Woche bis Monat | Nur einmal täglich veröffentlicht. Auf 5 Minuten praktisch eine Konstante |
| **CPI/FOMC-Filter** | Krypto-Schwankung steigt rund um die Fed-Entscheidung stark, die Richtung ist zufällig ([arXiv](https://arxiv.org/pdf/2302.10252)) | Stunde | **Behalten als Risikofilter**, nicht als Punktequelle |
| **Regime-Detektor** | Hidden-Markov-Modelle finden 3–4 Marktphasen. Die Daten sind kurz (2016–2019) und meist ohne Kosten ([EconPapers](https://econpapers.repec.org/RePEc:eee:ecofin:v:57:y:2021:i:c:s1062940821000577)) [Einzel-Backtest] | Tag | Fügt Parameter hinzu. Ein einfacher Trend- plus Schwankungsfilter leistet Ähnliches |

Zwei Punkte aus der Tabelle verdienen eine Erklärung. **Erstens arbeiten Trend und Mittelwert-Rückkehr gegeneinander.** Ein Trendsignal sagt „kaufen, weil es steigt". RSI-2 sagt „kaufen, weil es gefallen ist". In einem gemeinsamen Score heben sie sich oft auf. Der Score landet dann in der Mitte, und der Schwellenwert filtert vor allem Rauschen. **Zweitens wird Fear & Greed nur einmal am Tag aktualisiert.** Mit 28 % Gewicht verschiebt er auf 5-Minuten-Kerzen im Grunde nur den Schwellenwert, und zwar einmal pro Tag. Das ist ein versteckter Regime-Schalter. Das kann man machen. Aber nichts rechtfertigt, sein Gewicht auf zwei Kommastellen genau festzulegen.

Auch eine wichtige Einschränkung bei Bitcoin gehört hierher. Hudson und Urquhart testeten rund 15.000 technische Regeln mit Korrektur für Mehrfachtests. Ergebnis: **Bei Bitcoin gab es außerhalb der Stichprobe keine Vorhersagekraft mehr**, nur noch bei anderen Coins [Wiss. Backtest] ([IDEAS](https://ideas.repec.org/a/spr/annopr/v297y2021i1d10.1007_s10479-019-03357-1.html)).

---

## Auf 5-Minuten-Kerzen frisst die Gebühr jeden Vorteil

### Die Kostenhürde ist drei- bis viermal größer als eine typische Kerze

Bitpanda Fusion verlangt in der ersten Stufe **0,25 % pro Order, für Maker und Taker gleich, bis 100.000 € Monatsvolumen** ([Cryptoticker](https://cryptoticker.io/en/bitpanda-fusion/reviews/)). *Maker* stellt eine Order ins Orderbuch, *Taker* nimmt eine bestehende an. Ein *Round Trip* (Kauf plus Verkauf) kostet also 0,50 %. Dazu kommt der Spread, also die Lücke zwischen Kauf- und Verkaufskurs.

Bitcoin schwankt grob 2,5–3,5 % pro Tag. Ein Tag hat 288 Fünf-Minuten-Kerzen. Eine typische 5-Minuten-Bewegung liegt damit bei etwa **0,15–0,2 %** [Eigene Rechnung]. Ein Round Trip kostet also so viel wie **2,5 bis 3,5 typische 5-Minuten-Bewegungen**. Das Signal muss Bewegungen vorhersagen, die mehrfach größer sind als das normale Rauschen. Erst dann ist man bei null.

Die Forschung bestätigt das. Eine Studie im *Journal of International Financial Markets* findet: **Kosten verändern die Profitabilität auf Minuten-Ebene dramatisch, auf Tagesebene kaum** [Wiss. Backtest] ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042443122000816)). Bakker testete 3.312 Regeln auf **5-Minuten-Daten** von BTC. Einige blieben nach Kosten signifikant. Aber **die Profitabilität war sehr instabil und nahm über die Zeit ab** [Wiss. Arbeit, Masterarbeit] ([Erasmus](https://thesis.eur.nl/pub/41546/)). In einer anderen Auswertung machte hoher Umsatz aus +73 % Brutto-Rendite −29 % netto (Zuordnung zur genauen Studie unsicher) ([Springer](https://link.springer.com/article/10.1007/s11408-025-00474-9)).

### Was das in Euro bedeutet

Bei 20 € pro Position [Eigene Rechnung]:

| Round Trips in 90 Tagen | Gebühren in € | Anteil am 100-€-Konto pro Quartal | Hochgerechnet pro Jahr |
|---|---|---|---|
| 30 (≈ 10 pro Monat) | 3 € | 3 % | ~12 % |
| 100 | 10 € | 10 % | ~40 % |
| 300 | 30 € | 30 % | ~120 % |

Der Spread kommt noch dazu. Ein Bot mit 300 Round Trips pro Quartal müsste also allein die Gebühren mit einem Vorteil von mehr als dem ganzen Konto pro Jahr aufholen.

### Trefferquote und Gewinnziel hängen zusammen

Die nötige Trefferquote ergibt sich aus Gewinnziel (TP), Stop (SL) und Kosten. Die Formel lautet: nötige Trefferquote = (SL + 0,5 %) / (TP + SL) [Eigene Rechnung]:

| Take-Profit / Stop-Loss | Netto-Gewinn / Netto-Verlust | Nötige Trefferquote |
|---|---|---|
| 1,0 % / 1,0 % | +0,5 % / −1,5 % | **75 %** |
| 2,0 % / 1,0 % | +1,5 % / −1,5 % | 50 % |
| 3,0 % / 1,5 % | +2,5 % / −2,0 % | 44 % |
| 4,0 % / 2,0 % | +3,5 % / −2,5 % | 42 % |

Trendsysteme treffen typischerweise in 35–55 % der Fälle. Sie leben von wenigen großen Gewinnern. **Der 1-%-Take-Profit-Boden gibt die Hälfte jedes Mindest-Gewinners an die Gebühr ab.** Als grobe Regel sollte das Gewinnziel mindestens viermal so groß sein wie die Round-Trip-Kosten, hier also 2 % oder mehr.

Die neuen Einstellungen (TP 12×ATR, SL 5×ATR) gehen in die richtige Richtung. *ATR* (Average True Range) ist die durchschnittliche Schwankungsbreite einer Kerze. Auf 5-Minuten-Kerzen ist die ATR aber klein. Liegt sie bei grob 0,15–0,2 %, dann liegt der Stop bei etwa 0,75–1 % und das Ziel bei etwa 1,8–2,4 %. Die nötige Trefferquote steigt in ruhigen Phasen damit auf **44–49 %** [Eigene Rechnung, ATR-Annahme]. Gemessen hat der Bot 35–40 %. Euer Backtest nennt ~41 % als Schwelle. Das passt zu Phasen mit höherer Schwankung. Die Lücke bleibt in jedem Fall bestehen. Ein Trade mit 12×ATR-Ziel auf 5-Minuten-Kerzen läuft in der Praxis ohnehin viele Stunden. Der Bot hält also schon fast wie ein Stunden-System, entscheidet aber im 45-Sekunden-Takt. Die Konsequenz aus der Forschung: **5-Minuten-Daten nur für den Einstiegs-Zeitpunkt nutzen, nicht für die Entscheidung selbst.**

### Ein anderer Handelsplatz senkt die Hürde stärker als jede Parameter-Optimierung

Weil Maker und Taker auf Fusion gleich viel zahlen, **sparen Limit-Orders auf Fusion keine Gebühren**. Sie sparen nur den Spread. Andere Plätze sind deutlich günstiger. Alle Werte stammen aus Suchauszügen und sind ungeprüft:

| Handelsplatz | Maker / Taker | Round Trip (Maker) | Quelle |
|---|---|---|---|
| Bitpanda Fusion | 0,25 % / 0,25 % | 0,50 % | [Cryptoticker](https://cryptoticker.io/en/bitpanda-fusion/reviews/) |
| One Trading (Wien) | 0,10 % / 0,20 % | 0,20 % | [onetrading.com](https://www.onetrading.com/fees) |
| Bybit EU | 0,10 % / 0,25 % | 0,20 % | [Bybit EU](https://www.bybit.eu/en-EU/help-center/article/Bybit-Spot-Fees-Explained) |
| Kraken Pro (ab 9. Juli 2026) | 0,40 % / 0,80 % | 0,80 % | [Kraken Blog](https://blog.kraken.com/product/pro/new-kraken-pro-fee-tiers) |
| Coinbase Advanced EU | 0,25 % / 0,50 % | 0,50 % | [Datawallet](https://www.datawallet.com/crypto/coinbase-fees) |

Mit 0,20 % statt 0,50 % Round Trip sinkt die nötige Trefferquote bei TP 1 % / SL 1 % von 75 % auf 60 % [Eigene Rechnung]. Die Notiz meldet zu Kraken allerdings widersprüchliche Angaben. Und ein Wechsel hat einen versteckten Preis bei der Steuer (siehe unten).

---

## BTC, ETH und SOL sind praktisch eine einzige Wette, und 100 € reichen nur zum Lernen

### Hohe Gleichläufigkeit

Die Wahl der drei Assets ist bei der Liquidität vernünftig. Bei BTC-EUR und ETH-EUR liegen die Spreads auf guten Börsen **unter 2 Basispunkten** (0,02 %) ([Kaiko](https://www.kaiko.com/news/kaiko-research-highlights-bitvavos-position-in-eur-spot-trading)). Bei 20-€-Orders ist der Spread neben 0,50 % Gebühr fast egal. **Die Gebühr ist die eigentliche Grenze, nicht der Spread.** Für SOL-EUR und speziell für Fusion fand die Recherche keine Spread-Zahlen.

Das Problem ist die Gleichläufigkeit. Die *Korrelation* misst, wie stark sich zwei Kurse gemeinsam bewegen (1 = völlig gleich). Über 90 Tage liegt sie bei **BTC/ETH ≈ 0,90, BTC/SOL ≈ 0,92 und ETH/SOL ≈ 0,90** ([Sharpe.ai](https://www.sharpe.ai/learn/crypto-correlation-matrix)). Zeitweise erreichte BTC/SOL sogar 0,99 ([CryptoPotato](https://cryptopotato.com/defillama-crypto-correlations-hit-record-highs-as-btc-sol-reaches-0-99/)). Zwei gleichzeitige Long-Positionen in BTC und SOL wirken daher wie etwa **eine einzige Wette mit doppelten Gebühren** [Eigene Rechnung].

Ein breiteres Coin-Universum ist aber kein Gratis-Vorteil. Die großen Momentum-Gewinne der Forschung stammen oft aus kleinen, illiquiden Coins. Dort fressen Kosten sie auf ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001509)). Liu, Tsyvinski und Wu finden Momentum vor allem bei großen Coins: **4,2 % pro Woche über dem Median der Größe, nicht signifikante 0,6 % darunter** [Wiss. Backtest] ([NBER](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). Diese Ergebnisse beruhen allerdings auf Long-Short-Portfolios über viele Coins mit wöchentlicher Umschichtung. Ein Long-only-Bot mit zwei Plätzen kann das nicht nachbauen.

### 100 € Kapital: die Mindestorder bremst, nicht die Risikoregel

Fusion verlangt mindestens 25 € pro Order, je nach Asset unterschiedlich ([Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/16663481714844-Bitpanda-Fusion)). Euer Bot hat für BTC-EUR genau 25 € gemessen. Positionen von 15–20 € sind deshalb teils gar nicht möglich. **Zwei Positionen binden mindestens 50 % des Kapitals.** Die 2-%-Risikoregel (2 € auf 100 €) erlaubt bei 25 € Positionsgröße einen Stop bis 8 % Abstand. Die Risikoregel ist also nicht der Engpass, die Mindestorder schon [Eigene Rechnung].

Mehr Kapital senkt die Gebühr in Prozent nicht. Die nächste Fusion-Stufe beginnt erst bei 100.000 € Monatsvolumen. Mehr Kapital hilft an zwei Stellen [Eigene Rechnung]:

| Kapital | Was sich ändert |
|---|---|
| 100 € | Lern- und Testkonto. +20 % pro Jahr wären 20 € brutto, nach 27,5 % KESt rund 14,50 € |
| 250–500 € | Positionen von 50–100 € überwinden alle Mindestorders, 2–3 Positionen und Teilausstiege möglich |
| 2.000–5.000 € | Absolute Gewinne übersteigen typische Nebenkosten (Server, Zeit). Die Gebühr in Prozent bleibt gleich |

### Steuer: Bitpanda ist bequem, ein Wechsel kostet Aufwand

Krypto-Gewinne auf Neuvermögen (gekauft ab 1. März 2021) werden in Österreich mit **27,5 % KESt** besteuert. Jeder Verkauf in Euro ist ein steuerpflichtiger Vorgang ([BMF](https://www.bmf.gv.at/themen/steuern/sparen-veranlagen/steuerliche-behandlung-von-kryptowaehrungen.html)). Seit 2023 gilt der **gleitende Durchschnittspreis** als Anschaffungswert ([crypto-tax.at](https://www.crypto-tax.at/einkuenfteermittlung-bei-realisierten-wertsteigerung-aus-kryptowaehrungen-gleitender-durchschnittspreis-ab-01-01-2023/)). Bitpanda ist „steuereinfach" und zieht die KESt seit 1.1.2024 automatisch ab ([crypto-tax.at](https://www.crypto-tax.at/krypto-steuereinfach-in-osterreich-kest-bei-bitpanda/)). Bei einem ausländischen Handelsplatz muss man Hunderte Bot-Trades selbst in der Steuererklärung (E1kv) abrechnen. Ob One Trading ebenfalls steuereinfach ist, wurde nicht geprüft. Wichtig für die Buchhaltung: Der Gewinn pro Trade in `trades.db` wird **nicht** der steuerlichen Zahl entsprechen, weil das Finanzamt Durchschnittspreise verwendet.

---

## Was erfolgreiche Strategien gemeinsam haben vs. unser Bot

| Merkmal | Belegte erfolgreiche Strategien | Unser Bot heute | Bewertung |
|---|---|---|---|
| **Entscheidungs-Zeitrahmen** | Tag bis Woche, höchstens 6-Stunden-Kerzen ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907), [arXiv](https://arxiv.org/html/2602.11708v1)) | 5-Minuten-Kerzen, Prüfung alle 45 s | **Schwach:** Deutlich zu schnell |
| **Handelshäufigkeit** | Wenige Dutzend Trades pro Jahr | Viele Signale pro Tag möglich | **Schwach:** Gebühren dominieren |
| **Zahl der Regeln** | Eine Signalfamilie, über Rückblick-Längen gemittelt | ~8 verschiedenartige Signale, 3 Gewichte, Regime-Logik | **Schwach:** Hohes Overfitting-Risiko |
| **Parameter-Einstellung** | Kaum fein eingestellt, robust über Varianten | Schwellenwert 0,647, wandert 0,65→0,87 | **Schwach:** Instabil |
| **Positionsgröße** | Nach Schwankung skaliert ([Man Group](https://www.man.com/insights/crypto-too-hot-to-handle)) | Feste Slots 15–25 €, ATR-Stops | **Teilweise:** feste Größe statt Schwankungs-Skalierung |
| **Trefferquote / Gewinnverhältnis** | Niedrige Trefferquote, große Gewinner laufen lassen | 35–40 % Treffer, TP-Boden 1 % | **Teilweise:** Gewinner zu klein im Verhältnis zur Gebühr |
| **Trendrichtung als Filter** | Kern der Strategie ([WUSTL](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)) | 200-EMA-Richtungsfilter | **Gut:** richtiger Kern |
| **Mittelwert-Rückkehr** | Bei BTC schwächer als Trend | RSI-2 im selben Score | **Schwach:** Arbeitet gegen den Trend |
| **Asset-Universum** | Viele liquide Coins oder bewusst 1–2 | 3 fast gleichläufige Coins | **Teilweise:** Wenig Streuung |
| **Kosten im Verhältnis zur Bewegung** | Kosten klein gegenüber Tagesbewegungen | 0,50 % ≈ 3× typische 5-Min-Bewegung | **Schwach:** Strukturell im Nachteil |
| **Verhalten im Bärenmarkt** | Schutz vor großen Verlusten | −7 % statt −28 % | **Gut:** Passt zum Muster |
| **Harte Risikogrenzen** | Üblich | 2 % pro Trade, 10 % Tagesstopp | **Gut:** vorhanden und sinnvoll |
| **Makro-Ereignisse** | Als Risikokontrolle sinnvoll | CPI/FOMC-Filter im Score | **Teilweise:** besser als harter Filter statt Punkte |
| **Prüfung gegen Overfitting** | Mehrfachtest-Korrektur, Walk-Forward | Drei 90-Tage-Fenster | **Teilweise:** Zu wenig |
| **Nachweis** | Lange Backtests oder Live-Zahlen | Paper-Modus, Backtests negativ | **Teilweise:** Gut, dass noch Paper |

---

## Empfehlungen: was bleibt, was sich ändert, was als Nächstes getestet wird

Die Reihenfolge folgt dem erwarteten Nutzen. Punkt 1 wirkt am stärksten.

### Behalten

| # | Was | Warum |
|---|---|---|
| B1 | **Paper-Modus** | Kein Backtest war positiv. Live-Geld erst nach stabil positivem Paper-Ergebnis |
| B2 | **Harte Risikogrenzen (2 % pro Trade, 10 % Tagesstopp)** | Sicherheitskritisch, unabhängig von der Strategie |
| B3 | **200-EMA-Trendrichtung** | Der am besten belegte Baustein. Er erklärt vermutlich das gute Bärenmarkt-Ergebnis |
| B4 | **CPI/FOMC-Filter** | Schwankung steigt um Termine nachweislich. Billige Risikokontrolle |
| B5 | **Realistisches Gebührenmodell (0,25 %) und echter Bid/Ask aus `/v1/orderbook`** | Ohne das sind alle Backtests wertlos |

### Ändern (nach Priorität)

| # | Änderung | Konkret | Erwarteter Effekt |
|---|---|---|---|
| **Ä1** | **Signal auf Tages-Kerzen (oder mindestens 4-Stunden-Kerzen) verlegen** | Einstieg nur, wenn der Tages-Schlusskurs über einem Bündel von Durchschnitten oder Donchian-Hochs liegt, z. B. 20/50/100/200 Tage. 5-Minuten-Daten nur noch für den genauen Einstiegs-Zeitpunkt | Weniger Trades, größere Bewegungen pro Trade. Gebühren werden klein im Verhältnis |
| **Ä2** | **Trade-Zahl hart begrenzen** | Ziel: insgesamt unter ~10–15 Round Trips pro Monat | Gebühren unter ~12 % des Kontos pro Jahr [Eigene Rechnung] |
| **Ä3** | **Score vereinfachen** | RSI-2 aus dem Trend-Score entfernen. Gleiche Gewichte statt 0,55/0,28/0,17. Fear & Greed nur als langsamer Größenregler (z. B. halbe Position bei Extremwerten), nicht als 28-%-Punktequelle. Makro als harte Pause, nicht als Punkte | Weniger Stellschrauben, geringeres Overfitting-Risiko |
| **Ä4** | **Keine Schwellenwerte auf drei Nachkommastellen** | Runde Werte wählen und prüfen, ob benachbarte Werte ähnlich gut sind („Plateau" statt „Spitze") | Robuster außerhalb der Stichprobe |
| **Ä5** | **Gewinnziel an die Gebühr koppeln** | TP-Boden von 1 % auf mindestens 2 % anheben (≥ 4× Round-Trip-Kosten). Alternativ: kein festes Ziel, sondern Trailing-Ausstieg über Trendbruch (z. B. Tagesschluss unter Donchian-Tief) testen | Gewinner bezahlen die Gebühr nicht mehr zur Hälfte |
| **Ä6** | **Nur eine Position gleichzeitig (oder BTC + ETH, nicht drei Coins)** | Bei Korrelation 0,9 bringt eine zweite Position kaum Streuung | Weniger Gebühren, gleiches Risiko |
| **Ä7** | **Positionsgröße nach Schwankung skalieren** | Größe umgekehrt zur 30-Tage-Schwankung, aber nie unter der 25-€-Mindestorder | Volatilitäts-Skalierung verbesserte die Sharpe-Ratio einer Bitcoin-Position um rund 0,4 ([Man Group](https://www.man.com/insights/crypto-too-hot-to-handle)) [Branchenstudie] |
| **Ä8** | **Regime-Detektor verschlanken** | Auf zwei langsame Zustände reduzieren: Tages-Trend auf/ab und Schwankung hoch/niedrig. Damit Größe skalieren, nicht zwischen Strategien wechseln | Weniger Parameter |
| **Ä9** | **Hebel-Obergrenze von 50× auf 1× setzen** | Fusion ist Spot. 50× ist weit über allem, was Privatkunden in der EU legal nutzen können | Konfiguration entspricht der Wirklichkeit |
| **Ä10** | **Gebühren-Standardwerte im Code korrigieren** | `core/order_engine.py`, `core/live_order_engine.py`, `data/backtester.py`, `main.py` und `tools/backtest_confluence.py` fallen ohne Konfiguration auf **0,04 % / 0,06 %** zurück (Fund im Code, `settings.yaml` selbst steht korrekt auf 0,25 %). Standard auf 0,0025 setzen | Kein Backtest läuft mehr versehentlich mit Fantasie-Gebühren |

### Als Nächstes testen

| # | Test | Wie |
|---|---|---|
| T1 | **Einfacher Tages-Trendfilter gegen Confluence gegen Kaufen-und-Halten** | Gleiche Zeitfenster, gleiche Kosten (0,25 % + echter Spread). Vergleichen nach Sharpe-Ratio und maximalem Drawdown, nicht nur nach Rendite |
| T2 | **Overfitting-Test** | Walk-Forward (auf altem Zeitraum einstellen, auf neuem prüfen) über mehr als drei Fenster. Zusätzlich PBO-Test ([CRAN pbo](https://packages.oit.ncsu.edu/cran/web/packages/pbo/readme/README.html)) und Deflated Sharpe ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). Die Zahl aller probierten Varianten mitzählen |
| T3 | **Längere Historie** | Mindestens einen vollen Zyklus mit Bullen-, Bären- und Seitwärtsphase testen, nicht nur 90 Tage |
| T4 | **Gebühren-Szenarien** | Dieselbe Strategie mit 0,20 % Round Trip (One Trading Maker) rechnen. Lohnt der Unterschied den Steueraufwand? |
| T5 | **Mindestorders aller Paare** | `/v1/pairs` für ETH-EUR und SOL-EUR abfragen |
| T6 | **Kapital erst erhöhen, wenn Paper-Ergebnis positiv** | Über mehrere Monate Paper-Handel mit der vereinfachten Strategie. Dann auf 250–500 € gehen, damit Mindestorders kein Problem mehr sind |

---

## Vor dem Margin-Handel: 2:1 statt 50:1, und Shorts bringen weniger als erhofft

### Was Hebel und Shorts sind und was sie kosten

*Margin-Handel* heißt: Man leiht sich Geld oder Coins und handelt mit einer größeren Position als dem eigenen Kapital. Das ist der *Hebel*. Ein *Short* ist eine Wette auf fallende Kurse. *Liquidation* bedeutet: Die Börse schließt die Position zwangsweise, weil das eigene Kapital aufgebraucht ist. *Perpetuals* sind Terminkontrakte ohne Ablaufdatum. Dort zahlen Long- und Short-Seite einander alle 8 Stunden eine Ausgleichszahlung, das *Funding*.

| Kostenart | Höhe | Quelle |
|---|---|---|
| **Bitpanda Margin: Finanzierung** | Tag 1–60: **0,18 % pro Tag** (≈ 66 % pro Jahr auf die Positionsgröße), danach gestaffelt günstiger | [Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/21417526386588-Bitpanda-Margin-Trading) (Suchauszug, eine andere Quelle nennt abweichende Werte) |
| **Bitpanda Margin: Schließen / Liquidation** | 0,3 % beim Schließen, **1 % Liquidationsgebühr** | dito |
| **Perpetuals: Funding** | Grundwert **0,01 % pro 8 h ≈ 11 % pro Jahr**, zahlen meist die Longs | [BitMEX](https://blog.bitmex.com/2025q3-derivatives-report) |
| **Perpetuals: Funding in Boomphasen** | Carry bis zu **60 % pro Jahr** | [BIS](https://www.bis.org/publ/work1087.pdf) |
| **Perpetuals: Handelsgebühr** | z. B. Coinbase EU ab 0,02 % | [Cointelegraph](https://cointelegraph.com/news/coinbase-perpetual-futures-contracts-europe-esma) |

Ein Bitpanda-Margin-Trade über einen Tag kostet rund 0,18 % + 0,3 % ≈ **0,48 %**, fast so viel wie heute der Spot-Round-Trip. Über 10 Tage kommen 1,8 % Finanzierung auf die Positionsgröße dazu. Bei 5-fachem Hebel sind das rund 9 % des eingesetzten Kapitals [Eigene Rechnung]. **Für langsame Trendfolge mit Haltedauern von Tagen bis Wochen ist Bitpanda-Margin damit sehr teuer.** Außerdem steigen die Gebühren in Euro mit dem Hebel, weil sie auf die ganze Position berechnet werden. Bei 400 € Position (4× auf 100 €) kostet ein Round Trip 2 €, also das ganze Risiko-Budget eines Trades [Eigene Rechnung].

### Shorts: die Belege sind gemischt und eher negativ

Eine Studie zu Krypto-Momentum fand: **Long-only-Portfolios schlugen den Markt auch nach Kosten, Short-only-Portfolios lieferten ungünstige Ergebnisse** [Wiss. Backtest, Suchauszug] ([AUT](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)). Eine Erasmus-Abschlussarbeit fand bei keiner Long-Short-Variante einen signifikanten Mehrwert [Abschlussarbeit] ([Erasmus](https://thesis.eur.nl/pub/44390/Wisselink-NJ-483391-BA-thesis.pdf)). Andere Arbeiten melden das Gegenteil ([Portfolio123-Upload](https://community.portfolio123.com/uploads/short-url/amrMsuqIKzdHcHHyvMud4YNPwZB.pdf)). Auch die 6-Stunden-Studie mit Sharpe 2,41 gewichtet die Short-Seite bewusst schwächer ([arXiv](https://arxiv.org/abs/2602.11708v1)). Shorts haben bei Krypto drei Gegner: den langfristigen Aufwärtstrend, plötzliche Short-Squeezes und meist positives Funding in Boomphasen, wenn Shorts gerade am stärksten unter Druck stehen.

Wie gefährlich Hebel ist, zeigte der **10. Oktober 2025**. Bitcoin fiel um rund 14 %, über **19 Mrd. $** wurden an einem Tag liquidiert, über 1,6 Mio. Konten ausgelöscht. Rund 90 % davon waren Long-Positionen [Live, Marktdaten] ([21Shares](https://www.21shares.com//research/record-crypto-liquidations-amid-tariff-shock)). In einer Simulation mit 75-fachem Hebel wurden **97,3 % der Trades liquidiert, im Schnitt nach etwa 30 Minuten** ([NYU Stern](https://www.stern.nyu.edu/sites/default/files/assets/documents/Duron-Carielo_Is%20There%20A%20Future%20In%20Perpetual%20Futures.pdf)). Bei CFDs (Differenzkontrakten) verlieren in der EU **74–89 % der Privatkonten** Geld [Behörde] ([ESMA](https://www.esma.europa.eu/node/84933)). Diese Zahlen betreffen menschliche Trader, nicht Bots mit festem Stop. Einen Bot treffen aber dieselben Kosten und dieselben Kurslücken.

### EU- und Österreich-Regeln: realistisch sind 2:1, nicht 50:1

Für Privatkunden in Österreich gelten im Oktober 2026 zwei Regelwerke. Stand laut Suchauszügen, Originale nicht lesbar:

| Regel | Inhalt | Quelle |
|---|---|---|
| **ESMA-Erklärung, 24. Feb. 2026** | Perpetuals für Privatkunden fallen „wahrscheinlich" unter die CFD-Regeln. Das bedeutet **max. 2:1 Hebel auf Krypto**, Zwangsschließung bei 50 % Margin und Schutz vor Negativsaldo | [ESMA](https://www.esma.europa.eu/press-news/esma-news/esma-reminds-firms-their-obligations-under-cfd-product-intervention-measures) |
| **FMA-PIV (Österreich, seit 2019)** | Dauerhafte nationale CFD-Beschränkung inkl. virtueller Währungen, Hebel 30:1 bis 2:1 | [FMA](https://www.fma.gv.at/en/?p=22133), [RIS](https://www.ris.bka.gv.at/Dokumente/Bundesnormen/NOR40264128/NOR40264128.html) |
| **MiCA** | Übergangsfrist endete **1. Juli 2026**. Offshore-Börsen ohne EU-Lizenz sind für EU-Kunden nicht mehr legal nutzbar | [ESMA](https://www.esma.europa.eu/sites/default/files/2026-04/ESMA75-113276571-1679_Statement_on_the_end_of_transitional_periods_under_MiCA.pdf) |

Welche Anbieter kommen in Frage?

| Anbieter | Angebot | Einschränkungen |
|---|---|---|
| **Bitpanda Margin Trading** | Spot-basiert, 2×/3×/5×/10×, über 100 Coins, mit TP/SL | Bisher **nur Long**, Shorts „geplant". **Fusion selbst hat keine Margin.** API-Zugang ungeklärt ([Bitpanda Blog](https://blog.bitpanda.com/en/bitpanda-margin-trading-smarter-way-trade-crypto-10x-leverage)) |
| **One Trading** | Regulierte Perpetuals BTC/EUR, ETH/EUR, bis 10× für berechtigte Kunden, auch in Österreich ([The TRADE](https://www.thetradenews.com/one-trading-expands-retail-access-for-crypto-perpetual-futures-venue/)) | Hebel für Privatkunden nach ESMA-Erklärung evtl. auf 2× gesenkt (nicht bestätigt) |
| **Kraken EU / Coinbase EU** | Perpetuals bis 10× ([crypto.news](https://crypto.news/kraken-launches-crypto-collateral-futures-eu-2025/), [Cointelegraph](https://cointelegraph.com/news/coinbase-perpetual-futures-contracts-europe-esma)) | Gleiche Unsicherheit wie One Trading |
| **Bybit EU** | Nur Spot | Keine Perpetuals |

Mit dem Status „professioneller Kunde" kann man der 2:1-Grenze ausweichen. Dafür braucht man aber Vermögen, Handelserfahrung und Handelsfrequenz, die ein 100-€-Konto nicht erfüllt (Allgemeinwissen, nicht recherchiert). Wie Derivate-Gewinne in Österreich besteuert werden, wurde nicht untersucht. Das sollte vor dem Start geklärt werden.

### Der Hebel folgt aus dem Risiko, nicht umgekehrt

Hat man einen Stop, der verlässlich greift, dann gilt: Euro-Risiko = Positionsgröße × Stop-Abstand. Der Hebel legt nur fest, wie viel Kapital gebunden ist. Beispiel mit 100 € und 2 € Risiko: Bei 2 % Stop-Abstand ergibt das 100 € Position (1×). Bei 0,5 % Stop-Abstand ergibt das 400 € Position (4×) [Eigene Rechnung]. **Mit einer langsamen Tages-Strategie und weiten Stops braucht der Bot realistisch gar keinen oder höchstens 2× Hebel.** Das liegt genau im legalen Rahmen. Der eigentliche Nutzen von Margin wäre die Short-Seite, und die ist wie gezeigt fraglich. Die Liquidationsgrenze liegt bei 10× etwa 9–10 % vom Einstieg entfernt, bei 2× etwa 45–50 % [Eigene Rechnung]. Ein 14-%-Absturz wie am 10. Oktober 2025 hätte jede 10×-Long-Position ausgelöscht, deren Stop nicht rechtzeitig gefüllt wurde.

### Checkliste: was vor dem ersten Margin-Trade fertig sein muss

| # | Vorbereitung |
|---|---|
| M1 | **Margin als eigenes Produkt bauen**, nicht als „Hebel-Parameter" auf Spot. Eigene Order-Engine-Klasse hinter einem Börsen-Adapter. Spot bleibt Standard |
| M2 | **Konfiguration pro Börse:** `allow_shorts`, `max_leverage` (1 für Fusion, ≤ 2 für CFD-artige Perpetuals, ≤ 10 für Bitpanda Margin), `margin.mode: isolated`, Finanzierungsmodell (Satz, Intervall, Staffel nach Haltedauer, Schließ- und Liquidationsgebühr) |
| M3 | **Backtester erweitern:** Finanzierung/Funding pro Kerze auf die Positionsgröße buchen. Liquidationspreis simulieren. Liquidation auf Kerzen-Hoch/Tief **vor** dem Stop prüfen. Liquidationsgebühr abziehen |
| M4 | **Prüfungen vor jedem Trade:** Abstand zur Liquidation ≥ 2–3 × Stop-Abstand plus Puffer für Kurslücken. Trade ablehnen, wenn Finanzierung über die erwartete Haltedauer plus Gebühren einen großen Teil des erwarteten Gewinns frisst |
| M5 | **Ergebnisse getrennt ausweisen:** Long-Gewinn, Short-Gewinn, Funding, Gebühren, Liquidationen. Nur so sieht man, ob Shorts wirklich etwas bringen |
| M6 | **Test Long-only gegen Long/Short** mit identischem Kostenmodell, getrennt für Bullen-, Bären- und Seitwärtsphasen |
| M7 | **API-Zugang klären:** Ist Bitpanda Margin oder sind One-Trading-Perpetuals über die vom Bot genutzte Schnittstelle erreichbar? Davon hängt alles ab |
| M8 | **Steuer und Regeln prüfen:** Steuerliche Behandlung von Derivaten in Österreich. Aktueller Hebel für Privatkunden beim gewählten Anbieter |
| M9 | **Lange Paper-Phase mit Margin-Kosten**, bevor echtes Geld fließt. Das deckt sich mit den Live-Schutzmechanismen des Projekts |

---

## Fazit

Die eigentliche Erkenntnis ist keine Strategie-Frage, sondern eine Frage des Maßstabs. Unser Bot hat einen soliden Kern, den langsamen Trendfilter, und ein gutes Sicherheitsgerüst. Darüber liegt eine schnelle, fein justierte Signalschicht. Die erzeugt vor allem Gebühren und Scheingenauigkeit. Die verlustreichen Backtests sind deshalb ein nützlicher Befund und kein Versagen. Sie zeigen, dass die 5-Minuten-Schicht den Kern verwässert. Das Bärenmarkt-Ergebnis zeigt, dass der Kern funktioniert. Der wahrscheinlichste Weg zu einem positiven Ergebnis ist Weglassen, nicht Hinzufügen: Tagesentscheidung, ein Signaltyp, eine Position, wenige Trades.

Für Margin gilt dieselbe Logik verschärft. Hebel vervielfacht Gebühren und Finanzierung in Euro. Die Short-Seite hat in Krypto den schwächsten Rückenwind. Und das Recht erlaubt Privatkunden ohnehin kaum mehr als 2:1. Margin lohnt sich daher erst, wenn eine einfache Long-only-Tagesstrategie im Paper-Modus über Monate nach realen Kosten positiv ist. Dann sollte Margin als kleiner, getrennt gemessener Zusatz dazukommen, nicht als Hebel, der eine schwache Strategie retten soll. Eine schwache Strategie macht der Hebel nur schneller schwach.
