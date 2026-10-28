# TODO

## Offen

- [ ] Tages-Trendfilter beobachten (läuft seit 14.10.2026 als Trend-1, 1000 €; auf
  https://bot.silvertoast.at oben rechts umschalten). Erwartung laut Backtest: wenige Trades,
  Wert zeigt sich erst über Monate bzw. im nächsten Einbruch.

- [ ] Panda-9 beobachten: seit 28.10.2026 Donchian-Ausbruch 20/10 auf Tageskerzen,
  neu mit 100 € (alte Confluence-Datenbank am VPS gesichert). Erwartung laut Backtest:
  vorsichtiger als Trend-1, in Seitwärtsphasen leicht im Minus.

- [ ] Alten Branch `claude/kraken-pro-dashboard` auf GitHub löschen (Branches → Mülleimer)

## Später (aus `docs/KRYPTO_BOTS_IM_VERGLEICH.md`)

- [x] Vergleichstest: einfacher Tages-Trendfilter vs. Confluence vs. Kaufen-und-Halten (T1, 12.–14.10.2026)
- [x] Signal auf Tageskerzen verlegen (Panda-9 = Donchian, 28.10.2026)
- [ ] ~~Confluence-Score vereinfachen~~ — entfällt, Confluence ist abgeschaltet
- [ ] Nur eine Position gleichzeitig — bei 100 € steckt dann nur die 30-€-Mindestorder im Markt; live 2
- [x] Hebel-Obergrenze in der Config von 50× auf 1× (Spot, 28.10.2026)
- [ ] Vor Margin-Handel: Checkliste im Bericht abarbeiten (EU-Retail realistisch 2:1)
