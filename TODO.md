# TODO

## Offen

- [ ] **Auf den VPS deployen** — `main` enthält Änderungen, die auf dem Server noch nicht laufen:
  - Gewinnziel-Boden 2 % (`min_tp_pct: 0.020`), TP 12×ATR / SL 5×ATR
  - Gebühren-Standard 0,25 % überall im Code (vorher 0,06 % ohne Config)

  Am Mac im Repo-Ordner:
  ```
  git pull
  ./deploy/vps/deploy.sh
  ```
  Danach im Dashboard → Einstellungen prüfen: „Gebühr pro Order 0,25 %“.

- [ ] Alten Branch `claude/kraken-pro-dashboard` auf GitHub löschen (Branches → Mülleimer)

## Später (aus `docs/KRYPTO_BOTS_IM_VERGLEICH.md`)

- [ ] Vergleichstest: einfacher Tages-Trendfilter vs. Confluence vs. Kaufen-und-Halten
- [ ] Signal auf Tages- oder 4-Stunden-Kerzen verlegen, max. 10–15 Trades pro Monat
- [ ] Confluence-Score vereinfachen (RSI-2 raus, Fear & Greed nur als Größenregler)
- [ ] Nur eine Position gleichzeitig
- [ ] Hebel-Obergrenze in der Config von 50× auf 1× (Spot)
- [ ] Vor Margin-Handel: Checkliste im Bericht abarbeiten (EU-Retail realistisch 2:1)
