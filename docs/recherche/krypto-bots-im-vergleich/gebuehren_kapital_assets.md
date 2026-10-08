# Fees, Small Capital, Trade Frequency and Asset Selection: Can a 100 € Retail Crypto Bot on Bitpanda Fusion (AT/EU) Be Profitable?

Research date: 2026-10-06. Method note: all primary fee pages (bitpanda.com, docs.fusion.bitpanda.com, support.bitpanda.com, kraken.com, support.kraken.com, bmf.gv.at, cryptoslate.com) were **blocked by the network egress proxy** in this session, so numbers below come from search-engine extracts of those pages and from secondary sources (reviews, aggregators). Every figure should be re-checked against the official page before it goes into code or config. Dates are given where the source states them.

---

## 1. Fee math: how much gross edge per trade, and how many trades per month are sustainable at 0.50 % round trip?

### Takeaway
At 0.25 % per side, every round trip costs 0.50 % in fees plus roughly 0.02–0.2 % in spread/slippage. 300 round trips in 90 days cost about 150 % of position size, which is 30 % of a 100 € account per quarter and about 120 % per year. With a 1 % take-profit and a stop of about 1 %, the bot needs a win rate of roughly 72–75 % just to break even. The structural fix is fewer, larger moves: TP/SL distances of at least 2–3 % and well under about 30 round trips a month. Higher frequency only works on a venue with maker fees near 0.10 % or lower.

### Cited Findings
- Bitpanda Fusion charges one flat rate per tier with no maker/taker split, starting at **0.25 % for 30-day volume up to 100,000 €** and falling through seven tiers to 0.02 % above 250 M € — [Cryptoticker Bitpanda Fusion review 2026](https://cryptoticker.io/en/bitpanda-fusion/reviews/); [Cryptoticker Bitpanda vs Fusion 2026](https://cryptoticker.io/en/comparison/bitpanda-vs-bitpanda-fusion/)
- Academic evidence that turnover dominates crypto strategy results: one study reports that 138 % daily turnover produced about 101 % annual cost drag, turning a +73 % gross return into −29 % net — search extract attributed to work linked from [Springer, "Cryptocurrency momentum has (not) its moments" (2025)](https://link.springer.com/article/10.1007/s11408-025-00474-9) and [ScienceDirect, "Cryptocurrency anomalies and economic constraints" (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001509). The exact attribution of the 138 %/101 % figure to one specific paper could not be confirmed because full text was not accessible.
- Momentum and anomaly portfolios that are significant before costs often become insignificant after realistic transaction costs and daily price moves; moderate costs leave the long side profitable, while high costs hurt mainly the short leg — [ResearchGate/AUT: "Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market: A Comprehensive Analysis under Realistic Assumptions"](https://www.researchgate.net/publication/377457967_Time-Series_and_Cross-Sectional_Momentum_in_the_Cryptocurrency_Market_A_Comprehensive_Analysis_under_Realistic_Assumptions)
- Hudson & Urquhart (2019) tested about 15,000 technical rules on cryptocurrencies, mostly at **daily** frequency, and found break-even transaction costs well above typical crypto costs — [Annals of Operations Research](https://link.springer.com/article/10.1007/s10479-019-03357-1)
- Bakker (2018) tested 3,312 **intraday (5-minute)** rules on BTC/USD from 2013 to 2017. Some stayed significant after data-snooping and cost adjustment, but "profitability is highly unstable and declines over time" — [Erasmus thesis](https://thesis.eur.nl/pub/41546/)
- Svogun & Bazán-Palomino (2022) studied 69 moving-average and breakout rules on daily and 1-minute data, with and without transaction costs (2016–2021) — [Universidad del Pacífico](https://faculty.up.edu.pe/es/publications/technical-analysis-in-cryptocurrency-markets-do-transaction-costs/)

### Inferences (worked examples, own calculations from the 0.25 %/side fee above)
**A. Turnover cost**

| Scenario (round trips / 90 days) | Fee cost as % of position | € cost at 20 € position | % of 100 € account / 90 d | Annualised % of account |
|---|---|---|---|---|
| 100 | 50 % | 10 € | 10 % | ~40 % |
| 300 | 150 % | 30 € | 30 % | ~120 % |
| 500 | 250 % | 50 € | 50 % | ~200 % |
| 30 (≈10/month) | 15 % | 3 € | 3 % | ~12 % |

Spread and slippage come on top of these fees. At an assumed 0.05 % half-spread per side, 300 round trips add about another 30 % of position size (+6 € per quarter).

**B. Break-even win rate.** Per trade, net win = TP − 0.5 % and net loss = SL + 0.5 %, ignoring spread. Break-even p = (SL + 0.5) / (TP + SL).
- TP 1.0 % / SL 0.8 % → win +0.5 %, loss −1.3 % → **p ≥ 72 %**
- TP 1.0 % / SL 1.0 % → **p ≥ 75 %**
- TP 2.0 % / SL 1.0 % → win +1.5 %, loss −1.5 % → **p ≥ 50 %**
- TP 3.0 % / SL 1.5 % → win +2.5 %, loss −2.0 % → **p ≥ 44 %**
- TP 4.0 % / SL 2.0 % → **p ≥ 42 %**

A 1 % TP floor gives away half of each winning trade to fees. Most trend and momentum systems win only 35–55 % of trades, so they need TP/SL distances that are large compared with the 0.5 % cost. A rough rule is TP ≥ 4× the round-trip cost, which is about 2 % or more here.

**C. Required gross edge.** Expected gross return per trade must exceed about 0.55–0.70 % (fees plus spread) to break even. Target: +10 % a year on 100 € (10 €).
- At 1,200 round trips a year with 20 € positions: required gross edge = 0.50 % + spread + (10 € / 1,200 / 20 €) ≈ 0.50 % + spread + 0.04 %
- At 120 round trips a year: ≈ 0.50 % + spread + 0.42 %

The fee hurdle is the same in both cases. Fewer trades are only better if each trade captures a larger move, which is plausible for 1h/4h trend signals and implausible for 5-minute EMA crosses.

**D. Sustainable frequency.** If a realistic gross edge per signal on 5m–1h crypto trend signals is in the 0.1–0.5 % range (assumption, no source), then at 0.5 % round trip **no** frequency is sustainable. Signals with an expected move of at least 1.5–3 % (4h/daily trend, breakouts) are needed. These naturally fire only about 2–15 times per pair per month.

### Gaps
- No study found that measures the gross edge per trade of EMA9/21 crossovers on 5m crypto candles after 2022. The 0.1–0.5 % assumption is unsourced.
- Full text of the 2024–2025 momentum and cost papers was not accessible, so exact cost assumptions (bps) could not be extracted.

---

## 2. Bitpanda Fusion fee schedule (2026), minimum orders, API limits, and comparison with other EU venues. Can maker orders cut costs?

### Takeaway
Fusion's entry tier, 0.25 % flat for maker and taker up to 100 k€ in 30 days, is mid-to-expensive for the EU. Because maker and taker pay the same rate, **limit orders do not lower fees on Fusion**; they only avoid the spread. The cheapest EUR spot entry tiers found are One Trading (0.10 % maker / 0.20 % taker) and Bybit EU (0.10 % / 0.25 %). Kraken's entry tier reportedly rose sharply in July 2026 (0.40 % / 0.80 %), but sources conflict on this.

### Cited Findings
**Bitpanda Fusion**
- 7 tiers by 30-day volume, one rate per tier (no maker/taker distinction): 0.25 % up to 100,000 €, down to 0.02 % above 250 M € — [Cryptoticker 2026](https://cryptoticker.io/en/bitpanda-fusion/reviews/)
- Fusion aggregates order books of 12+ exchanges in real time for best execution — [Bitpanda Support: Bitpanda Fusion](https://support.bitpanda.com/hc/en-us/articles/16663481714844-Bitpanda-Fusion) (search extract)
- "Platform fees get reduced by 20 % if paid using BEST" — search extract from [Bitpanda Support / Cryptowisser on Bitpanda Pro](https://www.cryptowisser.com/exchange/bitpanda-pro/). **Caution:** this probably describes the old Bitpanda Pro / Bitpanda Global Exchange, which is now One Trading. It is unconfirmed whether any BEST discount applies on Fusion in 2026.
- Minimum trade: users must "deposit a minimum of €25 and trade a minimum amount of €25" (stated in a promo-eligibility context). The minimum trade amount "varies by asset" and is shown in the UI — [Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/16663481714844-Bitpanda-Fusion) (search extract). This is consistent with the bot's measured BTC-EUR `minOrderAmount` of 25 €.
- Fusion API rate limits: global 1,000 requests/min, market data endpoints 240/min, create-order 300/min, enforced **per Bitpanda user account**, not per API key — [docs.fusion.bitpanda.com Rate Limits](https://docs.fusion.bitpanda.com/rate-limits-370893m0) (search extract). The "Get trading pairs" endpoint exposes min/max order size, tick size and price increment per pair — same source.
- Intermediate Fusion tier rates (between 0.25 % and 0.02 %) were not available in any accessible source.

**Comparison: entry-tier spot fees for EUR pairs (retail, < ~10 k volume)**

| Venue | Maker | Taker | Maker round trip | Taker round trip | Notes / date |
|---|---|---|---|---|---|
| Bitpanda Fusion | 0.25 % | 0.25 % | 0.50 % | 0.50 % | to 100 k€/30 d; [Cryptoticker 2026](https://cryptoticker.io/en/bitpanda-fusion/reviews/) |
| One Trading | 0.10 % | 0.20 % | 0.20 % | 0.40 % | 0–9,999 €; 10 k–99,999 €: 0.04 %/0.08 %; [onetrading.com/fees](https://www.onetrading.com/fees) (search extract, undated; the jump between tiers looks unusually steep, verify) |
| Bybit EU | 0.10 % | 0.25 % | 0.20 % | 0.50 % | Non-VIP, all spot pairs; [Bybit EU Help Center](https://www.bybit.eu/en-EU/help-center/article/Bybit-Spot-Fees-Explained) (search extract). Global Bybit lists crypto-fiat pairs at 0.15 %/0.20 %, [Bitdegree 2026](https://www.bitdegree.org/crypto/tutorials/bybit-fees). EU accounts moving to bybit.eu in 2026, [Freenance](https://freenance.io/comparisons/binance-vs-bybit-2026-comparison/) |
| Binance | 0.10 % | 0.10 % | 0.20 % | 0.20 % | standard spot, "as of June 10, 2026"; BNB discount available; [Coin Bureau](https://coinbureau.com/review/binance-vs-bybit). EUR-pair and EU-licensing status not verified |
| Coinbase Advanced (EU/UK) | 0.25 % | 0.50 % | 0.50 % | 1.00 % | entry tier "as of September 2026"; [Datawallet](https://www.datawallet.com/crypto/coinbase-fees) / [Coinbase blog](https://www.coinbase.com/blog/were-lowering-fees-for-many-active-traders-on-coinbase-advanced) (search extract; the tier table in the extract was partly inconsistent) |
| Bitstamp | 0.30 % | 0.40 % | 0.60 % | 0.80 % | < $10 k; 0.20 %/0.30 % from $10 k; undated; [CoinLaw](https://coinlaw.io/crypto-exchange-fees/) / [TokenTax](https://tokentax.co/blog/crypto-exchange-with-lowest-fees) |
| Kraken Pro (old) | 0.25 % | 0.40 % | 0.50 % | 0.80 % | $0–10 k; pre-July-2026 schedule; [Bitsgap 2026](https://bitsgap.com/blog/kraken-trading-fees-explained-what-it-costs), [CryptoRyancy](https://www.cryptoryancy.com/kraken-fees-complete-guide-2026/) |
| Kraken Pro (from 9 Jul 2026) | 0.40 % | 0.80 % | 0.80 % | 1.60 % | Tier 1; 0.30 %/0.60 % at $2.5 k; 0.22 %/0.38 % at $10 k; tiers now based on the best of spot volume, futures volume or assets on platform; [MEXC Learn](https://www.mexc.com/en-GB/learn/article/kraken-fees-explained-2026-the-9-july-2026-tier-change-that-doubled-kraken-entry-taker-rate-to-0-80-/1), [Kraken Blog](https://blog.kraken.com/product/pro/new-kraken-pro-fee-tiers), [Kraken Support "Cross-platform fee tier changes (July 2026)"](https://support.kraken.com/articles/cross-platform-fee-tier-changes) |

- **Conflict on Kraken:** another search extract listed a "Starter < $10,000: 0.09 % maker / 0.16 % taker" schedule "as of May 20, 2026" — [Kraken Support (search extract)](https://support.kraken.com/ca/articles/201893638-How-trading-fees-work-on-Kraken). That tier list was internally inconsistent (fees rising with volume), and a second search found no such tier. The July 2026 schedule (0.40 %/0.80 %) is more consistent across sources but could not be checked on kraken.com.

### Inferences
- **Maker on Fusion:** fees cannot be reduced, because maker = taker = 0.25 %. Posting a limit order at the best bid can still save the half-spread (about 0.01–0.1 % per side on BTC/ETH, more on SOL-EUR) and slippage. The cost is non-fill risk and missing fast entries. Because Fusion routes to aggregated external books, how resting limit orders behave on Fusion should be checked in the API docs (time-in-force, post-only availability unknown).
- **Venue choice matters more than parameter tuning.** On One Trading with maker orders, a round trip costs 0.20 % instead of 0.50 %. That moves the break-even win rate for TP 1 %/SL 1 % from 75 % to 60 %, and lets about 2.5× as many trades fit in the same cost budget. Bybit EU maker (0.10 %) is similar. One Trading is Vienna-based, which could matter for Austrian tax handling (see section 6), but this was not verified.
- **Volume tiers are out of reach.** At 300 round trips per 90 days with 20 € positions, 30-day volume is about 4,000 € (two sides). Fusion's next tier starts at 100 k€, roughly 25× higher. One Trading's second tier at 10 k€ would need about 50 € positions at the same frequency, but chasing volume to lower fees is self-defeating.

### Gaps
- Official Bitpanda Fusion fee page, Fusion tier 2–6 rates and per-pair `minOrderAmount` for ETH-EUR and SOL-EUR could not be read (egress blocked). Query `/v1/pairs` from the bot to get them.
- Whether a BEST-token discount exists on Fusion in 2026 is unconfirmed.
- Whether Fusion supports post-only / maker-only orders is unknown.
- Binance's EU/MiCA status and EUR-pair fees in 2026 were not verified.

---

## 3. Slippage and spreads for BTC/ETH/SOL EUR pairs

### Takeaway
On top-tier EUR venues, BTC-EUR and ETH-EUR spreads are now around or below 1–2 bps, which is negligible next to 50 bps of fees. EUR liquidity is concentrated on a few venues (Bitvavo, Kraken, Coinbase, Binance hold 85 %+), and the "venue gap" means execution on smaller EUR books can be noticeably worse. For 20 € orders, slippage is effectively spread-only. SOL-EUR spreads are likely wider, but no figure was found.

### Cited Findings
- Kaiko: top EUR venues trade "below 2 basis points across major EUR pairs", on par with USD venues. Bitvavo had the tightest average EUR spread at **0.981 bps** across a basket including BTC — [Kaiko: Bitvavo's position in EUR spot trading](https://www.kaiko.com/news/kaiko-research-highlights-bitvavos-position-in-eur-spot-trading); [Bitvavo: Kaiko report 2026](https://bitvavo.com/en/news/kaiko-report-2026)
- Kaiko: EUR market depth (1 % depth) "has surpassed USD equivalents for the first time" (as reported in 2026 coverage) — same sources; [CryptoBriefing](https://cryptobriefing.com/kaiko-eur-fiat-pairs-dominate-crypto-volumes/)
- BTC-EUR's share of global BTC-fiat volume rose from 3.6 % to nearly 10 % in 2024. Bitvavo, Kraken (~20 %), Coinbase (~13 %) and Binance (~8 %) held more than 85 % of EUR volume in Nov 2024 — [Kaiko](https://www.kaiko.com/news/kaiko-research-highlights-bitvavos-position-in-eur-spot-trading); [Finance-Loop](https://www.finance-loop.net/largest-euro-crypto-exchanges/)
- CryptoSlate describes a "venue gap" in European crypto: headline EUR volume and stablecoin growth do not translate into equal execution quality across venues — [CryptoSlate](https://cryptoslate.com/mica-euro-stablecoins-doubled-did-btc-eth-liquidity-in-europe-follow/) (headline only; full text blocked)

### Inferences
- At 20 € order size, market impact is zero, so cost = half-spread per side. For BTC-EUR/ETH-EUR on Fusion this is probably about 0.5–5 bps per side if Fusion's aggregation reaches top venues; SOL-EUR is likely several bps. That is 2–10 % of the fee cost: **fees, not spread, are the binding constraint**.
- The bot's backtest should model the real orderbook bid/ask from Fusion `/v1/orderbook`, not the mid, plus 0.25 % per side.
- Bitpanda Fusion and One Trading are not named among the top EUR-liquidity venues in the Kaiko data found.

### Gaps
- No Kaiko or other figure found for SOL-EUR spreads or for Bitpanda Fusion / One Trading spreads specifically.
- Kaiko report dates and exact methodology (time window) were not readable in full.

---

## 4. Asset selection: BTC/ETH/SOL only vs a broader universe; correlation and diversification; how many assets for 100 €?

### Takeaway
BTC, ETH and SOL move as one block, with 90-day correlations around 0.85–0.92, so holding two of them at once gives little diversification. Academic evidence says the larger momentum and anomaly returns come from small and illiquid coins, where costs and spreads eat them. A broader alt universe is therefore not a free lunch for a 0.5 %-round-trip retail bot. For 100 €, trading one or two of the most liquid assets (BTC, ETH) with one position at a time is the defensible choice.

### Cited Findings
- 90-day rolling correlations: BTC/ETH ≈ 0.90, BTC/SOL ≈ 0.92, ETH/SOL ≈ 0.90. As of 24 Apr 2026, BTC-SOL was 0.92 on 90 days and 0.83 on 1 year — [Sharpe.ai correlation matrix](https://www.sharpe.ai/learn/crypto-correlation-matrix); [AhaSignals BTC/SOL](https://ahasignals.com/crypto-correlation/btc-sol/); [RektCalc](https://rektcalc.com/crypto-correlation-matrix.html)
- DeFiLlama data reported record-high crypto correlations, with BTC-SOL reaching 0.99 — [CryptoPotato](https://cryptopotato.com/defillama-crypto-correlations-hit-record-highs-as-btc-sol-reaches-0-99/)
- Volatile crypto assets "cluster above 0.85 correlation with each other across 90-day windows" and trade as one block in most conditions — search extract from correlation-tool sources above ([Spark](https://www.spark.money/tools/crypto-portfolio-correlation-calculator)); CME comparison of SOL/BTC/ETH — [CME OpenMarkets 2025](https://www.cmegroup.com/openmarkets/economics/2025/Solana-vs-Bitcoin-vs-Ethereum-How-Do-They-Compare.html)
- Size and volume anomalies come from micro-cap coins "of negligible economic importance". Momentum persists in larger coins but "incurs substantial trading costs". Anomalous returns fall with size — [ScienceDirect: "Cryptocurrency anomalies and economic constraints" (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001509)
- Positive momentum findings may be explained by illiquid small caps. After realistic costs, many momentum portfolios lose significance — [Springer 2025](https://link.springer.com/article/10.1007/s11408-025-00474-9); [AUT/ResearchGate](https://www.researchgate.net/publication/377457967_Time-Series_and_Cross-Sectional_Momentum_in_the_Cryptocurrency_Market_A_Comprehensive_Analysis_under_Realistic_Assumptions)
- Cross-sectional crypto return factors (size, momentum, trend) exist in the literature — [Liu, Tsyvinski & Wu, NBER w25882](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf); [JFQA: A Trend Factor for the Cross Section of Cryptocurrency Returns](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178). These are typically weekly-rebalanced long-short portfolios across hundreds of coins, which a long-only spot bot with 2 slots cannot replicate.

### Inferences
- With ρ ≈ 0.9, two simultaneous long positions in BTC and SOL behave like about 1.05 independent bets: same direction, more fees. The 2-position limit mostly doubles exposure to one market factor.
- Cross-sectional momentum (pick the strongest coin, rotate weekly) is the academically supported way to use a broader universe. At 0.5 % per side, weekly rotation of the whole book costs up to about 52 % a year, so rotation must be infrequent (monthly) or rule-gated.
- Small caps on Fusion: higher volatility increases gross move per trade, which helps against fixed % fees, but EUR spreads on alts are wider and per-pair minimums may apply. There is no evidence that this is net positive for retail.
- For 100 €: trade 1–2 assets (BTC-EUR, ETH-EUR) with one position at a time, or use a simple time-series trend filter with full allocation instead of 2 × 20 € slots.

### Gaps
- No 2026 Kaiko spread data for EUR altcoin pairs.
- No source quantifying the diversification benefit of BTC/ETH/SOL during crashes, when correlations typically rise.

---

## 5. Is 100 € viable for an active bot? What capital level changes that?

### Takeaway
On Fusion, costs are purely percentage-based (no fixed per-trade fee found), so a larger account does **not** lower the 0.5 % hurdle until the 100 k€/30-day tier. 100 € is viable as a learning and validation account, not as an income source. The concrete blocker is the 25 € minimum order: 15–25 € positions are partly impossible, and two positions lock up 50 % or more of capital. Capital mainly matters for (a) clearing minimums with sensible sizing and (b) making absolute profits worth the effort.

### Cited Findings
- Minimum trade on Bitpanda/Fusion: 25 € (and asset-dependent) — [Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/16663481714844-Bitpanda-Fusion) (search extract); measured 25 € for BTC-EUR via `/v1/pairs` (task context)
- Fusion fee tier 1 runs up to 100,000 € 30-day volume — [Cryptoticker 2026](https://cryptoticker.io/en/bitpanda-fusion/reviews/)
- Kraken's July 2026 tiers can now be reached through assets on platform as well as volume — [Kraken Blog](https://blog.kraken.com/product/pro/new-kraken-pro-fee-tiers); [MEXC Learn](https://www.mexc.com/en-GB/learn/article/kraken-fees-explained-2026-the-9-july-2026-tier-change-that-doubled-kraken-entry-taker-rate-to-0-80-/1)
- Rate limits are per user account (1,000/min global), which is irrelevant at this scale — [Fusion docs](https://docs.fusion.bitpanda.com/rate-limits-370893m0)

### Inferences (own arithmetic)
- **Minimum-order math:** with min 25 €, a 15 € position is impossible and a 20 € position must be raised to 25 €. Two concurrent positions need 50 € or more. With the bot's 2 % risk rule (2 € risk on 100 €), a 25 € position allows a maximum stop distance of 8 %, so risk sizing is not the constraint; the minimum order is.
- **Absolute money:** a very good bot at +20 % a year after fees makes 20 € on 100 € before 27.5 % KESt (about 14.50 € net). Infrastructure (VPS, time, data) usually exceeds that, so 100 € can only be justified as a test budget.
- **Capital thresholds where things change** (using tier-1 0.25 %):
  - about 250–500 €: positions of 50–100 € clear all per-pair minimums with room for 2–3 positions and partial exits.
  - about 2,000–5,000 €: absolute returns start to exceed typical running costs. The % fee hurdle is unchanged.
  - Fusion tier 2 (> 100 k€/30 d): at 100 round trips a month (200 orders) this needs about 500 € average order size. Reaching it at 300 trades/90 d needs a capital base of several thousand € and is pointless unless the strategy is already profitable at 0.25 %.
- On venues with lower entry fees (One Trading, Bybit EU maker 0.10 %), the same 100 € has a 2.5× lower hurdle, which matters more than any capital increase on Fusion.

### Gaps
- Fusion deposit/withdrawal fees (SEPA) and any inactivity or fixed fees were not found. Assumed zero, unverified.
- Per-pair minimums for ETH-EUR and SOL-EUR are unknown (query `/v1/pairs`).

---

## 6. Austrian/EU tax aspects that affect bot design

### Takeaway
Since the 2022 eco-social tax reform, gains on crypto acquired after 28 Feb 2021 are taxed at a flat 27.5 % special rate. Each sell to EUR is a taxable realisation; crypto-to-crypto swaps are tax-neutral. Cost basis uses the moving average price (gleitender Durchschnittspreis). Bitpanda is "steuereinfach" and withholds KESt automatically, which removes most of the reporting burden of thousands of bot trades. A foreign venue (Kraken, Bybit, Binance) shifts AVCO calculation and E1kv reporting to the user. Losses offset only against other 27.5 % capital income.

### Cited Findings
- Crypto gains are taxed at the special rate of 27.5 % (KESt). Exchange into fiat triggers taxation — [BMF: Steuerliche Behandlung von Kryptowährungen](https://www.bmf.gv.at/themen/steuern/sparen-veranlagen/steuerliche-behandlung-von-kryptowaehrungen.html) (search extract); [Blockpit AT guide](https://blockpit.io/tax-guides/krypto-steuer-guide-osterreich/)
- Crypto-to-crypto exchange is not a disposal and is not taxed. Transaction costs are "steuerlich unbeachtlich" and the acquisition cost carries over to the received coin — [BMF](https://www.bmf.gv.at/themen/steuern/sparen-veranlagen/steuerliche-behandlung-von-kryptowaehrungen.html) (search extract)
- Neuvermögen = crypto acquired from 1 March 2021. The moving average price method is mandatory for crypto acquired after 31.12.2022 (Kryptowährungsverordnung) — [crypto-tax.at](https://www.crypto-tax.at/einkuenfteermittlung-bei-realisierten-wertsteigerung-aus-kryptowaehrungen-gleitender-durchschnittspreis-ab-01-01-2023/); [Blockpit FIFO/LIFO/AVCO AT](https://www.blockpit.io/de-at/steuer-guides/krypto-fifo-lifo-verbrauchsfolgeverfahren)
- Bitpanda is a "steuereinfache" platform: KESt is withheld automatically since 1.1.2024 — [crypto-tax.at: KESt bei Bitpanda ab 1.1.2024](https://www.crypto-tax.at/krypto-steuereinfach-in-osterreich-kest-bei-bitpanda/); [Blockpit Bitpanda guide](https://www.blockpit.io/de-at/steuer-guides/bitpanda-steuer-guide); [broker-test.at](https://www.broker-test.at/news/krypto-steuer-2024-automatischer-kest-abzug-kommt/)
- Losses can be offset against other income taxed at 27.5 % (shares, dividends, bond interest) — [capitalo.at](https://www.capitalo.at/krypto/ratgeber/krypto-steuern)
- 2026 guides also mention DAC8 reporting obligations for crypto providers — [finfo.at](https://www.finfo.at/steuern/krypto-steuer-oesterreich/); [coinsteuer.com](https://www.coinsteuer.com/steuer/oesterreich)

### Inferences
- **Tax drag on edge:** a profitable bot keeps 72.5 % of net gains, while losses are only useful against other 27.5 % income in the same year. Required gross edge per trade does not change, but the after-tax payoff of a marginal strategy shrinks further.
- **Platform choice versus reporting burden:** staying on Bitpanda (steuereinfach, automatic KESt) means 1,000+ trades a year need no manual AVCO bookkeeping. Moving to a cheaper foreign venue for lower fees means self-reporting in the tax return (E1kv), with AVCO computed across all trades. That is a real hidden cost of switching venues for a high-frequency bot, unless a tax tool (Blockpit, CoinTracking) is used.
- **AVCO versus bot P&L:** under AVCO, overlapping buys of the same coin are averaged. Per-trade P&L in `trades.db` (per-position FIFO-like) will differ from the tax figure, so the bot should not assume its own P&L equals the taxable gain.
- **Fees and tax:** the BMF statement that transaction costs are "steuerlich unbeachtlich" was given in the crypto-to-crypto context. Whether the 0.25 % fees reduce the taxable gain on crypto-to-EUR sales could not be confirmed.
- **Crypto-to-crypto neutrality** has no benefit for an EUR-quoted bot. Routing through BTC pairs to defer tax would add fees and complexity and is not advisable.

### Gaps
- BMF page full text not readable (egress blocked). Exact rules on deductibility of trading fees for crypto sales to EUR, and on loss carry-forward (believed not possible for private capital income), were not confirmed from a primary source.
- Whether One Trading (Vienna) is also "steuereinfach" with automatic KESt was not verified.
- DAC8 details (start date, what is reported) were not checked in a primary source.
