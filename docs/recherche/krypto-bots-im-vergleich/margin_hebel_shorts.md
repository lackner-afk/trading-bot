# Margin, Leverage and Short Selling for a Small Retail Crypto Bot (Austria/EU, status October 2026)

Research note. Scope: what changes when a 100 EUR long-only spot bot (Bitpanda Fusion, BTC/ETH/SOL-EUR, 0.25 % fee per order) adds margin/leverage/shorts. Research done 2026-10-06. Many primary sources (esma.europa.eu, bis.org, onetrading.com, bitpanda support/CDN, financemagnates, kraken blog) were blocked by the network egress proxy in this session, so several findings rely on search-engine summaries of those pages rather than full-text reads. That is flagged where relevant.

## 1. Does the short side add return in crypto trend-following/momentum, or mostly risk and cost?

### Takeaway
The academic evidence is mixed and leans against the short leg: several crypto time-series-momentum studies find long-only portfolios beat the market while short-only legs lose and long-short versions often fail to beat benchmarks. Other studies and practitioner frameworks report positive long-short results, usually with asymmetric (long-heavy) allocation. Shorts also face the biggest crypto tail risks: upward drift, squeezes and positive funding in bull phases.

### Cited Findings
- In a crypto time-series momentum study, "all long-only portfolios exhibit superior performance compared to the market portfolio in terms of Sharpe ratio and cumulative return, and even after accounting for transaction costs, most long-only portfolios outperform the market. In contrast, short-only portfolios yield unfavorable results." (summary of search result) — [AUT/ACFR: Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- Erasmus thesis on crypto TSMOM: all long-only strategies produced positive excess returns over benchmarks (significant for 85 % of long-only portfolios), but "none of the long-short time-series momentum strategies significantly outperformed any benchmarks, and in many cases the long-short portfolios even had negative excess returns." This is a student thesis, so weaker evidence. — [Erasmus School of Economics thesis (Wisselink)](https://thesis.eur.nl/pub/44390/Wisselink-NJ-483391-BA-thesis.pdf)
- Contradicting result: another paper reports that long-short portfolio strategies, "producing positive cumulative returns in most subsample periods, consistently outperform conservative long-only portfolio strategies in the cryptocurrency market". Results depend on sample period and design. — [Dynamic time series momentum of cryptocurrencies](https://community.portfolio123.com/uploads/short-url/amrMsuqIKzdHcHHyvMud4YNPwZB.pdf) (attribution to this exact paper comes from a search summary; not verified in full text)
- Liu & Tsyvinski (NBER) document strong time-series momentum in crypto returns. That is the basis for trend-following, but the effect is concentrated in the positive-return direction during boom periods. — [NBER w24877 Risks and Returns of Cryptocurrency](https://www.nber.org/system/files/working_papers/w24877/w24877.pdf)
- A 2026 arXiv framework (6h trend-following, "asymmetric long-short capital allocation") reports Sharpe 2.41 and max DD −12.7 %. This is a single in-sample backtest paper; the asymmetric allocation itself signals that the short side is treated as weaker. — [arXiv 2602.11708](https://arxiv.org/abs/2602.11708v1)
- During the 10 Oct 2025 crash, nearly 90 % of the >$19 bn liquidations were longs (BTC −14 %, ~1.6 m accounts wiped). The squeeze risk runs both ways: crowded, over-levered positioning gets punished in whichever direction the crowd is leaning. — [21Shares: Record crypto liquidations amid tariff shock](https://www.21shares.com//research/record-crypto-liquidations-amid-tariff-shock); [FTI Consulting](https://www.fticonsulting.com/zh-cn/china/insights/articles/crypto-crash-october-2025-leverage-met-liquidity)
- Futures carry/funding is usually positive (longs pay shorts) and spikes in booms. That makes shorts on perps slightly carry-positive most of the time, but they get squeezed in exactly those boom phases. — [BIS WP 1087 "Crypto carry" (Schmeling, Schrimpf, Todorov, 2023)](https://www.bis.org/publ/work1087.pdf)

### Inferences
- For a 5m EMA-crossover bot, the earlier 6x long/short backtest outperformance cannot be attributed to the short side without re-running it with realistic costs (0.25 % per side on Fusion spot, funding/financing, liquidation fees). The literature does not support assuming the short leg adds return by default.
- A reasonable test: run the backtester long-only vs long/short with identical cost models, and report the short-leg P&L separately (gross, after fees, after funding) over bull, bear and sideways sub-periods.

### Gaps
- I found no peer-reviewed study specifically on intraday (5m) EMA-crossover long/short vs long-only for BTC/ETH/SOL with realistic retail fees.
- Full texts of the AUT and portfolio123-hosted papers could not be fetched (egress blocked), so the exact sample periods are unknown.

## 2. Costs of leverage: funding and margin interest, liquidation mechanics, derivative vs spot fees

### Takeaway
Leverage carries a running cost that spot does not.
- **Perpetuals:** funding sits at a baseline of about 0.01 % per 8h (≈10.95 % APR, paid by longs) and can spike far higher in booms.
- **Bitpanda Margin Trading:** much more expensive. Financing is 0.18 %/day (≈66 % p.a. on position size) for the first 60 days, plus a 0.3 % closing fee and a 1 % liquidation fee.
- **Perp trading fees** (e.g. Coinbase EU "as low as 0.02 %") can be far below the 0.25 % spot fee. Funding, liquidation risk and spread then become the dominant costs.

### Cited Findings
- Binance funding formula: Funding Rate = Average Premium Index + clamp(Interest Rate − Premium Index, −0.05 %, 0.05 %), with the Interest Rate fixed at 0.01 % per 8h. 0.01 % × 3 × 365 ≈ 10.95 % APR. The funding average "consistently hovers around a baseline of 0.01 %/8-hour". — [BitMEX Q3 2025 derivatives report "The Anchor and the Ceiling"](https://blog.bitmex.com/2025q3-derivatives-report) (via search summary; full text blocked); [tradingstrategy.ai glossary](https://tradingstrategy.ai/glossary/funding-rate)
- A Coincub analysis headline: "Perpetual Futures Funding Rates: 78 % of a Quarter at 0.01 %". Funding sits at the baseline most of the time. — [Coincub](https://coincub.com/?p=14476) (headline only; not read in full)
- BIS: crypto futures carry "can become very large (up to 60 % p.a.) and varies strongly over time", driven by trend-chasing retail seeking leveraged upside plus scarce arbitrage capital. Arbitrage is risky "due to spikes in margins and liquidations amid drawdowns". Data period is pre-2023. — [BIS WP 1087 (April 2023)](https://www.bis.org/publ/work1087.pdf)
- A search snippet claimed BTC funding "averaged +0.51 % (70.2 % APR) in early 2026". I could not trace it to a primary source and it is inconsistent with the baseline evidence above, so treat it as **unreliable**. — [CoinGlass BTC Funding Rate page (search result)](https://www.coinglass.com/FundingRate/BTC)
- **Bitpanda Margin Trading costs:** open 0 %, closing fee 0.3 % of position size, liquidation fee 1 %. Funding fee for positions opened after 8 July 2026 is tiered by holding time: days 1–60 0.18 %/day (0.03 % every 4h), days 61–100 0.12 %/day, days 101–180 0.06 %/day, day 181+ 0.0312 %/day. Positions opened before 8 July 2026 pay a flat 0.03 % per 4h. — [Bitpanda Support: Margin Trading for Cryptocurrencies](https://support.bitpanda.com/hc/en-us/articles/21417526386588-Bitpanda-Margin-Trading) (via search summary; page blocked for direct fetch)
  - Conflict: another source describes it as "a 0.18 % funding fee every 4 hours" — [trendingtopics.eu](https://www.trendingtopics.eu/bitpanda-margin-trading-start/). The support-page figure (0.03 %/4h = 0.18 %/day) is more likely correct.
- Coinbase EU perps: fees "as low as 0.02 % per contract", up to 10x on select crypto contracts. — [Cointelegraph](https://cointelegraph.com/news/coinbase-perpetual-futures-contracts-europe-esma)
- One Trading's fee page says futures fees follow a 30-day-volume maker/taker schedule. Exact rates could not be retrieved. — [One Trading fees](https://www.onetrading.com/fees) (blocked)
- Kraken EU lets clients use BTC, ETH and stablecoins as collateral for futures, up to 10x, across 150+ perpetual markets. — [crypto.news](https://crypto.news/kraken-launches-crypto-collateral-futures-eu-2025/)
- Liquidation dynamics: a peer-reviewed BitMEX study finds daily forced liquidations of 3.51 % (longs) and 1.89 % (shorts) of outstanding futures, with leverage and BTC volatility raising liquidation risk. — [ScienceDirect: Hedging with automatic liquidation and leverage selection on bitcoin futures](https://www.sciencedirect.com/science/article/pii/S0377221722005975)

### Inferences
- **Bitpanda margin cost math:** a trade held 1 day costs about 0.18 % financing plus 0.3 % close, ≈0.48 %, on top of spread. That is close to the current spot round-trip of 0.5 % (2 × 0.25 %). For multi-day trend holds, financing dominates: 10 days ≈1.8 % of notional, so at 5x that is ≈9 % of the margin used.
- **Perp cost math:** at the baseline, a long pays ≈0.03 %/day of notional (≈0.3 % over 10 days). A short receives it. In boom phases (BIS: up to 60 % p.a.) the cost to longs rises and shorts earn carry but face squeeze risk.
- **Liquidation price for an isolated position:** roughly entry × (1 − 1/L + maintenance margin) for longs, mirrored for shorts. At 10x it sits ~9–10 % away; at 2x ~45–50 % away. On 5m crypto bars, the 10 Oct 2025 crash (−14 % BTC intraday) would have liquidated any 10x long whose stop did not fill.
- The bot's 0.06 % fee assumption is roughly perp-taker-like. It does not represent Bitpanda Fusion spot (0.25 %) or Bitpanda Margin (0.3 % close + financing).

### Gaps
- I could not retrieve verified 2024–2026 average annualized funding for BTC/ETH from CoinGlass or Kaiko (pages blocked or not returned).
- One Trading perp fee tiers and funding interval are unverified.
- Bitpanda Margin liquidation threshold (margin level %) and spread are not confirmed.

## 3. Evidence on retail leverage outcomes (losing accounts, liquidations)

### Takeaway
Retail leverage outcomes are poor across every source found:
- **EU CFDs:** 74–89 % of retail accounts lose money, with average losses of €1,600–29,000 per client (ESMA, 2018).
- **Exchange data:** most retail perp traders use >20x leverage, and simulations show near-certain liquidation at very high leverage.
- **2025:** over $150 bn was force-liquidated, including the record ~$19 bn on 10 Oct 2025.

### Cited Findings
- NCA analyses across EU jurisdictions found 74–89 % of retail CFD accounts typically lose money, with average losses per client of €1,600 to €29,000. This was the basis for ESMA's 2018 measures and the mandatory risk warning showing the provider's own loss percentage. — [ESMA press release 2018 (agree to restrict CFDs)](https://www.esma.europa.eu/node/84933); [NBS mirror](https://nbs.sk/en/news/esma-agrees-to-prohibit-binary-options-and-restrict-cfds-to-protect-retail-investors/)
- A Binance study cited in an NYU Stern paper (Duron-Carielo) finds 79 % of Binance customers traded perps with >20x leverage (2020 data). In a liquidation simulation at 75x, 97.30 % of trades were liquidated, after about 30 minutes on average. — [NYU Stern: "Is There a Future in Perpetual Futures?"](https://www.stern.nyu.edu/sites/default/files/assets/documents/Duron-Carielo_Is%20There%20A%20Future%20In%20Perpetual%20Futures.pdf) (via search summary; PDF fetch blocked)
- BIS Bulletin 69: using app data for 95 countries (Aug 2015–Dec 2022), an estimated ~73–81 % of retail app users would have lost money on bitcoin, with an average loss of $431 (47.89 % of $900 invested) as of 15 Dec 2022. Downloads clustered at high prices. This covers spot, not leverage, so it is a baseline for retail timing behaviour. — [BIS Bulletin 69](https://www.bis.org/publ/bisbull69.pdf); [Bloomberg coverage](https://www.bloomberg.com/news/articles/2022-11-16/vast-majority-of-retail-investors-in-bitcoin-lost-money-bis-says)
- 2025 full year: more than $154 bn in forced liquidations across perp markets (≈$400–500 m per day). BTC/ETH estimated leverage on major CEXs frequently surpassed 10x, with "a meaningful portion of retail" at 50–100x. This is a secondary/industry source. — [BeInCrypto](https://beincrypto.com/crypto-futures-trading-mistakes-2025/)
- 10 Oct 2025: >$19 bn liquidated in about a day (CoinGlass data), BTC $122k→$105k, >1.6 m accounts wiped, ~90 % of liquidations were longs. BTC futures open interest was $45.3 bn beforehand. — [21Shares](https://www.21shares.com//research/record-crypto-liquidations-amid-tariff-shock); [HackerNoon](https://hackernoon.com/why-a-14percent-bitcoin-drop-wiped-out-16-million-accounts)

### Inferences
- The evidence concerns discretionary retail traders, not rule-based bots with stops. A bot with a hard 2 % risk-per-trade does not automatically inherit these loss rates. It does share the cost drag, the gap and liquidation risk, and the crowd-positioning risk.
- Liquidation statistics (CoinGlass-style) are aggregate notional figures, not per-account loss rates. They show tail frequency, not the share of losing accounts.

### Gaps
- I found no EU-regulated crypto-perp venue (One Trading, Kraken EU, Coinbase EU) publishing a "% of retail accounts losing money" figure in this session. If their products count as CFD-like, such a warning would be mandatory. Worth checking on their websites manually.
- No academic paper on retail outcomes on EU-regulated crypto perps was found (the products are too new).

## 4. Regulation for an Austrian retail trader in 2026; which EU venues offer margin/perps

### Takeaway
- **Two regimes apply:** derivatives (perps, CFDs) fall under MiFID II, and the venue needs MiFID authorisation. Spot crypto services fall under MiCA, whose transitional period ended 1 July 2026.
- **ESMA, 24 Feb 2026:** perpetual futures for retail are "likely" in scope of the CFD product intervention measures. That means 2:1 maximum leverage on crypto, 50 % margin close-out and negative balance protection. Austria already has permanent national CFD measures (FMA-PIV, 2019).
- **Venues:** One Trading, Kraken EU and Coinbase EU launched EU-regulated perps with up to 10x, so their retail leverage is now under regulatory pressure. Bitpanda offers spot-based Margin Trading up to 10x under its MiCAR licence, currently long-only (shorts announced as "planned"). Bitpanda Fusion itself has no margin. Bybit EU (Vienna, FMA) offers spot only.

### Cited Findings
- ESMA public statement, 24 Feb 2026 (ESMA35-243228190-8024): the commercial name "perpetual futures" is irrelevant. A derivative giving leveraged exposure that is not settled exclusively physically "would likely fall in scope of the product intervention measures on CFDs". Venue trading, a funding-rate mechanism or voluntary negative balance protection do not change this. Implication: 2:1 cap for retail on crypto underlyings. — [ESMA statement PDF](https://www.esma.europa.eu/sites/default/files/2026-02/ESMA35-243228190-8024_-_Public_statement_on_derivatives_in_scope_of_the_CFD_product_intervention_measures.pdf) (blocked; content via search summaries); [ESMA news item](https://www.esma.europa.eu/press-news/esma-news/esma-reminds-firms-their-obligations-under-cfd-product-intervention-measures); [TradeInformer](https://tradeinformer.com/regulations/esma-perpetual-futures-cfd-regulation-eu-2025)
- CFD measures (originally ESMA 2018, now national): leverage limits from 30:1 down to 2:1 for cryptocurrencies, margin close-out at 50 % of minimum required margin per account, negative balance protection per account, and a standardised risk warning. — [ESMA final measures](https://www.esma.europa.eu/node/85445); [FinanceFeeds](https://financefeeds.com/esma-bans-offering-binary-options-retail-investors-introduces-restrictions-cfds/)
- Austria: the FMA Product Intervention Regulation (FMA-PIV), published in BGBl. on 9 May 2019, is a permanent national restriction on CFD marketing, distribution and sale to retail clients in or from Austria. It includes virtual currencies, with leverage caps of 30:1 to 2:1 depending on the underlying. — [FMA](https://www.fma.gv.at/en/?p=22133); [RIS: FMA-PIV](https://www.ris.bka.gv.at/Dokumente/Bundesnormen/NOR40264128/NOR40264128.html); [gesetzefinden.at FMA-PIV](https://gesetzefinden.at/bundesrecht/verordnungen/fma-piv)
- MiCA: the transitional period ended across the EU on 1 July 2026. Since then, providing crypto-asset services to EU clients without a MiCA licence breaches EU law. — [ESMA statement on end of MiCA transitional periods (April 2026)](https://www.esma.europa.eu/sites/default/files/2026-04/ESMA75-113276571-1679_Statement_on_the_end_of_transitional_periods_under_MiCA.pdf) (via search summary)
- Perps and leverage for EU retail require both a MiCA CASP licence and a MiFID II authorisation (secondary source). Bybit EU, headquartered in Vienna and supervised by the FMA, has MiCA approval and offers spot but not perps. — [Finance Magnates: Europe's Crypto Market After July 1](https://www.financemagnates.com/cryptocurrency/regulation/europes-crypto-market-after-july-1-who-stays-who-leaves-and-what-changes-under-mica/) (via search summary)
- **One Trading** (ex-Bitpanda Pro, Amsterdam): Dutch AFM OTF licence (July 2024). It launched the EU's first MiFID II-regulated, cash-settled crypto perps (BTC/EUR, ETH/EUR) in 2025, then expanded to retail in Germany, the Netherlands and Austria. Leverage is up to 10x for eligible customers. In January 2026 the Dutch regulator backed its 24/7 equity perpetuals. — [The TRADE](https://www.thetradenews.com/one-trading-expands-retail-access-for-crypto-perpetual-futures-venue/); [FOW](https://www.fow.com/insights/one-trading-live-with-europes-first-regulated-perpetual-futures); [crypto.news](https://crypto.news/thiel-backed-one-trading-secures-license-from-dutch-regulator-for-perpetual-futures/)
- **Kraken:** MiCA authorisation via Payward Europe Solutions (Central Bank of Ireland, June 2025). Futures run through a separate MiFID II entity supervised in Cyprus, with up to 10x and 150+ perps for eligible clients. After the ESMA statement, observers expect venues to geo-fence EU retail or route users through "professional client" onboarding. — [Tangem: Is Kraken MiCA licensed](https://tangem.com/en/learning-hub/post/is-kraken-mica-licensed/); [crypto.news](https://crypto.news/kraken-launches-crypto-collateral-futures-eu-2025/); [Finance Magnates "10x Down to 2x"](https://www.financemagnates.com/forex/10x-down-to-2x-has-europe-killed-crypto-perps-even-before-it-started/) (via search summary)
- **Coinbase:** launched perps (perpetual-style futures with 5-year expiry, plus dated futures) for Coinbase Advanced users in 26 European countries on 9 March 2026, via its MiFID entity in Cyprus (CySEC). Up to 10x on select crypto contracts, up to 5x on others, fees from 0.02 %. This came about two weeks after the ESMA statement. — [Cointelegraph](https://cointelegraph.com/news/coinbase-perpetual-futures-contracts-europe-esma)
- **Bitpanda:** Bitpanda Margin Trading (successor to "Bitpanda Leverage") launched around August 2025 and is described as the first in Europe under a MiCAR licence. It covers 100+ crypto assets at 2x/3x/5x/10x, is "spot-based … not derivatives or futures", and has TP/SL orders. Short selling was "not yet available … planned" per a search summary of Bitpanda's pages. Bitpanda Fusion does not offer margin. In July 2026 Bitpanda added margin on 875+ stocks/ETFs up to 20x. — [Bitpanda blog](https://blog.bitpanda.com/en/bitpanda-margin-trading-smarter-way-trade-crypto-10x-leverage); [Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/21417526386588-Bitpanda-Margin-Trading); [Bitpanda Fusion support](https://support.bitpanda.com/hc/en-us/articles/16663481714844-Bitpanda-Fusion); [CryptoTicker stocks margin](https://cryptoticker.io/en/bitpanda-margin-trading-real-stocks-etfs-20x-launch/) (all via search summaries)
- Coinbase International Exchange / Deribit retail availability for Austria: not verified in this session. A "Coinbase Perpetuals Restricted Countries List 2026" page exists. — [CoinPerps](https://www.coinperps.com/learn/coinbase-perpetuals-restricted-countries)

### Inferences
- For an Austrian retail client, the realistic regulated routes are:
  - One Trading perps (BTC/ETH-EUR only; leverage may be cut to 2x for retail after ESMA)
  - Kraken EU or Coinbase EU perps (same caveat)
  - Bitpanda Margin Trading (spot-based, 10x, long-only so far, expensive financing, API access unconfirmed)
- Offshore perps (Bybit global, Binance) are not legitimately available to EU retail after 1 July 2026.
- "Professional client" status to escape the 2:1 cap requires meeting MiFID opt-up criteria (portfolio size, trading frequency, experience). A 100 € account would not qualify. *(MiFID criteria from general knowledge, not sourced here.)*
- The bot's 50x cap is far above anything legally reachable for retail in the EU (2:1 under CFD rules; 10x on Bitpanda Margin). It should be lowered in config to match the venue.

### Gaps
- I could not confirm whether One Trading, Kraken EU or Coinbase EU actually cut retail crypto-perp leverage to 2x after February 2026, or whether NCAs (AFM, CySEC, FMA) took enforcement action.
- Bitpanda Margin Trading short-selling launch status as of October 2026, and whether it is exposed via API (Fusion or public API), are unconfirmed.
- I did not find any One Trading roadmap for spot margin. Deribit's EU retail status is unverified.

## 5. Position sizing with leverage: volatility targeting, Kelly, risk-per-trade vs leverage number

### Takeaway
With a stop-loss that fills, the euro risk of a trade is position size × stop distance; leverage only determines how much margin is tied up. Risk-per-trade (and volatility-scaled sizing) is the real risk control. Leverage matters through gap risk, where the liquidation price can be hit before the stop executes, and through financing cost. Volatility targeting improves crypto trend-following Sharpe ratios in the studies found.

### Cited Findings
- Volatility targeting is effective at controlling risk, and trend-following has performed well in crypto. Shorter lookbacks give higher Sharpe but transaction costs erode profits through frequent rebalancing. — [Monash Centre for Financial Studies: Trend-following strategies for crypto investors](https://www.monash.edu/business/mcfs/our-research/all-projects/investment-strategy/trend-following-strategies-for-crypto-investors)
- An intraday BTC trend portfolio with exposure scaled to a 20 % annualized volatility target had Sharpe ≈1.6, vs just below 0.8 for a vol-targeted long-only BTC portfolio (practitioner research). — [Concretum Group](https://concretumgroup.com/seasonality-in-bitcoin-intraday-trend-trading/)
- 75x leverage simulation: 97.3 % of trades were liquidated within about 30 minutes on average. High nominal leverage with the stop beyond the liquidation price effectively turns every adverse move into a total loss of margin. — [NYU Stern, Duron-Carielo](https://www.stern.nyu.edu/sites/default/files/assets/documents/Duron-Carielo_Is%20There%20A%20Future%20In%20Perpetual%20Futures.pdf)
- Leverage and volatility jointly determine liquidation risk on bitcoin futures. Optimal leverage for hedgers is low once liquidation risk is modelled. — [ScienceDirect (European Journal of Operational Research 2022)](https://www.sciencedirect.com/science/article/pii/S0377221722005975)

### Inferences
- **Worked example (100 € equity):** 2 % risk = 2 €. With an ATR stop 2 % away, notional is 100 € (1x); with the stop 0.5 % away, notional is 400 € (4x). The leverage number follows from risk and stop distance and should never be chosen first.
- **Constraint to enforce:** the liquidation price must sit well beyond the stop. Example rule: distance to liquidation ≥ 2–3 × stop distance, plus a buffer for gap moves like 10 Oct 2025 (−14 % intraday).
- **Kelly:** full Kelly on noisy backtest edge estimates is known to overbet. The bot already uses half-Kelly; with leverage, quarter-Kelly or a cap at the risk-per-trade limit is more prudent. *(This is standard practice, not sourced in this session.)*
- **Fees and minimums at 100 €:** fixed percentages (0.25–0.3 %) plus financing are a large share of a 2 € risk budget. A 0.5 % round-trip on 400 € notional is 2 €, the whole risk budget. Leverage multiplies fees in euro terms because fees scale with notional.

### Gaps
- No crypto-specific study found comparing fixed-leverage vs vol-targeted sizing on 5m strategies with realistic EU retail fees.

## 6. How the bot architecture should prepare

### Takeaway
Treat margin as a different execution product with its own cost and risk model, not as a "leverage" parameter on spot. This means:
- an explicit shorts-enabled flag per venue
- funding/financing accrual in backtest and paper mode
- liquidation-price computation and checks before entry
- isolated margin by default
- a leverage cap matched to the venue and regulator (2x for CFD-like perps, ≤10x on Bitpanda Margin)

### Cited Findings
- Venue-specific cost schedules differ sharply and need separate cost models:
  - Bitpanda Margin: 0.18 %/day tiered financing, 0.3 % close, 1 % liquidation fee — [Bitpanda Support](https://support.bitpanda.com/hc/en-us/articles/21417526386588-Bitpanda-Margin-Trading)
  - Perps: funding baseline 0.01 %/8h with a clamp of ±0.05 % — [BitMEX research](https://blog.bitmex.com/2025q3-derivatives-report)
  - Coinbase EU perps: fees from 0.02 % — [Cointelegraph](https://cointelegraph.com/news/coinbase-perpetual-futures-contracts-europe-esma)
- ESMA CFD measures, if applied to the venue, impose a 50 % margin close-out per account and negative balance protection. The broker closes positions before the bot's own logic would. — [ESMA final measures](https://www.esma.europa.eu/node/85445)
- Bitpanda Margin aggregates all trades on the same asset into one position (relevant for position bookkeeping) and supports TP/SL orders. — [Bitpanda Support (search summary)](https://support.bitpanda.com/hc/en-us/articles/21417526386588-Bitpanda-Margin-Trading); [Bitpanda blog TP/SL](https://blog.bitpanda.com/en/take-profit-stop-loss-now-bitpanda-margin-trading)
- Cross-collateral (BTC/ETH/stablecoins) on Kraken EU futures means collateral value moves with the market. — [crypto.news](https://crypto.news/kraken-launches-crypto-collateral-futures-eu-2025/)

### Inferences (design recommendations derived from findings)
- **Config:** add `exchange_rules.allow_shorts`, `exchange_rules.max_leverage` (venue- and regulation-specific: 1 for Fusion spot, ≤2 for retail CFD-like perps, ≤10 for Bitpanda Margin), `margin.mode: isolated`, and a per-venue `financing` model (per-interval rate, interval, tiering by holding days, closing fee, liquidation fee).
- **Backtester and paper engine:**
  - accrue funding/financing per bar on notional, with sign by side and venue
  - charge the liquidation fee
  - simulate the liquidation price with the venue's maintenance margin
  - trigger liquidation on bar extremes (high/low) before the stop, checking whether intrabar gaps exceed the stop
- **Pre-trade checks:** reject if liquidation distance < k × stop distance. Reject if expected holding cost (financing × expected bars held) plus fees exceeds a share of the expected edge.
- **Reporting:** separate long and short leg P&L, funding P&L, fee P&L and liquidations in backtest output, so the earlier "6x shorts look great" effect can be decomposed.
- **Keep spot as default.** Add margin as a separate order-engine class behind a venue adapter. Do not raise the 50x cap; lower it to venue reality.
- **Validation:** paper-trade margin with realistic costs for a meaningful period before any live use (consistent with the project's live-mode guards).

### Gaps
- Whether Bitpanda Margin Trading or One Trading perps are reachable via the API the bot uses (Fusion / CCXT `onetrading`) was not verified. This determines feasibility.
- Exact maintenance-margin and liquidation formulas for One Trading, Bitpanda Margin and Kraken EU were not retrieved.
