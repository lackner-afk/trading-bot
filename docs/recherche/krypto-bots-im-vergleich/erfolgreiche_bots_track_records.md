# Crypto trading bots and systematic crypto strategies with verifiably positive track records: what works, what is only claimed, and what they share

Evidence grades used throughout:
- **[LIVE/AUDITED]**: real-money results from an index, fund or protocol, published by a third party or on-chain
- **[ACADEMIC-BT]**: peer-reviewed or working-paper backtest, sometimes with out-of-sample or data-snooping controls. Still a backtest
- **[BACKTEST-ONLY]**: single-author or preprint backtest with no independent replication
- **[MARKETING/UNVERIFIED]**: claims from vendors, platforms, blogs or aggregators with no checkable method

Method note: in this environment WebFetch was blocked for every domain tried (arxiv.org, bis.org, abarbon.com, ar5iv). All findings below therefore come from search-result abstracts and snippets, not from reading the full papers. Numbers marked with "(snippet)" should be checked against the primary PDF before anyone quotes them externally.

---

## Q1: Which strategy families have independently verifiable positive results in crypto (trend following, cross-sectional momentum, DCA, grid, market making, funding/basis arbitrage)?

### Takeaway
Only two families have robust, multi-source evidence. The first is **slow trend following and time-series momentum on daily or multi-day bars**, supported by academic backtests that hold up after costs and data-snooping controls. The second is **cash-and-carry / funding-rate basis arbitrage**, backed by academic work plus a large live on-chain track record (Ethena). Its edge has been shrinking sharply since 2024. Cross-sectional momentum is strong in academic data but needs long/short trading across many coins. For grid bots, DCA and retail market making I found only marketing claims or no independent evidence.

### Cited Findings

**Trend following / time-series momentum (daily or slower)**
- [ACADEMIC-BT] Zarattini, Pagani & Barbon, "Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins" (SSRN, April 2025). The data is survivorship-bias-free and covers all coins traded since 2015. The model is an ensemble of Donchian-channel trend models with different lookbacks, plus volatility-based position sizing. On a rotational top-20 liquid-coin portfolio it reached a **Sharpe above 1.5 and annualized alpha of 10.8% vs BTC, net of fees**. The paper also analyses transaction costs and proposes a portfolio technique to reduce them. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907); [ResearchGate](https://www.researchgate.net/publication/391508704_Catching_Crypto_Trends_A_Tactical_Approach_for_Bitcoin_and_Altcoins)
- [ACADEMIC-BT, pre-2020 data, mark as older] "A Decade of Evidence of Trend Following Investing in Cryptocurrencies" (Rozario/Holt et al., arXiv 2009.12155, 2020). It reports **walk-forward annualized returns of 255%**. Best full-period Sharpe ratios for moving-average rules were about **1.09 (SMA), 1.35 (EMA), 1.32 (DEMA)**. EMA and DEMA did poorly in the 2017–18 and 2018–19 slices. The authors find crypto behaves like commodities in risk-adjusted terms and diversifies equity bear markets (snippet). — [arXiv](https://arxiv.org/abs/2009.12155); [Semantic Scholar](https://www.semanticscholar.org/paper/A-Decade-of-Evidence-of-Trend-Following-Investing-Rozario-Holt/c7a18dc98f6cba8ab6dda9aa21a6b10024135f6d)
- [ACADEMIC-BT] Le & Ruthbah (Monash Centre for Financial Studies), "Trend-following Strategies for Crypto Investors": shorter lookbacks give higher Sharpe ratios, but **transaction costs from frequent rebalancing can erode the profits**. — [Monash](https://www.monash.edu/business/monash-centre-for-financial-studies/our-research/all-projects/investment-strategy/trend-following-strategies-for-crypto-investors); [PDF](https://www.monash.edu/__data/assets/pdf_file/0011/3744821/Trend-following-Strategies-for-Crypto-Investors.pdf)
- [BACKTEST-ONLY, preprint] "AdaptiveTrend" (arXiv 2602.11708, Feb 2026). Trend following on **6-hour bars**, monthly portfolio rebalancing, asymmetric long/short across 150+ pairs. Out-of-sample 2022–2024: **Sharpe 2.41, max drawdown −12.7%, Calmar 3.18**. It reportedly beats a plain TSMOM benchmark. Single unreplicated preprint, so the very high Sharpe should be treated with suspicion. — [arXiv](https://arxiv.org/html/2602.11708v1)
- [ACADEMIC-BT] A University of Twente study of currencies including crypto found that **cross-sectional** momentum suits crypto better than time-series momentum, while time-series momentum works best for fiat currencies. — [University of Twente](https://research.utwente.nl/en/publications/momentum-and-trend-following-trading-strategies-for-currencies-re/)

**Technical rules in general (with data-snooping controls)**
- [ACADEMIC-BT] Hudson & Urquhart, "Technical trading and cryptocurrencies" (Annals of Operations Research 297, 2021). They test about 15,000 technical rules from 5 classes. After multiple-hypothesis (data-snooping) corrections, a large share still shows significant returns, and breakeven transaction costs are well above typical crypto costs. However, **Bitcoin showed no predictability in the out-of-sample period**; predictability remained only in other coins. — [IDEAS/RePEc](https://ideas.repec.org/a/spr/annopr/v297y2021i1d10.1007_s10479-019-03357-1.html); [Springer PDF](https://link.springer.com/content/pdf/10.1007/s10479-019-03357-1.pdf)
- [ACADEMIC-BT] "Technical analysis in cryptocurrency markets: Do transaction costs and bubbles matter?" (J. Int. Financial Markets, Institutions & Money, 2022). The number of profitable rules falls after costs. **Costs change profitability dramatically at 1-minute frequency but not much at daily frequency.** — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042443122000816)
- [ACADEMIC-BT] "Are simple technical trading rules profitable in bitcoin markets?" (Int. Review of Economics & Finance, 2024). Tests 75,360 rules in 6 classes at daily and intraday frequency, with realistic investor behaviour and transaction costs (snippet only; I could not extract the headline result). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1059056024003010)
- [Weak: bachelor/master thesis] Ratia (Theseus, 2023). The selected technical-analysis methods were mostly unprofitable after trade fees, **even the best (RSI)**. — [Theseus PDF](https://www.theseus.fi/bitstream/handle/10024/802533/Ratia_Kristian.pdf)

**Cross-sectional momentum**
- [ACADEMIC-BT, peer-reviewed] Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency" (NBER WP 25882; Journal of Finance). Three factors (market, size, momentum) explain cross-sectional crypto returns. A long/short momentum portfolio earns about **3% excess return per week**: 2.7% at 1-week, 3.3% at 2-week, 4.1% at 3-week and 2.5% at 4-week lookbacks. The effect sits among large coins: **4.2%/week above median size vs an insignificant 0.6%/week below median.** — [NBER PDF](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf); [Yale](https://economics.yale.edu/research/common-risk-factors-cryptocurrency)

**Funding-rate / basis (cash-and-carry) arbitrage**
- [ACADEMIC, data-based] Schmeling, Schrimpf & Todorov, "Crypto Carry" (BIS WP 1087, April 2023, revised Oct 2025; also Management Science). Carry, meaning long spot and short futures, **sometimes exceeds 40% p.a.** and varies a lot over time. Two forces drive it: leveraged demand from small trend-chasing investors, and arbitrage capital limited by regulatory and margin frictions. — [BIS](https://www.bis.org/publications/working-paper-1087-crypto-carry); [Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.2024.05069)
- [ACADEMIC, attribution uncertain] One search snippet reports that crypto carry had an **annualized Sharpe of 6.45 over 2020–2025, falling to 4.06 from 2024 onward and turning negative in 2025**. The snippet credits only "recent research". It appeared next to arXiv 2510.14435, "Cryptocurrency as an Investable Asset Class: Coming of Age", and arXiv 2605.29309, but I could not confirm which paper it comes from. — [arXiv 2510.14435](https://arxiv.org/pdf/2510.14435); [arXiv 2605.29309](https://arxiv.org/pdf/2605.29309)
- [LIVE, on-chain protocol; numbers via secondary sites] Ethena sUSDe (a delta-neutral funding-rate strategy run at scale). Average APY was about **18% in 2024**, ranging from a 55.9% peak in March 2024 to about 4.3% in August 2024. In 2025 it was about **4–15%**. Funding averaged about 11% APY over 2023–2025, but ranged from **−6% (late-2022 bear) to +75% (early-2024 bull)**. The trailing 90-day average was 11.8% as of April 2026. These figures come from secondary explainer sites quoting Ethena's dashboard, not from an audit. — [eco.com](https://eco.com/support/en/articles/15254002-ethena-usde-and-susde-2026-delta-neutral-yield); [Stablecoin Insider Q1 2026](https://stablecoininsider.org/ethena-usde-q1-2026-report/); [Messari](https://messari.io/project/ethena)

**Grid bots**
- [MARKETING/UNVERIFIED] Pionex-adjacent sources claim "average annual returns of 15–50%" and "8–12% monthly in ranging markets". They also give single-user anecdotes: 11.3% over 156 cycles on BTC/USDT, and 147% over 479 days on ETH/USDT. They add that grid bots **underperform in trending phases**. None of it is audited, and the sources are platform blogs and affiliate reviews. — [Pionex blog](https://www.pionex.com/blog/15-reddit-questions-about-crypto-trading-bots-and-pionex-answered-with-real-data/); [BotVerdict](https://botverdict.com/articles/pionex-review-2026-grid-trading-bot-platform-with-built-in-exchange-features/); [Medium guide](https://medium.com/@SingaporeHODL/the-complete-guide-to-pionex-grid-bot-pionex-trading-bot-series-f89b09eddefd)

**Classic CTA / trend benchmark (multi-asset, mostly non-crypto; context)**
- [LIVE/AUDITED index] SG Trend Index: **+2.4% in 2025**, after reaching **−9.33% YTD in April 2025** (April alone was −4.89%). SG CTA ended 2025 at −0.2%. Since inception the index has a **CAGR of 5.33% and max drawdown of 20.61%**. — [Top Traders Unplugged, Dec 2025](https://www.toptradersunplugged.com/trend-following-performance-report-december-2025/); [Aussie Turtles](https://www.aussieturtles.com/battle-of-the-trend-following-indexes-december-2025/); [SG CTA update](https://content.sgmarkets.com/CTA_UPDATE_KEEPING_UP_WITH_THE_TRENDFOLLOWERS_2025)

### Inferences
- Every family with credible positive evidence works on **daily or multi-day decisions**, a **6-hour bar at the fastest** (AdaptiveTrend), or is a **structural premium** rather than a price prediction (carry). Nothing credible supports 5-minute technical scoring for retail after costs.
- Trend following's documented benefit is mainly **downside protection and convexity**, not a high hit rate. Our bot's bear-window result (−7% vs −28% for buy-and-hold) fits this. Its losses in the other windows also fit: trend systems pay for that protection with whipsaw costs.
- Carry and basis arbitrage needs a short leg (perps or futures). It is **not available** to a long-only Bitpanda Fusion spot account, and its excess return appears to be fading as capital (for example Ethena) arbitrages it away.
- Cross-sectional momentum needs a long/short book across many coins. A long-only BTC/ETH/SOL version only captures part of it.

### Gaps
- **DCA**: I found no independent study of DCA's risk-adjusted performance in crypto. It is basically buy-and-hold with time diversification and has no "edge" to verify.
- **Market making (Hummingbot or exchange MM programs)**: I found no public live P&L for retail market makers. Professional MMs (Wintermute, Jump and others) do not publish audited strategy returns.
- **Fear & Greed Index as a signal**: not researched within the tool budget, and no academic evidence was found.
- I could not open the full papers (fetch blocked) to extract drawdowns and costs for Zarattini et al. and Liu/Tsyvinski/Wu (whether the momentum returns survive realistic costs and shorting constraints).

---

## Q2: What do audited or public track records show (crypto fund indices, academic papers, Freqtrade/Jesse/Hummingbot live results, bot marketplaces, copy trading)?

### Takeaway
Independently audited track records of **systematic crypto trading** are scarce. Fund-index data mostly reflects market beta. Open-source bot frameworks publish **no live performance statistics**, and their own docs warn that backtests do not transfer to live trading. Copy-trading and bot-marketplace data shows a large gap between what leaders display and what followers actually get.

### Cited Findings
- [Secondary statistics site, low confidence] Crypto hedge funds reportedly averaged **36% in 2025**, with crypto hedge fund AUM put at $136.2bn in Q2 2025. The **HFR Cryptocurrency Index fell −8.0% in November 2025**. — [SQ Magazine](https://sqmagazine.co.uk/crypto-hedge-funds-statistics/). The AUM figure looks implausibly high and is unverified.
- [Industry analysis] CAIA (2022) asks whether crypto hedge funds are "just Bitcoin-beta plays", meaning much of their reported return is market exposure rather than alpha. — [CAIA blog](https://caia.org/blog/2022/01/30/cryptocurrency-hedge-funds-just-bitcoin-beta-plays); see also [Hedge Fund Alpha](https://hedgefundalpha.com/news/how-do-crypto-hedge-funds-compare-to-bitcoin-and-ethereum/)
- [Data-provider note] Eurekahedge crypto indices are being moved under With Intelligence. I found no 2024–2025 crypto-CTA index figures through search. — [With Intelligence](https://www.withintelligence.com/eurekahedge-data-on-with-intelligence/); [Finadium](https://finadium.com/eurekahedge-ai-and-crypto-fund-performance-flat-in-june/)
- [Framework docs] Freqtrade's own documentation says that **"Backtesting will never replace running a strategy in dry-run mode"** and that good backtest results don't guarantee live profits, and neither do dry-run results. Backtests assume every order fills, while live and dry runs do not. — [Freqtrade backtesting docs](https://www.freqtrade.io/en/stable/backtesting/); [GitHub issue #8451 backtest vs dry-run gap](https://github.com/freqtrade/freqtrade/issues/8451)
- [MARKETING] Freqtrade "case study" posts such as "2509% Profit Unlocked" are backtest showcases on Medium, not verified live results. — [Medium](https://imbuedeskpicasso.medium.com/2509-profit-unlocked-a-case-study-on-algorithmic-trading-with-freqtrade-39b1051c0f1e)
- [Low-quality study, platform-affiliated] A "90-day multi-exchange study" of over 100,000 copy-trading outcomes on Binance, Bybit and MEXC reports that **97.04% of lead traders were profitable on their own accounts, but only 43.61% made money for followers, and only 48.48% of followers ended profitable**. It also notes a leader can show a >57% win rate and still lose followers money if average losses exceed average wins. The publisher is a commercial site and the method is unaudited. — [YieldFund](https://yieldfund.com/is-copy-trading-profitable-a-90-day-multi-exchange-study)
- [ACADEMIC] Kawai et al., "Stranger Danger? Investor Behavior and Incentives on Cryptocurrency Copy-Trading Platforms" (CHI 2024, Carnegie Mellon). It studies leader incentives (profit share) and copier behaviour. Related experimental work (Apesteguia, Oechssler & Weidenholzer) finds that **copy trading leads to excessive risk taking**. — [CMU PDF](https://www.andrew.cmu.edu/user/nicolasc/publications/Kawai-CHI24.pdf); [ACM DL](https://dl.acm.org/doi/full/10.1145/3613904.3642715); [BSE "Copy Trading"](https://bse.eu/research/publications/copy-trading)
- [ACADEMIC, University of Florida, 2025] Social crypto traders who gained or lost followers then traded more often and took more risk, and performed **about 10% worse** in the following weeks. — [UF News](https://news.ufl.edu/2025/03/crypto-social-traders/)
- [Platform marketing] Pionex describes its own "AI 2.0" backtester as "comparable to Goldman Sachs GS Quant". This is a pure marketing claim. — [Pionex blog](https://www.pionex.com/blog/botai2_en/)

### Inferences
- In the public domain, the **only real-money systematic crypto track records** with meaningful scale and transparency are (a) on-chain delta-neutral or funding-rate products like Ethena, and (b) multi-asset CTA indices where crypto is a small part. I found no audited record of a retail-style TA bot.
- Copy-trading leaderboards suffer from survivorship and selection bias, and leaders' displayed returns do not carry over to followers once delay, fees and profit share are paid.

### Gaps
- I found no published, audited live track records for Jesse, Hummingbot or Freqtrade strategies. Community "live results" threads exist, but they are self-reported and not independently checkable. I could not open them within the budget.
- I found no exchange disclosure, from Binance, Bitget or 3Commas, of the share of bot or copy users who are profitable.
- Crypto-specific systematic or CTA fund indices (Eurekahedge Crypto, HFR Crypto, Galaxy, CoinShares or Kaiko research) with annual returns and Sharpe for 2023–2025: not retrieved.

---

## Q3: What share of retail algo/bot traders actually make money?

### Takeaway
There is no rigorous, crypto-bot-specific study. The best hard evidence comes from adjacent retail markets: **97% of persistent day traders lose money** (Brazil, full population data). The crypto-bot figures in circulation (about 27% successful after six months) are unverified aggregator numbers.

### Cited Findings
- [ACADEMIC, full population, equity futures, data 2013–2015 (older)] Chague, De-Losso & Giovannetti, "Day Trading for a Living?". The sample is 19,646 individuals day trading mini-Ibovespa futures. **97% of those who persisted for more than 300 days lost money.** Only 1.1% earned more than the minimum wage, and only 0.5% more than a bank teller's starting salary. There was **no evidence of learning**. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101); [IDEAS/RePEc](https://ideas.repec.org/p/fgv/eesptd/525.html); [Quantpedia summary](https://quantpedia.com/?p=4771)
- [MARKETING/UNVERIFIED aggregator] "About 27% of automated crypto trading accounts are successful after six months; about 73% fail", attributed to "For Traders and Coincub". There is no traceable method. — [lenz.io](https://lenz.io/q/percentage-of-crypto-trading-bots-successful); [lenz.io](https://lenz.io/q/success-rate-of-crypto-trading-bots)
- [Survey, small] A CryptoDataDownload survey found that **38.05% of respondents used bots, but 86.33% of the money in the study was traded by bots**. It concerns usage, not profitability. — [CryptoDataDownload](https://www.cryptodatadownload.com/blog/posts/humans-bot-investors-behavior-findings/); [Adam Cochran Substack](https://adamcochran.substack.com/p/86-of-crypto-capital-is-traded-by?open=false)
- [Adjacent crypto market] CryptoRank data reports that **71% of prediction-market (Polymarket) users lose money**, while bots capture gains. — [Crypto Briefing](https://cryptobriefing.com/prediction-market-users-lose-money-cryptorank/)
- [Low-quality, see Q2] Only 48.48% of copy-trading followers were profitable over 90 days. — [YieldFund](https://yieldfund.com/is-copy-trading-profitable-a-90-day-multi-exchange-study)

### Inferences
- A reasonable prior is that **a clear majority of retail algo traders lose money after fees**, with the active, high-turnover subset losing most. The evidence for this prior is indirect: day-trading population studies, copy-trading data, and the fee-sensitivity results in Q1.

### Gaps
- No peer-reviewed study measures realized P&L of retail crypto bot users, whether on 3Commas, Pionex or Freqtrade.
- EU-mandated CFD loss disclosures (the "X% of retail accounts lose money" line, often in the 70–80% range) are relevant, but I did not source them here. The writer should cite an ESMA or broker page directly if needed.
- Barber, Lee, Liu & Odean (Taiwan day traders) is another key reference that was not retrieved.

---

## Q4: What traits do surviving strategies share (timeframe, trade frequency, number of rules, holding period, assets, fee sensitivity)?

### Takeaway
The strategies that survive scrutiny share a pattern:
- **slow signals**: daily to weekly, at most 6-hour bars
- **few, robust rules**, often averaged over several lookbacks instead of one optimized setting
- **volatility-scaled position sizing**
- **low turnover**: holds lasting days to weeks
- **a diversified universe** of the most liquid coins
- a **low hit rate with large winners**

Fee drag is the main killer of fast strategies.

### Cited Findings
- **Timeframe and fee drag.** Costs change profitability dramatically at 1-minute frequency but not much at daily frequency. — [ScienceDirect 2022](https://www.sciencedirect.com/science/article/abs/pii/S1042443122000816)
- **Lookback and turnover trade-off.** Shorter lookbacks raise gross Sharpe, but rebalancing costs erode it. — [Monash, Le & Ruthbah](https://www.monash.edu/business/monash-centre-for-financial-studies/our-research/all-projects/investment-strategy/trend-following-strategies-for-crypto-investors)
- **Ensembles instead of a single tuned parameter.** Zarattini et al. combine Donchian models across many lookbacks into one signal and size positions by volatility. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)
- **Universe.** The best results come from rotating among about 20 liquid coins (Zarattini et al.). Momentum works among larger coins: 4.2%/week above median size vs 0.6% (insignificant) below. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907); [NBER](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)
- **BTC alone is hard.** Out of sample, Bitcoin showed no technical-rule predictability, while altcoins did. — [Hudson & Urquhart, IDEAS](https://ideas.repec.org/a/spr/annopr/v297y2021i1d10.1007_s10479-019-03357-1.html)
- **Win rate is not profitability.** A >57% win rate can still lose money if average losses exceed average wins. — [YieldFund](https://yieldfund.com/is-copy-trading-profitable-a-90-day-multi-exchange-study)
- **Regime dependence, even for proven systems.** The SG Trend Index fell to −9.3% YTD in April 2025 before ending +2.4%. Its long-term max drawdown is 20.6%. — [Top Traders Unplugged](https://www.toptradersunplugged.com/trend-following-performance-report-december-2025/)
- **Grids** profit in ranges and underperform in trends, according to platform sources. — [BotVerdict](https://botverdict.com/articles/pionex-review-2026-grid-trading-bot-platform-with-built-in-exchange-features/)

### Inferences
- **Fees vs our bot.** At 0.50% round-trip cost, every trade must clear a material hurdle. A 5-minute signal checked every 45 s produces many trades whose expected gross move (TP 12×ATR on 5-minute ATR) is small next to the 0.5% fee. The literature says to move decisions to daily, or at least multi-hour, bars, or to sharply cut trade count.
- **Rule count.** Our "Confluence" score has about 8 inputs and a precisely tuned threshold (0.647). That is the opposite of the "few rules, ensemble over parameters" pattern, and the three decimal places suggest optimisation to the sample.
- **Win rate vs break-even.** Our measured ~35–40% win rate vs ~41% break-even is typical of trend systems. These survive only when the payoff ratio is large enough after fees. The fixed 12×ATR/5×ATR on 5-minute ATR may be too small in euro terms relative to 0.5% costs.
- **Asset count.** With only three coins (BTC/ETH/SOL), the diversification that drives Sharpe in the top-20 rotational studies is missing.

### Gaps
- I found no quantitative "survivor profile" study (for example, average holding period of profitable live bots). The traits above are inferred from the academic backtests.

---

## Q5: Red flags (survivorship bias, backtest-only claims, overfitting)

### Takeaway
Most "proven" bot claims fail at least one of these checks: backtest-only, survivorship-biased coin universe, untested parameter sweeps, unrealistic fills, or leaderboard selection. The credible studies explicitly correct for survivorship and data snooping and model costs.

### Cited Findings
- **Survivorship bias in the coin universe.** Zarattini et al. explicitly use a survivorship-bias-free dataset of every coin traded since 2015. Studies that don't are suspect. — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)
- **Data snooping.** Testing thousands of rules (15,000 in Hudson & Urquhart; 75,360 in the 2024 IREF paper) requires multiple-hypothesis corrections. Even after correction, BTC showed no out-of-sample predictability. — [IDEAS](https://ideas.repec.org/a/spr/annopr/v297y2021i1d10.1007_s10479-019-03357-1.html); [ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S1059056024003010)
- **Backtest-to-live gap.** Freqtrade notes that backtests assume all orders fill, and that dry-run and live entry prices differ. — [Freqtrade docs](https://www.freqtrade.io/en/stable/backtesting/); [Issue #8451](https://github.com/freqtrade/freqtrade/issues/8451)
- **Sub-period fragility.** In "A Decade of Evidence", EMA and DEMA rules did poorly in the 2017–18 and 2018–19 slices despite good full-period Sharpe. The headline 255% walk-forward annual return is dominated by BTC's early hyper-growth. — [arXiv](https://arxiv.org/abs/2009.12155)
- **Edge decay.** Carry Sharpe reportedly fell from 6.45 (2020–2025) to negative in 2025 (attribution uncertain, see Q1). Ethena yields fell from about 18% (2024) to about 4–15% (2025). — [arXiv 2510.14435](https://arxiv.org/pdf/2510.14435); [eco.com](https://eco.com/support/en/articles/15254002-ethena-usde-and-susde-2026-delta-neutral-yield)
- **Leaderboard and marketing claims.** Leaders' own-account profitability (97%) vs follower profitability (44%). Platform claims such as "Goldman Sachs-level backtester" and "8–12% monthly". — [YieldFund](https://yieldfund.com/is-copy-trading-profitable-a-90-day-multi-exchange-study); [Pionex](https://www.pionex.com/blog/botai2_en/)
- **Exceptionally high preprint Sharpe values** (2.41 in AdaptiveTrend across 150+ pairs) without replication call for scepticism. — [arXiv 2602.11708](https://arxiv.org/html/2602.11708v1)
- **Fund returns that are really beta.** Crypto hedge fund "returns" are often Bitcoin beta. — [CAIA](https://caia.org/blog/2022/01/30/cryptocurrency-hedge-funds-just-bitcoin-beta-plays)

### Inferences
- A practical checklist for judging any claim, our own bot included:
  1. live or audited, or only a backtest?
  2. survivorship-free universe?
  3. how many parameter combinations were tried, and is there a deflated-Sharpe or White's-reality-check correction?
  4. are fees and slippage modelled at the real tier (0.25% per side here)?
  5. sub-period stability, including bull, bear and range?
  6. turnover relative to costs?
  7. benchmark vs buy-and-hold on risk-adjusted terms (Sharpe and max DD), not raw return?
- On point 7, our bot's bear-window result is a legitimate positive. The overall negative result across three windows with a tuned 0.647 threshold is a typical sign of overfitting plus fee drag.

### Gaps
- I found no study measuring how much published crypto bot backtests overstate live performance (a "haircut" factor, like the backtest-overfitting literature for equities).
