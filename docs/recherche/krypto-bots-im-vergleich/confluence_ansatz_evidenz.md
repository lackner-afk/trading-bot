# Evidence assessment: multi-factor "confluence" scoring on 5-minute crypto candles

Research note, October 2026. Scope: this note checks the evidence for (a) combining many indicators into one weighted score and (b) each building block the bot uses: multi-timeframe EMA trend, momentum, breakout, volume confirmation, volatility filter, Connors RSI-2 mean reversion, the Fear & Greed Index and the CPI/FOMC filter. It looks at these on short timeframes, for long-only spot trading with a fee of 0.25 % per order.

**About the sources.** The network proxy blocked full-text fetching from arXiv, SSRN mirrors, Quantpedia, CXO Advisory, Concretum and journal.fsv.cuni.cz. Most findings below therefore come from abstracts and search-result summaries, not from reading the full papers. Items marked **[background]** are well-known publications cited from prior knowledge with their DOI or SSRN link; they were not re-fetched in this session. Evidence strength is marked as **strong** (several independent studies or a leading journal with a multiple-testing correction), **moderate** or **weak/single-study**.

---

## 1. Does combining many technical indicators (confluence, ensembles, voting) improve out-of-sample results, or mainly add parameters and overfitting?

### Takeaway
Every free parameter, weight and threshold adds to the number of strategy variants that were effectively tried, and the backtest-overfitting literature shows that this sharply raises the chance that the best in-sample configuration underperforms out of sample. Combining signals helps only when the combination is constrained: equal weights, averages over many lookbacks, or principal components. It does not help when weights and thresholds are fitted. The bot's best threshold moving from 0.65 to 0.87 depending on the backtest window is a textbook symptom of a high probability of backtest overfitting (PBO).

### Cited Findings
- Bailey, Borwein, López de Prado and Zhu (*J. Computational Finance*, 2015/2017) argue that standard hold-out methods are "unreliable and inaccurate" for investment backtests. They propose combinatorially symmetric cross-validation (CSCV) to estimate the probability that the best in-sample configuration is overfit (PBO). Strength: strong (methodological standard). — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253); [Risk.net / JCF](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting)
- An R package (`pbo`) implements CSCV. It reports PBO, performance degradation, probability of loss and stochastic dominance, so the test is practical to run on a parameter grid. — [CRAN pbo README](https://packages.oit.ncsu.edu/cran/web/packages/pbo/readme/README.html)
- **[background]** Bailey and López de Prado, "The Deflated Sharpe Ratio" (2014): the Sharpe ratio expected from the best of N trials grows with N even when no trial has skill. A reported Sharpe has to be deflated by the number of trials and by non-normal returns. — [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- **[background]** Bailey, Borwein, López de Prado and Zhu, "Pseudo-Mathematics and Financial Charlatanism" (*Notices of the AMS*, 2014): with enough configurations tried, a high in-sample Sharpe can be reached on random data, and backtests that do not report the number of trials are uninformative. — [SSRN 2308659](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)
- **[background]** Harvey, Liu and Zhu (*RFS*, 2016), "…and the Cross-Section of Expected Returns": after multiple testing, a new factor should clear a t-statistic of about 3.0 rather than 2.0. — [doi:10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059)
- **[background]** Neely, Rapach, Tu and Zhou (*Management Science*, 2014) combined 14 technical indicators with macro variables for the equity premium, using principal components, which is a constrained combination with no fitted per-indicator weights. The combination improved monthly out-of-sample forecasts. This is the strongest evidence that combining technical signals can help, but it applies to monthly equity data and to a combination method with almost no tuning. — [doi:10.1287/mnsc.2013.1838](https://doi.org/10.1287/mnsc.2013.1838)
- Zarattini, Pagani and Barbon (Swiss Finance Institute, 2025) use an ensemble of Donchian-channel trend models with different lookbacks plus volatility-based sizing. They report a net-of-fees Sharpe above 1.5 and annual alpha of 10.8 % versus BTC on a rotational top-20 coin portfolio, using a survivorship-bias-free dataset of all coins since 2015. This is the kind of ensemble that has evidence behind it: one signal family, many lookbacks averaged, nothing fitted per component. Strength: moderate (single working paper, by practitioner authors). — [RePEc SFI rp2580](https://ideas.repec.org/p/chf/rpseri/rp2580.html); [Barbon paper page](https://abarbon.com/papers/catching-crypto-trends)

### Inferences
- The bot's score has at least three fitted component weights (0.55, 0.28, 0.17), a fitted threshold (0.647), and the internal parameters of six technical sub-signals, the Fear & Greed mapping and the macro filter. Including parameter sweeps, the effective number of trials is probably in the hundreds to thousands, so the Deflated Sharpe and PBO frameworks apply directly.
- The optimal threshold moving from 0.65 to 0.87 across windows means the in-sample optimum is unstable. Under CSCV this usually corresponds to a high PBO. A threshold quoted to three decimals (0.647) suggests fine-tuning to noise.
- The ensembles with evidence behind them (Neely et al.; Zarattini et al.) average similar signals across horizons with few or no fitted weights, on daily or monthly data. A heterogeneous weighted vote that mixes trend and mean reversion, as the bot does, has no comparable support. Trend and mean-reversion signals tend to cancel each other, which pushes the score towards the middle, and the threshold then mostly selects noise.

### Gaps
- I found no peer-reviewed study that tests a heterogeneous "confluence score" (trend + mean reversion + sentiment + macro) on intraday crypto data with multiple-testing correction. The idea is common among practitioners but has no academic validation.
- I could not fetch full texts to quote PBO values from the empirical examples in Bailey et al.

---

## 2. Evidence for each component in crypto

### Takeaway
The evidence is positive mainly for **daily and weekly trend or momentum** in crypto (moderate to strong). Intraday technical rules often show *statistical* predictability that does not survive *transaction costs* or later sample periods. Fear & Greed has some weekly-to-monthly predictive power, but not at the 5-minute horizon. FOMC (and probably CPI) releases reliably raise volatility around the release but give no reliable direction, so a filter that reduces risk is reasonable as risk control, not as a source of alpha.

### Cited Findings

**Technical rules in general (daily level)**
- Hudson and Urquhart (*Annals of Operations Research*, 2021) tested about 15,000 rules from five classes on two BTC markets and three other coins, with multiple-hypothesis (data-snooping) corrections. They found significant predictability, but **no predictability for Bitcoin in the out-of-sample period**; predictability remained in other coins. Strength: strong for "it worked historically", weak for "it still works on BTC". — [Springer](https://link.springer.com/article/10.1007/s10479-019-03357-1); [RePEc](https://ideas.repec.org/a/spr/annopr/v297y2021i1d10.1007_s10479-019-03357-1.html)
- Detzel, Liu, Strauss, Zhou and Zhu (*Financial Management*, 2021): ratios of price to its moving average predict **daily** Bitcoin returns both in and out of sample, with economically significant alpha and Sharpe gains over buy-and-hold. They give a rational-learning explanation for why technical analysis works for assets whose fundamentals are hard to value. Strength: strong (leading journal, out-of-sample). — [WUSTL profile](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)
- **[background]** Liu and Tsyvinski (*RFS*, 2021), "Risks and Returns of Cryptocurrency": crypto returns show strong time-series momentum at **daily to weekly** horizons, and investor attention also predicts returns. — [doi:10.1093/rfs/hhaa113](https://doi.org/10.1093/rfs/hhaa113)

**Breakout and short horizons, after costs**
- Search summaries of the crypto technical-trading literature say that significant positive returns often follow breakout signals, but after transaction costs, trading-range-breakout strategies do not beat buy-and-hold; that adjusting for costs "wipes away most of the profits when trading at any frequency"; and that profitability is "highly unstable and declines over time", weakening since 2017. *Caveat: the search tool returned these statements without clear attribution. They appear to come from one of the following sources, which I could not open to confirm.* — [Bakker, Erasmus thesis](https://thesis.eur.nl/pub/41546/Bakker.pdf); [FRL v35 2020 (RePEc)](https://ideas.repec.org/a/eee/finlet/v35y2020ics1544612319308025.html); ["Are simple technical trading rules profitable in bitcoin markets?" IREF 2024](https://ideas.repec.org/a/eee/reveco/v93y2024ipbp858-874.html). Strength: moderate, attribution uncertain.
- **[background]** Corbet, Eraslan, Lucey and Sensoy (*Finance Research Letters*, 2019) tested variable moving-average and trading-range-breakout rules on high-frequency (1-minute) BTC data. They found support mainly for VMA rules, with weaker results for breakouts. — [doi:10.1016/j.frl.2019.04.027](https://doi.org/10.1016/j.frl.2019.04.027)
- Market efficiency follows a U-shape across sampling frequencies: there is an intraday frequency at which the market is most efficient. — [unecon.ru conference paper "Does frequency…"](https://en2023.unecon.ru/wp-content/uploads/2023/08/does_frequency.pdf). Strength: weak/single-study.

**Short-term mean reversion (RSI-2 style)**
- Padysak and Vojtko (2022), "Seasonality, Trend-following, and Mean reversion in Bitcoin", via Quantpedia, on **daily** data: buying BTC at a 10-day **maximum** (trend) beat buying at a 10-day **minimum** (mean reversion), with higher returns and smaller drawdowns, although both were profitable. Shorter lookbacks worked better. The combined MAX-or-MIN strategy returned 98.4 % a year at 47.8 % volatility with a −37.7 % maximum drawdown (in-sample period). Strength: weak to moderate (practitioner research, in-sample). — [Quantpedia: Trend-following and Mean-reversion in Bitcoin](https://quantpedia.com/trend-following-and-mean-reversion-in-bitcoin/); [Quantpedia: Revisiting…](https://quantpedia.com/revisiting-trend-following-and-mean-reversion-strategies-in-bitcoin/) (follow-up post; could not fetch its out-of-sample numbers)
- I found no peer-reviewed evidence that Connors RSI-2 on **5-minute** crypto candles is profitable after costs of around 0.5 % per round trip.

**Volume confirmation**
- Shen, Urquhart and Wang (*Financial Review*, 2022) use volume as a proxy for trading time in 24/7 BTC. They find that the first half-hour return predicts the last half-hour return, with the strongest predictability in the highest-volume and highest-volatility sessions, and the effect is driven by liquidity provision. Volume matters here as a timing variable, not as a per-signal "confirmation" filter. — [Birmingham repository](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/); [Reading eprint](https://reading-9.eprints-hosting.org/100181/3/21Sep2021Bitcoin%20Intraday%20Time-Series%20Momentum.R2.pdf)
- I found no robust study showing that a "volume > X × average" confirmation adds out-of-sample value to 5-minute crypto signals (gap).

**Fear & Greed Index (alternative.me)**
- Albrecht, Pastorek and Maňoušek (2025), "Riding the Waves of Crypto Sentiment" (BTC, ETH, BNB, XRP, ADA): Fear & Greed predicts returns over **one week to one month** after a sentiment change. — [journal.fsv.cuni.cz PDF](https://journal.fsv.cuni.cz/storage/1548_attachment.pdf) (abstract via search; the full text was blocked, so I could not confirm the sign, contrarian or momentum)
- Cavalheiro, Vieira and Thue (*Review of Behavioral Finance*, 2024) run Granger-causality tests of Fear & Greed on BTC and ETH returns. — [RePEc](https://ideas.repec.org/a/eme/rbfpps/rbf-08-2023-0224.html) (the direction of causality was not visible in the snippet; it may well run from returns to sentiment)
- ARDL/ECM on **monthly** data, 2016–2021: a positive and significant relationship between the Fear & Greed Index and BTC returns, i.e. a co-movement or momentum relation rather than a contrarian one. — [Virtus Interpress](https://virtusinterpress.org/How-does-the-Bitcoin-Sentiment-Index-of-Fear-Greed-affect-Bitcoin-returns.html). Strength: weak (small journal, monthly frequency).
- The BTC risk-return relation is positive only in "Extreme Fear" periods. — [Asian Review of Financial Research 2022](https://www.doi.org/10.37197/ARFR.2022.35.3.2). Strength: weak/single-study.
- Contradiction: the evidence is mixed on whether the index acts as a contrarian or a momentum signal. No study I found uses it at an intraday horizon. The index is published **once a day**, so on 5-minute candles it is a nearly constant regime variable.

**Macro events (FOMC, CPI)**
- Digital-asset return volatility rises significantly at the FOMC statement release and in the window from one hour before to one hour after, with peaks at the release and about 30 minutes later (press conference). — [arXiv 2302.10252, "Monetary Policy, Digital Assets, and DeFi Activity"](https://arxiv.org/pdf/2302.10252) (abstract via search; the full text was blocked)
- FOMC events have a significant negative effect about 4 days after the announcement for most cryptocurrencies studied. — [arXiv 2311.10739](https://arxiv.org/pdf/2311.10739). Strength: weak/single-study; this directional result is unlikely to be stable.
- BTC moves in close correlation with Nasdaq and the S&P 500 around FOMC meetings. — [investinglive.com](https://investinglive.com/cryptocurrency/bitcoin-swings-wildly-as-volatility-in-fed-expectations-increases-ahead-of-us-cpi-and-fomc-decision/) (news, low quality)
- CPI-specific academic work is mostly theses, for example a 2025 Kyiv School of Economics thesis, "Bitcoin's Reaction to U.S. FOMC and CPI Announcements". — [KSE thesis PDF](https://kse.ua/wp-content/uploads/2026/05/illia-nazaruk_268722_assignsubmission_file_nazaruk_final_thesis.pdf). Strength: weak.

### Inferences
- The components with real support (MA-ratio trend and time-series momentum) are supported on **daily** data. Moving them to 5-minute candles leaves that evidence behind.
- Mean reversion and trend-following in the same score work against each other. The Bitcoin evidence (Padysak/Vojtko) favours trend over mean reversion even on daily data.
- Fear & Greed works at best as a slow regime input over weeks. At a 28 % weight, it mostly shifts the threshold by day, which is an implicit regime switch. That can be reasonable, but nothing supports fitting its weight precisely.
- A CPI/FOMC filter that reduces size or pauses entries around releases fits the volatility evidence and costs little. It should be treated as risk control and not scored as a return source.

### Gaps
- I could not verify whether Fear & Greed acts contrarian or as momentum in the Albrecht et al. study.
- I found no peer-reviewed evidence on volatility filters (for example ATR bands) as entry filters on intraday crypto.

---

## 3. Is 5-minute data a sensible timeframe for a retail bot paying 0.25 % per order? What does research say about predictability by horizon?

### Takeaway
The intraday crypto effects that are documented are small and tied to specific times (for example the half-hour momentum effect around high-volume sessions). Breakeven-cost analyses and post-2017 samples show they mostly disappear after realistic costs. With 0.5 % per round trip, a 5-minute strategy has to beat a cost that is several times the typical 5-minute price move, which is a structural disadvantage.

### Cited Findings
- Bitcoin intraday time-series momentum: the first half-hour return predicts the last half-hour return, and the effect is strongest in high-volume and high-volatility sessions and in market downturns. — [Shen, Urquhart & Wang 2022, Financial Review](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/)
- Daily MA-ratio signals predict BTC out of sample. — [Detzel et al. 2021](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)
- Momentum is documented mainly at daily to weekly horizons. — **[background]** [Liu & Tsyvinski 2021, RFS](https://doi.org/10.1093/rfs/hhaa113)
- Cost adjustment removes most technical-rule profits at any frequency, and profitability has declined since 2017 (attribution uncertain, see Section 2). — [Bakker thesis](https://thesis.eur.nl/pub/41546/Bakker.pdf); [IREF 2024](https://ideas.repec.org/a/eee/reveco/v93y2024ipbp858-874.html)
- Hudson and Urquhart: no out-of-sample BTC predictability, even on daily data with low-cost assumptions. — [Springer](https://link.springer.com/article/10.1007/s10479-019-03357-1)
- A single seasonality rule (long BTC from 21:00 to 23:00 UTC) is reported at 33 % annual return, 20.9 % volatility and a −22.5 % maximum drawdown. This is one of the few intraday effects with a documented edge, and it trades only once a day. Strength: weak (practitioner backtest, cited via a TradingView script). — [TradingView script referencing Padysak & Vojtko](https://de.tradingview.com/script/IzFZxayj)

### Inferences (own arithmetic, not from a source)
- **Cost hurdle.** At 0.25 % per side, a round trip costs about 0.50 % plus spread and slippage. If BTC's daily volatility is around 2.5–3.5 %, the 5-minute standard deviation is about 3 %/√288 ≈ 0.15–0.2 %. A round trip therefore costs about 2.5–3.5 typical 5-minute moves. The signal has to predict moves several times larger than normal 5-minute noise just to break even.
- **Win-rate arithmetic.** With a 35–40 % win rate, the average net win has to be about 1.5–1.9 times the average net loss to break even ((1−p)/p). The fixed 0.5 % cost lowers every win and raises every loss, so the gross ratio needed is even higher. This matches the observed result: the backtest loses money with real fees.
- **Practical conclusion.** If short timeframes are kept, they should be used only for **execution** (timing an entry decided on daily or 4-hour data), not for **signal generation**.

### Gaps
- I found no paper that maps the realistic net Sharpe of crypto technical signals by horizon (5m vs 1h vs 1d) at retail costs around 0.5 % per round trip. The cost arithmetic above is an inference.
- Wen et al. and other intraday momentum/reversal crypto papers (for example in *North American Journal of Economics and Finance*) could not be fetched.

---

## 4. Which simpler alternatives have better evidence?

### Takeaway
The alternatives with the best support are (a) a **daily trend filter or time-series momentum** on BTC/ETH, preferably an ensemble over several lookbacks, and (b) **volatility-targeted position sizing**. Both trade rarely, so the 0.25 % fee matters far less, and both have out-of-sample or multi-study support.

### Cited Findings
- Daily price-to-moving-average signals give out-of-sample alpha and Sharpe gains on BTC. — [Detzel et al. 2021](https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/)
- A Donchian ensemble with volatility sizing reaches a net Sharpe above 1.5 on a top-20 coin rotation. It is applied to BTC and to the full coin universe since 2015, and the paper proposes ways to reduce transaction costs. — [Zarattini, Pagani & Barbon 2025](https://ideas.repec.org/p/chf/rpseri/rp2580.html)
- Buying BTC at a 10-day high (trend) beat mean reversion on daily data. — [Quantpedia, Padysak & Vojtko](https://quantpedia.com/trend-following-and-mean-reversion-in-bitcoin/)
- Volatility scaling of a Bitcoin allocation improves the Sharpe ratio by about 0.40 ("40 Sharpe points"), whatever volatility estimator is used. — [Man Group, "Crypto: too hot to handle"](https://www.man.com/insights/crypto-too-hot-to-handle). Strength: moderate (industry research).
- Low-volatility crypto portfolios with 6–12-month look-backs earn significant excess returns, and stop-loss rules improve Sharpe ratios. — [FRL v46 2022 (RePEc)](https://ideas.repec.org/a/eee/finlet/v46y2022ipbs1544612321004116.html). Strength: moderate.
- **[background]** Moreira and Muir (*Journal of Finance*, 2017), "Volatility-Managed Portfolios": scaling exposure by inverse past variance raises Sharpe ratios across equity factors. This is the general basis for volatility targeting. — [doi:10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513)

### Inferences
- For long-only spot trading with 0.25 % fees, a design such as "hold BTC/ETH when the daily close is above an ensemble of moving averages or Donchian levels (for example 20/50/100/200 days), size by inverse 30-day volatility, rebalance daily or weekly" is supported by much more evidence than a 5-minute confluence score. It would also make only a few dozen trades a year.
- The bot already uses a 200-EMA direction filter. The evidence suggests that a daily trend filter of this kind is the most useful *existing* component, and that the intraday layers on top of it mainly add costs and parameters.

### Gaps
- I could not get Bitcoin-only figures (CAGR, maximum drawdown) for the Zarattini et al. strategy because the full text was blocked.
- Volatility targeting may help less for crypto, where volatility clustering and the volatility-return relation behave differently from equities; I found no peer-reviewed crypto-specific replication of Moreira–Muir.

---

## 5. Does regime detection (trending / ranging / high volatility) help in practice?

### Takeaway
Academic hidden-Markov and regime-switching models find three to four regimes in crypto and report out-of-sample predictive ability. However, the evidence comes from small samples (for example 2016–2019), uses daily data and rarely includes costs. Explicit regime classifiers add more parameters. A simple trend filter combined with volatility scaling already behaves like a regime model, with fewer parameters.

### Cited Findings
- Regime-switching models with Markov-modulated parameters, January 2016 to October 2019: three to four states per cryptocurrency, at most three common states for the basket, and the authors claim the models "can be exploited to build profitable investment strategies". — [Bayesian HMM paper, arXiv 2011.03741](https://ar5iv.labs.arxiv.org/html/2011.03741); [EconPapers, Econ. & Finance v57 2021](https://econpapers.repec.org/RePEc:eee:ecofin:v:57:y:2021:i:c:s1062940821000577). Strength: weak to moderate (short in-sample period, costs not confirmed).
- A four-state non-homogeneous HMM separates bull, bear and calm regimes for BTC. — [MDPI JRFM 13(12):311](https://www.mdpi.com/1911-8074/13/12/311/htm)
- Three-state HMMs with diagonal covariance show good out-of-sample predictive performance. — search summary of the [Vilnius University publication](https://epublications.vu.lt/object/elaba:246222004/246222004.pdf). Strength: weak (attribution and details unverified).
- Bitcoin intraday momentum is strongest in high-volatility and high-volume sessions and in downturns, an implicit regime dependence. — [Shen, Urquhart & Wang 2022](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/)

### Inferences
- "Predictive performance" in HMM papers usually means statistical forecast accuracy, not net trading profit after retail fees. That is not enough to justify adding a regime classifier to a strategy that is already overparameterized.
- If regimes are used, the evidence points to a few slow, interpretable states (daily trend up or down; high or low volatility from realized volatility) that scale exposure, rather than switching between strategies (trend vs RSI-2) every 5 minutes. A per-regime choice of strategy multiplies the number of parameters and raises PBO further.

### Gaps
- I found no study that measures the *incremental* net-of-cost benefit of a regime classifier over a plain trend filter with volatility targeting in crypto.
- I found no practitioner out-of-sample results (QuantConnect, Robot Wealth) on crypto regime switching at intraday frequency; these sources could not be fetched.
