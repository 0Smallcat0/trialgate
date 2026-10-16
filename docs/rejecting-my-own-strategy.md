# I built a validation gate. Then it rejected my own strategy.

*July 2026 — the story behind [trialgate](https://github.com/0Smallcat0/trialgate).*

## The trap every backtest sets

Optimizing four parameters on a **pure random walk** can produce an in-sample
Sharpe ratio of 1.27. That number comes from the backtest-overfitting
literature, and it should terrify anyone who has ever tuned a strategy until
the equity curve looked right: the curve *always* eventually looks right,
whether or not there is anything real underneath.

The academic answer has existed for a decade — Bailey & López de Prado's
Deflated Sharpe Ratio (2014), the CSCV estimate of the Probability of
Backtest Overfitting (2015), Arnott, Harvey & Markowitz's backtesting
protocol (2019). The retail answer is usually to run one backtest and ship a
bot. I wanted the academic answer as running code, applied to my own ideas,
with no escape hatches.

So the rule became: **a strategy earns belief by surviving verification, not
by looking good in-sample.** Concretely, six gates — an append-only trial
registry (unregistered results are void), a data floor, PBO ≤ 0.05, DSR ≥
0.95, a single-use locked holdout, and a ≥3-month paper-trading period with
measured real costs. Claims are pre-registered *before* the first trial runs,
so the pass/fail line cannot quietly move after seeing results.

The first deployment was a daily long-only trend system for crypto spot
(BTC/ETH). This story is about the second deployment — the one the gate
killed.

## The test case: same strategy family, Taiwan market

The strategy is deliberately boring: a four-SMA ensemble (20/65/150/200-day)
on daily closes; target exposure equals the fraction of trend lines the price
sits above, giving a 0/25/50/75/100% ladder. Long only, signal break → cash.
Adapting it to Taiwan's 0050 ETF (dividend- and split-adjusted series,
2004–2025, 5,217 trading days) felt promising — the same family had strong
published evidence in crypto.

Two claims were pre-registered before trial #1, both required, all costs
included, measured against dividend-included buy-and-hold:

- **C1:** max drawdown ≤ 60% of buy-and-hold's max drawdown.
- **C2:** CAGR ≥ buy-and-hold CAGR − 3 percentage points.

Five trials went into the registry — including two voided for defects (a
tick-size check that produced zero fills, an annualization bug) that stayed
in the log anyway, because the registry is append-only and the trial count N
feeds the statistics.

## The verdict

Trial #4, the adjudication run, with Taiwan's statutory cost structure
(commission 8.55 bps/side with a minimum ticket, 10 bps ETF sell tax, 5 bps
slippage), over 21.2 years:

| | Strategy | 0050 buy-and-hold |
|---|---|---|
| CAGR | 4.91%/yr | 10.21%/yr |
| Max drawdown | **26.32%** | 55.75% (2008) |
| Ladder changes | **49.2/yr** | 0 |

- **C1: PASS.** 26.32% ≤ 33.45%. The drawdown protection is real — the
  system sat in cash through most of 2008 and fully dodged the April 2025
  crash day.
- **C2: FAIL.** 4.91% vs the required 7.21% — trailing by **5.3 pp/year**,
  far outside the pre-registered 3 pp tolerance.

**Overall: FAIL.** Per the pre-registered rule, the runtime — notifications,
dashboard, paper account — was never built. An adversarial audit re-derived
the numbers three days later and upheld the verdict.

## Why it failed (the interesting part)

The whipsaw census told the story. The ladder moved 49 times per year — this
is not a "few signals a year" product, it is a weekly rebalancing machine,
and the 20-day SMA sub-signal was the main culprit (in one month of 2025 it
bounced between 75% and 100% exposure four times).

Decomposing the 5.3 pp gap: roughly **2.6 pp/year was cost drag** (blended
~18.6 bps/side × 13.7× annual turnover) and roughly **2.7 pp/year was the
whipsaw itself** — systematically selling low and rebuying high around trend
lines. The damning number: even at *zero* transaction costs the gross CAGR
(~7.5%/yr) barely reached the tolerance line. Under Taiwan's statutory cost
structure, this rule family has no room to live.

And the part worth being honest about: the strategy *did* deliver its
promise of crash protection — it just paid for it with the bull market's
compounding. Faber-style timing is disaster insurance, and the premium is
what you miss. Taiwan 2004–2025 compounded at 10.2%/year with dividends; the
premium was unaffordable. Timing didn't fail because the idea is stupid; it
failed because *this market's* buy-and-hold is too strong a benchmark, and
the gate was pointed at exactly that question.

## What I actually bought with the gate

A "no" that cost two weeks instead of a live system quietly bleeding 5 pp a
year — or worse, my own money following it. The failure report is not the
project failing; the failure report **is the product**. Without
pre-registration I would have moved the goalposts ("C2 was too strict…");
without the registry the two buggy trials would have vanished; without the
locked holdout I would have "just checked" the recent data until something
passed. The gate's job is to say no by default and make every yes expensive.

The crypto deployment gets no favoritism: it is held to the same bars and
the same report.

## The tooling

The mechanical parts of the gate — registry, PBO/CSCV, DSR, holdout lock,
threshold report — are extracted into
[`trialgate`](https://github.com/0Smallcat0/trialgate), a zero-dependency,
strictly-typed Python package, so the next strategy (mine or yours) can be
rejected with less ceremony. The full FAIL report lives in the
[tw-stock-trading](https://github.com/0Smallcat0/tw-stock-trading)
repository; the crypto system it came from is
[crypto-quant-signal](https://github.com/0Smallcat0/crypto-quant-signal).

*Method note: the gate specification, thresholds, and pre-registered claims
are mine, drawn from the primary literature; implementation was AI-assisted
and accepted only after review against the spec and a green test suite. The
same doctrine applies to code and to strategies: trust comes from
verification, not from provenance.*
