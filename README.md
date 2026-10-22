# trialgate

[![CI](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml/badge.svg)](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trialgate.svg)](https://pypi.org/project/trialgate/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

You spend three months on a strategy. The backtest shows an annualized Sharpe
of 1.5. You put real money in. Six months later you are down 35%, and you
cannot tell whether you were unlucky or whether you fooled yourself — because
the number that would settle it was never written down anywhere: how many
variants you tried before you kept this one.

That number is the whole game. The best of several hundred attempts on pure
noise looks exactly like skill. You do not have to take that on faith:

```bash
pip install trialgate
python -m trialgate demo
```

Thirty seconds. No data, no keys, no network. Four parameters, 450
configurations, and a **random walk** underneath — noise by construction, with
no edge in it to find:

```
Best in-sample annualized Sharpe: 1.20   (entry=150 exit=20 vol=20/0.7)

Registered trials (the honest N): 450
OK    450 trials, sealed against edits: 1-449

FAIL  data floor (gate 2): 39 (needs >= 1000)  -- only 39 of 2000 periods had a
      return; a period that returned exactly zero is not evidence. 961 more needed
PASS  PBO (gate 3): 0.0418803 (needs <= 0.05)
FAIL  DSR (gate 4): 0.603118 (needs >= 0.95)  -- WARNING returns are serially
      correlated (lag-1 autocorrelation +0.35, Ljung-Box Q(10) p=4.5e-160), which
      this DSR assumes away: adjusted for it, DSR is 0.57409 on 1022 effective of
      2000 observations (would FAIL); annualized Sharpe 1.20 by sqrt(252) is 0.67
      by Lo's eta(q). Pass autocorrelation="adjust" to judge the adjusted value
PASS  stop-line survivability (gate 7): 1.10249e-15 (needs <= 0.1)  -- ran at 2.4%
      annualized volatility against a -15% stop
PASS  declared costs (gate 8): 450 (needs >= 450)
FAIL  cost stress (gate 9): 0.508305 (needs >= 0.95)  -- at 2x the costs its
      returns were charged (DSR 0.603118 at 1x); its mean return reaches zero at
      19.80x those costs
NOT QUALIFIED
```

Sharpe 1.20 is a number people quit their jobs over. There is nothing in that
data. Note which gates caught it, and which did not: PBO passed, comfortably.
One gate clearing is not the same as a strategy clearing, which is why there is
more than one. The winning configuration holds a position on 39 of 2000 days;
the other 1961 days it sat in cash and returned exactly zero, so the Sharpe
rests on 39 observations, not 2000. Those 39 are bunched into a few runs of up
days, so they are also serially correlated and annualizing by √252 overstates
them (see [Serially correlated returns](#serially-correlated-returns)). The
demo's source is also the integration example — swap its price series and its
backtest for yours and the rest is unchanged.

## Quickstart

Register every backtest as it runs, then judge the one you would trade. Save
this as `quickstart.py` and run it in an empty folder; it writes `state/`.
`my_backtest()` is the one function to replace with your engine:

```python
import random
from datetime import UTC, datetime

from trialgate import (
    append_trial,
    evaluate_registry,
    initialize_holdout,
    load_trials,
    sharpe_ratio,
)

NOW = datetime.now(tz=UTC)


def my_backtest(lookback: int) -> tuple[list[float], list[float]]:
    """Stand-in for your engine: each day's return net of costs, and what each
    day paid for execution. Replace it with yours."""
    rng = random.Random(lookback)
    costs = [0.0025 if rng.random() < 0.05 else 0.0 for _ in range(1600)]
    returns = [rng.gauss(0.0004, 0.01) - cost for cost in costs]
    return returns, costs


# Gate 5 — lock the most recent ~12 months BEFORE the first backtest.
initialize_holdout(
    "state/holdout.json",
    holdout_start=datetime(2025, 7, 1, tzinfo=UTC),
    locked_at=NOW,
)

# Gate 1 — register EVERY backtest execution, including the quick ones.
# Passing holdout= enforces gate 5 here: a run reaching into the locked
# period raises instead of being silently recorded.
for lookback in range(5, 205, 5):
    returns, costs = my_backtest(lookback)
    append_trial(
        "state/registry.jsonl",
        recorded_at=NOW,
        code_version="8b91661",
        strategy_id="daily_trend",
        parameters={"lookback": str(lookback)},
        universe=("BTCUSDT",),
        data_start=datetime(2019, 1, 1, tzinfo=UTC),
        data_end=datetime(2025, 6, 30, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "25"},
        metrics={"annualized_sharpe": f"{sharpe_ratio(returns, periods_per_year=252):.4f}"},
        operator_note="lookback sweep",
        holdout="state/holdout.json",
        period_returns=returns,  # what gate 3 builds its matrix from
        period_costs=costs,  # what gate 9 charges a second time
    )

# Judge the one you would trade (the best Sharpe) against every trial registered.
trials = load_trials("state/registry.jsonl")
best = max(trials, key=lambda trial: float(trial.metrics["annualized_sharpe"]))
report = evaluate_registry(
    "state/registry.jsonl",
    candidate_trial_id=best.trial_id,
    periods_per_year=252,
    effective_trials=len(trials),  # required: the registered N is the safe choice
    strategy_id="daily_trend",
    stop_loss_fraction=0.15,  # your hard stop; omitting it skips gate 7
)
print(
    f"trial {best.trial_id}, lookback {best.parameters['lookback']}, "
    f"Sharpe {best.metrics['annualized_sharpe']}"
)
print(report.summary())
```

```
trial 6, lookback 30, Sharpe 1.3950
PASS  data floor (gate 2): 1600 (needs >= 1000)
FAIL  PBO (gate 3): 0.552758 (needs <= 0.05)
FAIL  DSR (gate 4): 0.919357 (needs >= 0.95)
PASS  stop-line survivability (gate 7): 0.0563969 (needs <= 0.1)  -- ran at 16.3%
      annualized volatility against a -15% stop
PASS  declared costs (gate 8): 40 (needs >= 40)
FAIL  cost stress (gate 9): 0.804975 (needs >= 0.95)  -- at 2x the costs its
      returns were charged (DSR 0.919357 at 1x); its mean return reaches zero at
      7.66x those costs
NOT QUALIFIED
```

Every lookback in that sweep is the same strategy with a different random
seed: a true annualized Sharpe of about 0.44 (0.0275% a day after costs, on 1%
daily volatility). The best of the 40 shows 1.40, three times the truth, and it
is the one you would have traded. PBO is 0.55: the trial that looks best in one
half of the data lands below the median in the other half more often than not,
so picking by backtest here is a coin flip. DSR is 0.92: the best of 40 tries is
expected to look good, and 1.40 does not beat that expectation by enough. Gate 7
passed: at 16.3% volatility a −15% stop is hit inside a year 5.6% of the time.

When a gate fails on thin data rather than on a bad strategy,
`observations_required_for_dsr()` answers "how many more" instead of "not
enough" — or reports that no sample size rescues a Sharpe already below the
expected maximum of N noise trials, which is a different problem and needs a
different strategy rather than more patience.

`effective_trials` has no default on purpose. Trials from one sweep are
correlated, so the honest count is below the registered N, and nobody agrees
how to find it. Passing the registered N is the conservative choice: it can
only reject, never wave something through. The library will not guess for you,
because a guess dressed as a computation is the failure mode this whole package
exists to prevent.

## From a freqtrade hyperopt, one command

If you optimize with freqtrade, the registry already exists: every epoch
hyperopt evaluated is in `user_data/hyperopt_results/*.fthypt`, including the
ones it did not show you.

```bash
trialgate import-freqtrade user_data/hyperopt_results --registry state/registry.jsonl
```

That registers each epoch as a trial with its daily returns (every calendar day
of the backtest, compounding back to freqtrade's own `profit_total`), the fee
each trade paid, booked on the day it closed, and its parameters, and prints
the command that judges freqtrade's pick. Importing again registers only new
epochs.

Here it is on a real hyperopt file that ships in this repository, so you can
run it from a clone with `trialgate import-freqtrade tests/fixtures/freqtrade
--registry state/registry.jsonl`: freqtrade's bundled `SampleStrategy`, 4h
candles on BTC/USDT for the first half of 2024, 8 epochs. freqtrade's pick won
13 of its 14 trades:

```
registered 8 new trials (1-8) into state/registry.jsonl from 1 file(s); 0 were already registered
freqtrade's own pick (lowest loss) is trial 4, strategy_SampleStrategy_2026-10-20_18-45-34.fthypt#4. Judge it with:
  trialgate evaluate state/registry.jsonl --trial 4 --periods-per-year 365 --effective-trials 8 --strategy SampleStrategy

FAIL  data floor (gate 2): 14 (needs >= 1000)  -- only 14 of 183 periods had a
      return; a period that returned exactly zero is not evidence. 986 more needed
FAIL  PBO (gate 3): 1 (needs <= 0.05)
FAIL  DSR (gate 4): 0.357357 (needs >= 0.95)
PASS  declared costs (gate 8): 8 (needs >= 8)
FAIL  cost stress (gate 9): 0.254494 (needs >= 0.95)  -- at 2x the costs its
      returns were charged (DSR 0.357357 at 1x); its mean return reaches zero at
      2.06x those costs
NOT QUALIFIED
```

Thirteen wins out of fourteen are fourteen data points. A PBO of 1 says that
in every way of splitting the half-year in two, the epoch that looks best in
one half is at or below the median in the other. And the 1.3% it kept is about
what it paid in fees: charge them a second time and the profit is gone.
`evaluate` exits 1 when the trial is not qualified, so it can sit in front of a
deploy script.

Gate 2 is the one freqtrade users meet first. freqtrade books a trade's profit
on the day it closes, so a backtest has one observation per day with a close,
however long its range: clearing 1,000 takes years of history or a strategy
that closes trades most days.

With Optuna, register every trial as the study runs. trialgate does not import
Optuna; your objective stores the returns, and what execution cost in each
period, on the trial (`my_backtest()` as in the quickstart):

```python
from datetime import UTC, datetime

import optuna
from trialgate import OptunaRecorder, sharpe_ratio


def objective(trial):
    returns, costs = my_backtest(trial.suggest_int("lookback", 5, 200))
    trial.set_user_attr("period_returns", returns)  # net of costs
    trial.set_user_attr("period_costs", costs)  # for gate 9
    return sharpe_ratio(returns, periods_per_year=252)


recorder = OptunaRecorder(
    "state/registry.jsonl",
    strategy_id="daily_trend",
    universe=("BTCUSDT",),
    data_start=datetime(2019, 1, 1, tzinfo=UTC),
    data_end=datetime(2025, 6, 30, tzinfo=UTC),
    cost_assumptions={"round_trip_bps": "25"},
)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[recorder])
```

trialgate numbers trials from 1 in the order they reach the registry and
Optuna numbers them from 0, so in a fresh registry Optuna's trial n is
trialgate's n + 1; each record's `metrics["capture_source"]` names the Optuna
trial it came from. Pruned trials are registered too, because a run you stopped
early was still a try. They count in N, but without a full return series they
keep gate 3 from building its matrix: attach the series before pruning, or do
not prune backtests. Failed trials produced nothing to select and are not
registered. `import_optuna_study()` backfills a study that already ran.

## Going live: write the stop down first

Money is lost after the backtest, not in it. The strategy goes live, runs below
what the backtest promised, and keeps running because nobody decided in advance
how far below was too far. Gate 6 decides it before the first live period. On
the quickstart's registry:

```bash
trialgate live-lock state/registry.jsonl --trial 6 --horizon 90 --out state/live_bands.json
```

```
LOCKED  gate 6 for trial 6 -> state/live_bands.json
        over 90 live periods, STOP if cumulative return falls below -17.67%,
        drawdown exceeds 21.25%, or 13 losing periods run in a row (tighter bands
        apply earlier; see the file)
        false alarm on a strategy that is what its backtest says: 5.0% (target 5%)
        the same returns with no edge are stopped 12% of the time; a longer
        horizon raises that
```

The bands come from block-bootstrapping the trial's own registered returns,
one bound per period for each of the three statistics, calibrated jointly so
the whole horizon carries one 5% false-alarm budget rather than 5% per period.
The lock file refuses to be rewritten with different arguments, and
`--registry` on the check recomputes the bands to confirm nobody edited them.
Then feed it the live or paper returns as they arrive (a JSON list, one number
per line, or a CSV whose last column is the return):

```bash
trialgate live-check state/live_bands.json live_returns.txt --registry state/registry.jsonl
```

```
PENDING  live deviation (gate 6): 30 of 90 periods inside the locked bands so far
         (cumulative +2.14%, drawdown 4.56%, longest losing run 4)
```

A series averaging −0.8% a day (a bug, or costs far above the model) was
stopped on day 10, and the command exits 1:

```
STOP  live deviation (gate 6): period 10 of 90, cumulative return -8.82% < -8.68%
      a strategy that is what its backtest says crosses a band 5% of the time over
      the horizon; this is the condition written down before going live. Stop
      trading it.
```

The last line of the lock is the honest one. If the edge was luck, the live
returns look like the backtest's with the mean removed, and over 90 days this
lock stops that only 12% of the time: a quarter of live data cannot tell this
strategy from no strategy, and the lock says so up front instead of letting a
PASS imply more. What gate 6 does catch quickly is a live series that does not
look like the backtest at all: look-ahead and other bugs, fills and costs worse
than modeled. `horizon` has no default for the same reason the stop line has
none.

## Why the formula alone is not enough

The mathematics that corrects for this was published in 2014 and 2015. It is
not secret, it is not hard to implement, and it is not what this library
contributes. It takes one input — how many times you tried — and **every
implementation of it takes that number on trust.**

You will get that number wrong, and not dishonestly. The overnight grid search
was 2,000 runs. The tweaking last week was another 30. The five variants you
"just quickly checked" left no trace at all. By morning the honest recollection
is "about twenty", the arithmetic downstream is flawless, and the verdict is
worthless.

trialgate makes N a file instead of a memory. Every backtest appends a line.
The file is hash-chained, so deleting the runs that embarrassed you breaks it
and anyone can check with one command. And every gate downstream reads only
from that file — you never hand it a matrix of results, so there is no set of
columns left for you to curate.

## What it will not do

Stated up front, because a gate that oversells itself is the failure mode it
exists to prevent.

- **It cannot make you register everything.** `import-freqtrade` and the
  Optuna callback capture every run the optimizer made, so for those N no
  longer depends on memory. A notebook, another engine or a quick check run
  outside them still depends on you. The chain proves the log was not
  rewritten; it cannot prove the log is complete.
- **It does not backtest.** Bring your own engine, data, and return series.
  This is a judge, not a player; it never touches an exchange or a price feed.
- **Passing is necessary, not sufficient.** These gates remove strategies that
  are probably luck. They cannot promise the market will keep behaving.
- **It does not look for look-ahead bias.** A backtest that reads tomorrow's
  candle is a bug, not luck, and the engine is the place to find it (freqtrade
  ships `lookahead-analysis`). What trialgate catches is the consequence: live
  returns that do not look like the backtest, which is what gate 6 stops.
- **It does not cap how much of the profit came from a few days.** On
  simulated strategies whose edge is real by construction (1,000 independent
  days at an annual Sharpe of 1.0), the best five days are a median 28% of the
  profit, and removing the best 21 (2.1% of days) erases it. A fixed threshold
  on that rejects good strategies for being volatile; the DSR's skew and
  kurtosis terms and PBO's time blocks already price a profit that sits in a
  few periods.

## The nine gates

Built from the anti-overfitting literature (Bailey & López de Prado, 2014;
Bailey, Borwein, López de Prado & Zhu, 2015; Arnott, Harvey & Markowitz, 2019)
the circular block bootstrap for gate 6 (Politis & Romano, 1992) and, for
gate 7, the first-passage identity for Brownian motion with drift:

| Gate | What it enforces | Module |
|------|------------------|--------|
| 1. Trial registry | Every backtest of every variant is recorded in a tamper-evident append-only chain; unregistered results are void. The registry maintains the honest trial count N. | `trialgate.registry` |
| 2. Data floor | Enough observations (default ≥ 1,000 daily) spanning real regimes. Only periods with a return count: a day spent flat in cash is not evidence. | `trialgate.gates` |
| 3. PBO ≤ 0.05 | Probability of Backtest Overfitting via CSCV — all C(S, S/2) symmetric train/test splits, over a matrix assembled from the registry rather than handed in. | `trialgate.validation` |
| 4. DSR ≥ 0.95 | Deflated Sharpe Ratio — the observed Sharpe must beat the expected maximum of N noise trials, adjusted for skew, kurtosis, and sample length. | `trialgate.validation` |
| 5. Single-use holdout | The most recent data slice is locked at first backtest; spending it is a one-way, logged event. Iterated out-of-sample is not out-of-sample. | `trialgate.holdout` |
| 6. Live deviation | Before going live, lock bands for cumulative return, drawdown from peak and the longest losing run, bootstrapped from the candidate's own registered returns with a joint false-alarm rate (default 5%). Live or paper returns that leave them mean STOP. The lock also states how often a strategy with no edge at all would stay inside, so a PASS is not overread. | `trialgate.live` |
| 7. Stop-line survivability | The probability that the strategy's own volatility trips the operator's hard stop, from an exact first-passage formula rather than a simulation. Reports the volatility to size down to. Off unless you supply a stop-loss policy — defaulting one would be inventing your risk appetite. | `trialgate.risk` |
| 8. Declared costs | Every trial in the registry priced execution above zero. A missing key and an explicit `"0"` are the same verdict; only one of them looks like an oversight. | `trialgate.gates` |
| 9. Cost stress | The candidate's registered per-period costs are charged again (2× in total by default) and the DSR of what is left must still clear gate 4's bar. Reports the multiple of those costs at which the mean return reaches zero. A trial registered without its costs fails rather than being assumed cheap. | `trialgate.gates` |

Gate 7 exists because a strategy can clear 2–4 and still be untradeable. A
*genuine* annualized Sharpe of 1.0 run at 15% volatility breaches a −15% stop
inside a year 8.1% of the time; at Sharpe 0.5 that becomes 16.7%. The signal is
fine in the second case — the position size and the stop line were chosen by
different people on different days and never checked against each other, and
the shutdown looks exactly like the signal being wrong. When it fails, the
detail says what volatility would survive and the position size that gets
there (`size x0.53`, say).

That gate measures loss from starting equity, not drawdown from a running
peak. The two differ by roughly 3× (23% against 8.1% in the case above), so
quoting one for the other is the easy way to be confidently wrong. The
from-peak variant is absent rather than approximated: the cheap route through
Lehoczky's expected first-passage time overstated by 3× in the low-probability
region when measured against simulation, and a gate is the wrong place for a
number that is wrong exactly where it matters.

Gate 9 exists because one basis point satisfies gate 8. A backtest charges the
fee its author typed in; the costs that are hard to know — the spread, the
slippage, the fill a tick later — are the ones left out (freqtrade's
backtester, for one, fills every order at the requested price). Paying the
recorded costs twice is the default allowance for them. An edge several times
its costs survives that. An edge the size of its costs, which a sweep of
hundreds of configurations finds easily, does not: in the tests, a candidate
netting 0.1% a day after 0.1% of costs passes gate 4 at a DSR of 0.98 and
fails gate 9 at 0.12.

## Serially correlated returns

The DSR's standard error and the usual √p annualization both assume one
period's return says nothing about the next. Smoothed marks, illiquid holdings,
overlapping signals and trend filters break that, almost always in the
flattering direction: with lag-1 autocorrelation 0.5 the IID standard error of
the Sharpe ratio is √3 too small, and about one zero-edge strategy in six clears
a 0.95 DSR instead of one in twenty.

trialgate estimates the long-run variance of the Sharpe estimator (Lo 2002,
non-IID case, Eq. (14) p. 39 and Appendix A Eqs. (A3), (A7)-(A11) pp. 46-47;
AR(1)-prewhitened Bartlett kernel, Andrews & Monahan 1992, where Lo uses
Newey-West at a fixed lag, Eqs. (A12)-(A14) p. 47) and recomputes the DSR on
the equivalent number of independent observations:

```
z_adj = (SR - SR_0) * sqrt((T - 1) / ratio) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2)
T_eff = 1 + (T - 1) / ratio,   ratio = long-run / IID variance of SR-hat
```

**This adjustment is trialgate's extension, not Bailey & López de Prado's
method.** The DSR (2014, Eq. (2) p. 8) and the PSR (2012, Eq. (11) p. 9) use
the IID non-normal Sharpe variance the 2012 paper takes from Mertens (2002)
(Eq. (8) p. 8). The DSR
paper does not discuss serial correlation. The PSR paper states that it does
not "explicitly address the case of serially-conditioned processes" (p. 4)
and takes the Mertens variance to remain valid for stationary, ergodic
returns, citing Christie (2005) and Opdyke (2007) (pp. 4, 8). trialgate does
not rely on that reading: Lo's GMM variance carries the lag covariances the
Mertens term drops, and on AR(1) returns the IID standard error misses the
Monte Carlo spread of the estimator by up to 3× (tests/test_autocorrelation.py).
What is borrowed from the papers is each piece — Lo's variance, Bailey &
López de Prado's statistic — not their combination.

`lo_annualized_sharpe()` annualizes with Lo's `eta(q) = q / sqrt(q + 2 *
sum_{k<q} (q - k) rho_k)` (Eq. (20) p. 40) in place of √q. Lo's estimator
uses all q - 1 sample autocorrelations (p. 42); trialgate defaults to the
Andrews (1991) bandwidth and treats later lags as zero, which is its own
choice (pass `max_lag=periods_per_year - 1` for Lo's). Every cell of Lo's
AR(1) Table 2 (p. 41) is reproduced to the printed two decimals
(tests/test_lo_2002_crosscheck.py). Page numbers for Lo are the Financial
Analysts Journal's; for Bailey & López de Prado, the authors' preprints (PSR:
July 2012; DSR: July 31, 2014).

| `autocorrelation=` | Gate 4 judges | Report |
|---|---|---|
| `"report"` (default) | the IID DSR, exactly as before | a warning with the adjusted DSR, effective sample size and η(q) Sharpe when Ljung-Box finds the correlation significant (p < 0.05) |
| `"adjust"` | the adjusted DSR | the IID value alongside |
| `"ignore"` | the IID DSR | nothing; the estimate is skipped |

The default leaves verdicts where they were on purpose: a significance test
fires on 5% of IID series by construction, and a gate whose verdict changes
between releases, or between two runs of the same strategy on either side of
p = 0.05, is a gate nobody can reason about. The warning says what the adjusted
verdict would be; `"adjust"` makes it binding.

## How the registry is made auditable

Each record carries `prev_hash`: the SHA-256 over every preceding line of the
file, exactly as written on disk.

```json
{"trial_id": 3, "operator_note": "lookback sweep", "prev_hash": "5c20ef7d4e3f...", ...}
```

Delete the embarrassing trial, edit a Sharpe, reorder the log — the next
record's `prev_hash` stops matching, `load_trials()` raises
`RegistryTamperError`, and `append_trial()` refuses to extend the log rather
than launder it. Anyone can check this without reading your code. On the
quickstart's registry:

```bash
trialgate verify state/registry.jsonl
```

```
OK    state/registry.jsonl
      40 trials, sealed against edits: 1-39
      head: dd6ec370f6b46c91652e61e7682d37c1efdfd7110563af305020f4155fd1f8a8
      trials 40-40 are still editable and truncation is undetectable; commit this
      file after every run
```

Exit status is 0 when intact and 1 when not (a path with no file counts as
not), so it drops into CI or a pre-commit hook. After deleting trial 2:

```
FAIL  state/registry.jsonl
      state/registry.jsonl:2 breaks the hash chain: trial 3 was appended after
      a history hashing to 99aadc9c735f..., but the preceding 1 line(s) hash to
      0f870733749e.... Earlier trials have been edited, reordered, or removed.
      Trial count and every gate verdict derived from this registry are void.
```

**What this proves:** no trial before the last anchored record was edited,
reordered, or removed since it was written.

**What it does not prove**, beyond the limits listed above: a local file
has no external anchor, so dropping the last k records leaves a shorter but
internally consistent chain. Committing the registry after every run closes
that and costs nothing.

Gate 3 needs a T×N matrix of per-period returns, one column per configuration,
and it never takes one from you: pass `period_returns=` when registering a
trial, and `performance_matrix()` assembles the CSCV input from the registry
itself, one column per registered trial. Leave a trial's returns out and it
refuses to build anything:

```
1 of 51 registered trials have no stored period_returns (trial ids [51]); a
matrix built from the rest would be a flattering subset of the registry, which
voids gate 3.
```

Each series is hashed into its chained record, so editing one after the fact is
detected the same way editing a record is.

Existing registries need no migration. Pre-chain records carry no `prev_hash`
but are still hashed into the chain, so **the first chained append seals the
entire legacy history behind it**. Rewriting old records to add hashes
retroactively would itself be the tampering this gate detects.

## The gate's job is to say no

This library was extracted from a daily crypto signal system, and the method
it enforces — write the pass line down before the first trial, then keep the
verdict either way — has already produced a FAIL on record:

- **Where it came from** —
  [crypto-quant-signal](https://github.com/0Smallcat0/crypto-quant-signal):
  the daily crypto trend-signal system (paper scoreboard, manual execution,
  no exchange keys) whose validation code became this library. Its
  single-use holdout is still sealed. It ships a
  [two-minute offline demo](https://github.com/0Smallcat0/crypto-quant-signal#try-it-in-two-minutes)
  (`python -m scripts.run_demo` — no keys, no network) if you want to see
  the system these gates were built for.
- **A registered FAIL** —
  [tw-stock-trading](https://github.com/0Smallcat0/tw-stock-trading): the
  same strategy family adapted to the Taiwan 0050 ETF **failed its
  pre-registered claim** (CAGR trailed dividend-included buy-and-hold by
  5.3 pp/year over 21 years; even the zero-cost upper bound barely reached
  the tolerance line). Per the pre-registered rule, the runtime was never
  built. The FAIL report *is* the product.
  Full story: [docs/rejecting-my-own-strategy.md](docs/rejecting-my-own-strategy.md).

## Design notes

- **Zero dependencies.** Standard library only; frozen, slotted dataclasses;
  ships `py.typed` and passes `mypy --strict`.
- **UTC or it didn't happen.** Naive datetimes are rejected everywhere.
- **Raises instead of clamping.** When the DSR variance approximation breaks
  down, you get an exception, not a confident 0.0/1.0.
- **Append-only state.** The registry is a hash-chained JSONL log; the holdout
  lock is a one-way JSON file. Both are plain text you can audit in any editor,
  and the registry is tamper-evident against anyone who edits it there.

## Provenance

The gate specification, thresholds, and doctrine were defined and verified by
the author from the primary literature; implementation was AI-assisted
(spec-driven development with Claude), with every module reviewed against the
spec and covered by tests before acceptance. This mirrors the library's own
thesis: artifacts earn trust through verification, not through how they were
produced.

## References

- Bailey & López de Prado — *The Deflated Sharpe Ratio* (Journal of Portfolio
  Management 40(5), 2014); DSR Eq. (2), p. 8 of the July 31, 2014 preprint
- Bailey, Borwein, López de Prado & Zhu — *The Probability of Backtest
  Overfitting* (Journal of Computational Finance, 2015)
- Arnott, Harvey & Markowitz — *A Backtesting Protocol in the Era of Machine
  Learning* (Journal of Financial Data Science, 2019)
- Lo — *The Statistics of Sharpe Ratios* (Financial Analysts Journal 58(4),
  2002, pp. 36–52); η(q) Eqs. (20) p. 40 and (22) p. 41, GMM variance Eq. (14)
  p. 39, Appendix A pp. 46–50
- Bailey & López de Prado — *The Sharpe Ratio Efficient Frontier* (Journal of
  Risk 15(2), 2012); Sharpe variance Eq. (8) p. 8 and PSR Eq. (11) p. 9 of the
  July 2012 preprint
- Andrews — *Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix Estimation* (Econometrica, 1991); Andrews & Monahan — *An Improved
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
  Estimator* (Econometrica, 1992)
- Politis & Romano — *A Circular Block-Resampling Procedure for Stationary
  Data* (in *Exploring the Limits of Bootstrap*, Wiley, 1992); Hall, Horowitz
  & Jing — *On Blocking Rules for the Bootstrap with Dependent Data*
  (Biometrika 82(3), 1995), whose variance-estimation rate T^(1/3) sets the
  default block length

## License

[MIT](LICENSE) © Tsai Chih-Chun (0Smallcat0)
