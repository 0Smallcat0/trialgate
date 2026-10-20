# trialgate

[![CI](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml/badge.svg)](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trialgate.svg)](https://pypi.org/project/trialgate/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
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

PASS  data floor (gate 2): 2000 >= 1000
PASS  PBO (gate 3): 0.0418803 <= 0.05
FAIL  DSR (gate 4): 0.603118 >= 0.95
PASS  stop-line survivability (gate 7): 1.10249e-15 <= 0.1
PASS  declared costs (gate 8): 450 >= 450
NOT QUALIFIED
```

Sharpe 1.20 is a number people quit their jobs over. There is nothing in that
data. Note which gate caught it: PBO passed, comfortably. One gate clearing is
not the same as a strategy clearing, which is why there is more than one. The
demo's source is also the integration example — swap its price series and its
backtest for yours and the rest is unchanged.

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

- **It cannot make you register anything.** Nothing watches your process. The
  chain proves the log was not rewritten; it cannot prove the log is complete.
  You can always run a backtest outside it and never tell anyone.
- **It does not backtest.** Bring your own engine, data, and return series.
  This is a judge, not a player.
- **Passing is necessary, not sufficient.** In a verified 2022 case study the
  least-overfit strategy in the set still lost ~35% in two months. These gates
  remove strategies that are probably luck. They cannot promise the market will
  keep behaving.
- **One number is still yours to choose.** Trials from a nested grid are
  correlated, so the honest count is below the registered N and nobody agrees
  how to find it. The library refuses to guess and makes you type it.

## The eight gates

Built from the anti-overfitting literature (Bailey & López de Prado, 2014;
Bailey, Borwein, López de Prado & Zhu, 2015; Arnott, Harvey & Markowitz, 2019)
and, for gate 7, the first-passage identity for Brownian motion with drift:

| Gate | What it enforces | Module |
|------|------------------|--------|
| 1. Trial registry | Every backtest of every variant is recorded in a tamper-evident append-only chain; unregistered results are void. The registry maintains the honest trial count N. | `trialgate.registry` |
| 2. Data floor | Enough observations (default ≥ 1,000 daily) spanning real regimes. | `trialgate.gates` |
| 3. PBO ≤ 0.05 | Probability of Backtest Overfitting via CSCV — all C(S, S/2) symmetric train/test splits, over a matrix assembled from the registry rather than handed in. | `trialgate.validation` |
| 4. DSR ≥ 0.95 | Deflated Sharpe Ratio — the observed Sharpe must beat the expected maximum of N noise trials, adjusted for skew, kurtosis, and sample length. | `trialgate.validation` |
| 5. Single-use holdout | The most recent data slice is locked at first backtest; spending it is a one-way, logged event. Iterated out-of-sample is not out-of-sample. | `trialgate.holdout` |
| 6. Paper trading | ≥ 3 months live paper execution with measured real costs. Procedural — deliberately **not** in this library: it is a measurement, not a computation. | — |
| 7. Stop-line survivability | The probability that the strategy's own volatility trips the operator's hard stop, from an exact first-passage formula rather than a simulation. Reports the volatility to size down to. Off unless you supply a stop-loss policy — defaulting one would be inventing your risk appetite. | `trialgate.risk` |
| 8. Declared costs | Every trial in the registry priced execution above zero. A missing key and an explicit `"0"` are the same verdict; only one of them looks like an oversight. | `trialgate.gates` |

Gate 7 exists because a strategy can clear 2–4 and still be untradeable. A
*genuine* annualized Sharpe of 1.0 run at 15% volatility breaches a −15% stop
inside a year 8.1% of the time; at Sharpe 0.5 that becomes 16.7%. The signal is
fine in the second case — the position size and the stop line were chosen by
different people on different days and never checked against each other, and
the shutdown looks exactly like the signal being wrong.

That gate measures loss from starting equity, not drawdown from a running
peak. The two differ by roughly 3× (23% against 8.1% in the case above), so
quoting one for the other is the easy way to be confidently wrong. The
from-peak variant is absent rather than approximated: the cheap route through
Lehoczky's expected first-passage time overstated by 3× in the low-probability
region when measured against simulation, and a gate is the wrong place for a
number that is wrong exactly where it matters.

Zero dependencies, pure standard library, `mypy --strict`, Python 3.11+.

## Quickstart

```python
from datetime import UTC, datetime

from trialgate import append_trial, evaluate_registry, initialize_holdout

NOW = datetime.now(tz=UTC)

# Gate 5 — lock the most recent ~12 months BEFORE the first backtest.
initialize_holdout(
    "state/holdout.json",
    holdout_start=datetime(2025, 7, 1, tzinfo=UTC),
    locked_at=NOW,
)

# Gate 1 — register EVERY backtest execution, including the quick ones.
# Passing holdout= enforces gate 5 here: a run reaching into the locked
# period raises instead of being silently recorded.
for lookback in candidate_lookbacks:
    append_trial(
        "state/registry.jsonl",
        recorded_at=NOW,
        code_version="8b91661",
        strategy_id="daily_trend_ensemble",
        parameters={"lookback": str(lookback)},
        universe=("BTCUSDT", "ETHUSDT"),
        data_start=datetime(2019, 1, 1, tzinfo=UTC),
        data_end=datetime(2025, 6, 30, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "25"},
        metrics={"annualized_sharpe": f"{my_backtest_sharpe(lookback):.4f}"},
        operator_note="lookback sweep",
        holdout="state/holdout.json",
        # The return series is what makes this trial usable by gate 3.
        period_returns=my_backtest_returns(lookback),
    )

# Gates 2-4 in one call, over data that only came from the registry.
report = evaluate_registry(
    "state/registry.jsonl",
    candidate_trial_id=37,
    periods_per_year=252,
    effective_trials=450,  # required: see below
    strategy_id="daily_trend_ensemble",
    stop_loss_fraction=0.15,  # your hard stop; omitting it skips gate 7
)
print(report.summary())
# PASS  data floor (gate 2): 1848 >= 1000
# PASS  PBO (gate 3): 0.02 <= 0.05
# PASS  DSR (gate 4): 0.97 >= 0.95
# FAIL  stop-line survivability (gate 7): 0.358932 <= 0.1  -- ran at 24.0% annualized
#       volatility against a -15% stop; survivable volatility is 12.7% (size x0.53)
# PASS  declared costs (gate 8): 450 >= 450
# NOT QUALIFIED
```

Gate 7 is the one that fails with an instruction attached. The edge is real —
gates 2-4 all cleared — but a Sharpe-0.55 strategy run at 24% volatility
breaches a −15% stop 36% of the time in a year, so the account gets shut down
before the edge pays. `size x0.53` is what to do about it, not a suggestion to
think harder.

When a gate fails on thin data rather than on a bad strategy,
`observations_required_for_dsr()` answers "how many more" instead of "not
enough" — or reports that no sample size rescues a Sharpe already below the
expected maximum of N noise trials, which is a different problem and needs a
different strategy rather than more patience.

`my_backtest_returns()` is yours — `trialgate` judges, it does not backtest.

`effective_trials` has no default on purpose. Passing the registered N is the
conservative choice: it can only reject, never wave something through. The
library will not guess for you, because a guess dressed as a computation is
the failure mode this whole package exists to prevent.

## How the registry is made auditable

Each record carries `prev_hash`: the SHA-256 over every preceding line of the
file, exactly as written on disk.

```json
{"trial_id": 3, "operator_note": "lookback sweep", "prev_hash": "5c20ef7d4e3f...", ...}
```

Delete the embarrassing trial, edit a Sharpe, reorder the log — the next
record's `prev_hash` stops matching, `load_trials()` raises
`RegistryTamperError`, and `append_trial()` refuses to extend the log rather
than launder it. Anyone can check this without reading your code:

```bash
trialgate verify state/registry.jsonl
```

```
OK    state/registry.jsonl
      3 trials, sealed against edits: 1-2
      head: 53a92f32906489760e1236eab9ac0d042276db13c6b3326c6a3d60db1b873cf7
      trials 3-3 are still editable and truncation is undetectable; commit this
      file after every run
```

Exit status is 0 when intact and 1 when not, so it drops into CI or a
pre-commit hook. After deleting trial 2:

```
FAIL  state/registry.jsonl
      state/registry.jsonl:2 breaks the hash chain: trial 3 was appended after
      a history hashing to 5c20ef7d4e3f..., but the preceding 1 line(s) hash to
      e05ded93f917.... Earlier trials have been edited, reordered, or removed.
      Trial count and every gate verdict derived from this registry are void.
```

**What this proves:** no trial before the last anchored record was edited,
reordered, or removed since it was written.

**What it does not prove**, beyond the limits listed at the top: a local file
has no external anchor, so dropping the last k records leaves a shorter but
internally consistent chain. Committing the registry after every run closes
that and costs nothing.

The same reasoning applies one level down. Gate 3 needs a T×N matrix of per-period
returns, one column per configuration — and a matrix you hand in is a matrix
you can curate. So you don't hand it in: pass `period_returns=` when
registering a trial, and `performance_matrix()` assembles the CSCV input from
the registry itself, one column per registered trial. Leave a trial's returns
out and it refuses to build anything:

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

This library was extracted from a daily crypto signal system, and the same
gates already have a registered FAIL on record:

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
- **Judgement, not orchestration.** Bring your own backtest engine and data;
  the library never touches an exchange or a price feed.

## Provenance

The gate specification, thresholds, and doctrine were defined and verified by
the author from the primary literature; implementation was AI-assisted
(spec-driven development with Claude), with every module reviewed against the
spec and covered by tests before acceptance. This mirrors the library's own
thesis: artifacts earn trust through verification, not through how they were
produced.

## References

- Bailey & López de Prado — *The Deflated Sharpe Ratio* (Journal of Portfolio
  Management, 2014)
- Bailey, Borwein, López de Prado & Zhu — *The Probability of Backtest
  Overfitting* (Journal of Computational Finance, 2015)
- Arnott, Harvey & Markowitz — *A Backtesting Protocol in the Era of Machine
  Learning* (Journal of Financial Data Science, 2019)

## License

[MIT](LICENSE) © Tsai Chih-Chun (0Smallcat0)
