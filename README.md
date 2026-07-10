# trialgate

[![CI](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml/badge.svg)](https://github.com/0Smallcat0/trialgate/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An **anti-overfitting validation gate** for backtested trading strategies.
Zero dependencies, pure standard library, fully typed.

> A strategy earns belief by surviving verification, not by looking good
> in-sample. Optimizing four parameters on a **pure random walk** can produce
> an in-sample Sharpe of 1.27. A single backtest — however pretty — is not
> evidence.

`trialgate` packages the mechanical parts of a six-gate qualification
standard, built from the anti-overfitting literature (Bailey & López de
Prado, 2014; Bailey, Borwein, López de Prado & Zhu, 2015; Arnott, Harvey &
Markowitz, 2019):

| Gate | What it enforces | Module |
|------|------------------|--------|
| 1. Trial registry | Every backtest of every variant is recorded append-only; unregistered results are void. The registry maintains the honest trial count N. | `trialgate.registry` |
| 2. Data floor | Enough observations (default ≥ 1,000 daily) spanning real regimes. | `trialgate.gates` |
| 3. PBO ≤ 0.05 | Probability of Backtest Overfitting via CSCV — all C(S, S/2) symmetric train/test splits. | `trialgate.validation` |
| 4. DSR ≥ 0.95 | Deflated Sharpe Ratio — the observed Sharpe must beat the expected maximum of N noise trials, adjusted for skew, kurtosis, and sample length. | `trialgate.validation` |
| 5. Single-use holdout | The most recent data slice is locked at first backtest; spending it is a one-way, logged event. Iterated out-of-sample is not out-of-sample. | `trialgate.holdout` |
| 6. Paper trading | ≥ 3 months live paper execution with measured real costs. Procedural — deliberately **not** in this library: it is a measurement, not a computation. | — |

Passing all six is a **necessary condition, not a profit guarantee**: in a
verified 2022 case study, the least-overfit strategy still lost ~35% in two
months.

## Install

```bash
pip install trialgate
```

## Quickstart

```python
from datetime import UTC, datetime

from trialgate import (
    append_trial, config_hash_for, load_trials, trial_count,
    initialize_holdout, require_data_allowed,
    deflated_sharpe_ratio, probability_of_backtest_overfitting,
    non_annualized_sharpe_variance, evaluate_gates,
)

NOW = datetime.now(tz=UTC)

# Gate 5 — lock the most recent ~12 months BEFORE the first backtest.
initialize_holdout(
    "state/holdout.json",
    holdout_start=datetime(2025, 7, 1, tzinfo=UTC),
    locked_at=NOW,
)
# Every regular run must prove it stays out of the holdout.
require_data_allowed("state/holdout.json", data_end=datetime(2025, 6, 30, tzinfo=UTC))

# Gate 1 — register EVERY backtest execution, including the quick ones.
append_trial(
    "state/registry.jsonl",
    recorded_at=NOW,
    config_hash=config_hash_for({"lookbacks": [20, 65, 150, 200]}),
    code_version="8b91661",
    strategy_id="daily_trend_ensemble",
    parameters={"lookbacks": "20,65,150,200"},
    universe=("BTCUSDT", "ETHUSDT"),
    data_start=datetime(2019, 1, 1, tzinfo=UTC),
    data_end=datetime(2025, 6, 30, tzinfo=UTC),
    cost_assumptions={"round_trip_bps": "25"},
    metrics={"annualized_sharpe": "1.10"},
    operator_note="baseline",
)

# Gates 3 + 4 — computed over ALL registered trials, never a flattering subset.
trials = load_trials("state/registry.jsonl")
annualized = [float(t.metrics["annualized_sharpe"]) for t in trials]

pbo = probability_of_backtest_overfitting(performance_matrix, block_count=16)
dsr = deflated_sharpe_ratio(
    candidate_daily_returns,
    trial_sharpe_variance=non_annualized_sharpe_variance(annualized),
    effective_trials=trial_count("state/registry.jsonl"),
)

# Gates 2-4 combined verdict.
report = evaluate_gates(
    observations=len(candidate_daily_returns),
    pbo=pbo.pbo,
    deflated_sharpe=dsr.deflated_sharpe_ratio,
)
print(report.summary())
# PASS  data floor (gate 2): 1848 >= 1000
# PASS  PBO (gate 3): 0.02 <= 0.05
# PASS  DSR (gate 4): 0.97 >= 0.95
# QUALIFIED
```

`performance_matrix` is a T×N list of per-period returns, one column per
registered configuration; `candidate_daily_returns` is the candidate's daily
return series. Both come from your backtest engine — `trialgate` judges, it
does not backtest.

## The gate's job is to say no

This library is extracted from a live system and has one verdict of each kind
on record:

- **Reference deployment (in progress)** —
  [crypto-quant-signal](https://github.com/0Smallcat0/crypto-quant-signal):
  a daily crypto trend-signal system currently spending its locked holdout
  and 90-day paper period against these exact gates.
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
- **Append-only state.** The registry is a JSONL log; the holdout lock is a
  one-way JSON file. Both are plain text you can audit in any editor.
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
