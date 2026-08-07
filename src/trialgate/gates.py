"""Threshold gates and the combined qualification report.

The six-gate qualification standard (numbering follows the reference
deployment, a live crypto signal system):

1. Trial registry (structural)      -> trialgate.registry
2. Data floor (statistical)         -> judged here
3. PBO <= 0.05 (statistical)        -> computed by trialgate.validation, judged here
4. DSR >= 0.95 (statistical)        -> computed by trialgate.validation, judged here
5. Single-use holdout (structural)  -> trialgate.holdout
6. Paper trading with measured real costs (procedural) -> out of library
   scope on purpose: it is a live measurement over months, not a computation.

Passing every gate is a NECESSARY condition for trusting a backtest, not a
profit guarantee: in a verified 2022 case study the least-overfit strategy
still lost ~35% in two months.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trialgate.registry import load_trials, performance_matrix
from trialgate.validation import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

DEFAULT_MIN_OBSERVATIONS = 1000
DEFAULT_MAX_PBO = 0.05
DEFAULT_MIN_DSR = 0.95


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One threshold verdict."""

    gate: str
    passed: bool
    observed: float
    threshold: float
    comparison: str

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status}  {self.gate}: {self.observed:g} {self.comparison} {self.threshold:g}"


@dataclass(frozen=True, slots=True)
class GateReport:
    """Combined verdict over the statistical gates (2-4)."""

    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        lines = [check.describe() for check in self.checks]
        lines.append("QUALIFIED" if self.passed else "NOT QUALIFIED")
        return "\n".join(lines)


def evaluate_gates(
    *,
    observations: int,
    pbo: float,
    deflated_sharpe: float,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    max_pbo: float = DEFAULT_MAX_PBO,
    min_dsr: float = DEFAULT_MIN_DSR,
) -> GateReport:
    """Judge the statistical gates (2-4) against their thresholds.

    Feed the honest numbers: ``observations`` from the actual data span,
    ``pbo`` from probability_of_backtest_overfitting() over ALL registered
    trials, ``deflated_sharpe`` from deflated_sharpe_ratio() using the
    registry's full trial count. Flattering inputs void the report.
    """

    return GateReport(
        checks=_checks(observations, pbo, deflated_sharpe, min_observations, max_pbo, min_dsr)
    )


def evaluate_registry(
    path: str | Path,
    *,
    candidate_trial_id: int,
    periods_per_year: int,
    effective_trials: int,
    strategy_id: str | None = None,
    block_count: int = 16,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    max_pbo: float = DEFAULT_MAX_PBO,
    min_dsr: float = DEFAULT_MIN_DSR,
) -> GateReport:
    """Judge gates 2-4 for one registered trial, using only registered data.

    Every input the operator could otherwise curate comes out of the registry:
    the CSCV matrix is assembled from all registered trials, the candidate's
    return series is a column of that matrix, and the trial Sharpe spread that
    the DSR deflates by is computed from those same columns rather than read
    from the self-reported ``metrics`` strings.

    ``effective_trials`` is required, with no default, because it is the last
    number the library cannot derive: trials from a nested parameter grid are
    correlated, so the honest count is somewhere below the registered N and
    nobody agrees on how to find it. Passing the registered N is the
    conservative choice — it can only reject, never wave something through.
    Whatever you pass, you had to type it.
    """

    trials = [
        trial
        for trial in load_trials(path)
        if strategy_id is None or trial.strategy_id == strategy_id
    ]
    trial_ids = [trial.trial_id for trial in trials]
    if candidate_trial_id not in trial_ids:
        msg = (
            f"trial {candidate_trial_id} is not registered"
            + (f" under strategy_id {strategy_id!r}" if strategy_id else "")
            + f"; registered ids are {trial_ids[:10]}{'...' if len(trial_ids) > 10 else ''}"
        )
        raise ValueError(msg)

    matrix = performance_matrix(path, strategy_id=strategy_id)
    candidate_column = trial_ids.index(candidate_trial_id)
    candidate_returns = [row[candidate_column] for row in matrix]
    trial_sharpes = [
        sharpe_ratio([row[column] for row in matrix], periods_per_year=periods_per_year)
        for column in range(len(matrix[0]))
    ]

    pbo = probability_of_backtest_overfitting(matrix, block_count=block_count)
    dsr = deflated_sharpe_ratio(
        candidate_returns,
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
        effective_trials=effective_trials,
    )
    return GateReport(
        checks=_checks(
            len(matrix),
            pbo.pbo,
            dsr.deflated_sharpe_ratio,
            min_observations,
            max_pbo,
            min_dsr,
        )
    )


def _checks(
    observations: int,
    pbo: float,
    deflated_sharpe: float,
    min_observations: int,
    max_pbo: float,
    min_dsr: float,
) -> tuple[GateCheck, ...]:
    return (
        GateCheck(
            gate="data floor (gate 2)",
            passed=observations >= min_observations,
            observed=float(observations),
            threshold=float(min_observations),
            comparison=">=",
        ),
        GateCheck(
            gate="PBO (gate 3)",
            passed=pbo <= max_pbo,
            observed=pbo,
            threshold=max_pbo,
            comparison="<=",
        ),
        GateCheck(
            gate="DSR (gate 4)",
            passed=deflated_sharpe >= min_dsr,
            observed=deflated_sharpe,
            threshold=min_dsr,
            comparison=">=",
        ),
    )
