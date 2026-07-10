"""Qualification-gate statistics: Deflated Sharpe Ratio and CSCV/PBO.

Sources:
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (JPM 2014).
- Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
  Overfitting" (Journal of Computational Finance 2015).

These statistics operate on measurement floats, not account money, so
float arithmetic is intentional here (unlike a Decimal trading path).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import combinations

_EULER_GAMMA = 0.5772156649015329


class ValidationInputError(ValueError):
    """Raised when gate statistics receive unusable inputs."""


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """DSR output with the inputs that produced it."""

    deflated_sharpe_ratio: float
    observed_sharpe: float
    expected_max_sharpe: float
    effective_trials: int
    observations: int


@dataclass(frozen=True, slots=True)
class PBOResult:
    """CSCV output: probability of backtest overfitting."""

    pbo: float
    combinations_evaluated: int
    block_count: int
    strategies: int
    observations: int


def sharpe_ratio(returns: list[float], *, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio of a periodic return series (risk-free = 0)."""

    if len(returns) < 2:
        return 0.0
    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        return 0.0
    return statistics.fmean(returns) / stdev * math.sqrt(periods_per_year)


def deflated_sharpe_ratio(
    returns: list[float],
    *,
    trial_sharpe_variance: float,
    effective_trials: int,
) -> DeflatedSharpeResult:
    """Probability that the true Sharpe exceeds the expected max of N noise trials.

    ``returns`` are the candidate strategy's periodic (e.g. daily) returns.
    ``trial_sharpe_variance`` is the variance of (non-annualized) Sharpe ratios
    across ALL registered trials; ``effective_trials`` is the
    correlation-adjusted trial count N. Pass the honest numbers from the trial
    registry — a flattering N voids the gate.
    """

    if effective_trials < 1:
        msg = "effective_trials must be at least 1"
        raise ValidationInputError(msg)
    if trial_sharpe_variance < 0.0:
        msg = "trial_sharpe_variance must not be negative"
        raise ValidationInputError(msg)
    observations = len(returns)
    if observations < 3:
        msg = "returns must contain at least 3 observations"
        raise ValidationInputError(msg)

    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        msg = "returns must not be constant"
        raise ValidationInputError(msg)
    observed_sharpe = statistics.fmean(returns) / stdev

    normal = statistics.NormalDist()
    if effective_trials == 1 or trial_sharpe_variance == 0.0:
        expected_max_sharpe = 0.0
    else:
        spread = math.sqrt(trial_sharpe_variance)
        expected_max_sharpe = spread * (
            (1.0 - _EULER_GAMMA) * normal.inv_cdf(1.0 - 1.0 / effective_trials)
            + _EULER_GAMMA * normal.inv_cdf(1.0 - 1.0 / (effective_trials * math.e))
        )

    skewness = _skewness(returns)
    kurtosis = _kurtosis(returns)
    mertens_term = 1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if mertens_term <= 0.0:
        # Fabricating certainty by clamping would answer 0.0/1.0 confidently
        # on inputs where the variance approximation has broken down.
        msg = (
            "returns are too extreme or too short for a stable DSR estimate "
            f"(Mertens variance term {mertens_term:.6f} <= 0)"
        )
        raise ValidationInputError(msg)
    denominator = math.sqrt(mertens_term)
    statistic = (
        (observed_sharpe - expected_max_sharpe) * math.sqrt(observations - 1.0)
    ) / denominator
    return DeflatedSharpeResult(
        deflated_sharpe_ratio=normal.cdf(statistic),
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max_sharpe,
        effective_trials=effective_trials,
        observations=observations,
    )


def probability_of_backtest_overfitting(
    performance_matrix: list[list[float]],
    *,
    block_count: int = 16,
) -> PBOResult:
    """CSCV estimate of PBO over a T x N trial performance matrix.

    ``performance_matrix``: T rows (periods) x N columns (one per registered
    trial/configuration), each cell the period return of that configuration.
    Splits the rows into ``block_count`` contiguous equal blocks, evaluates all
    C(S, S/2) symmetric train/test partitions, and returns the fraction where
    the in-sample-best configuration ranks in the bottom half out-of-sample.

    Rows beyond the largest multiple of ``block_count`` — the MOST RECENT
    ``T mod S`` observations — are dropped to keep blocks equal; compare
    ``result.observations`` against ``len(performance_matrix)`` to see how
    many were excluded.
    """

    if block_count < 2 or block_count % 2 != 0:
        msg = "block_count must be an even integer of at least 2"
        raise ValidationInputError(msg)
    observations = len(performance_matrix)
    if observations < block_count:
        msg = "performance_matrix must have at least block_count rows"
        raise ValidationInputError(msg)
    strategies = len(performance_matrix[0])
    if strategies < 2:
        msg = "performance_matrix must contain at least 2 strategy columns"
        raise ValidationInputError(msg)
    if any(len(row) != strategies for row in performance_matrix):
        msg = "performance_matrix rows must have equal length"
        raise ValidationInputError(msg)

    usable_rows = observations - (observations % block_count)
    block_size = usable_rows // block_count
    block_slices = [
        performance_matrix[block * block_size : (block + 1) * block_size]
        for block in range(block_count)
    ]

    overfit_count = 0
    combos = list(combinations(range(block_count), block_count // 2))
    for train_blocks in combos:
        train_set = frozenset(train_blocks)
        train_rows = [row for block in sorted(train_set) for row in block_slices[block]]
        test_rows = [
            row
            for block in range(block_count)
            if block not in train_set
            for row in block_slices[block]
        ]
        train_scores = [_column_sharpe(train_rows, column) for column in range(strategies)]
        test_scores = [_column_sharpe(test_rows, column) for column in range(strategies)]
        best_in_sample = max(range(strategies), key=lambda column: train_scores[column])
        rank = sum(
            1 for column in range(strategies) if test_scores[column] <= test_scores[best_in_sample]
        )
        relative_rank = rank / (strategies + 1)
        if relative_rank <= 0.5:
            overfit_count += 1

    return PBOResult(
        pbo=overfit_count / len(combos),
        combinations_evaluated=len(combos),
        block_count=block_count,
        strategies=strategies,
        observations=usable_rows,
    )


def _column_sharpe(rows: list[list[float]], column: int) -> float:
    values = [row[column] for row in rows]
    if len(values) < 2:
        return 0.0
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return 0.0
    return statistics.fmean(values) / stdev


def non_annualized_sharpe_variance(
    annualized_sharpes: list[float], *, periods_per_year: int = 365
) -> float:
    """De-annualize registry Sharpe values into the variance DSR expects.

    The trial registry stores ANNUALIZED Sharpe ratios; feeding their variance
    into deflated_sharpe_ratio() unconverted inflates the expected-max Sharpe
    by periods_per_year and false-rejects everything. Var(SR_ann / sqrt(p))
    equals Var(SR_ann) / p.
    """

    if len(annualized_sharpes) < 2:
        return 0.0
    return statistics.pvariance(annualized_sharpes) / periods_per_year


def _skewness(values: list[float]) -> float:
    # Population moments (ddof=0), matching the paper's gamma_3/gamma_4.
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return 0.0
    count = len(values)
    return sum(((value - mean) / stdev) ** 3 for value in values) / count


def _kurtosis(values: list[float]) -> float:
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return 3.0
    count = len(values)
    return sum(((value - mean) / stdev) ** 4 for value in values) / count
