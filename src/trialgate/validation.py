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
from collections.abc import Sequence
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
class DataRequirement:
    """The answer to "how much more data", instead of "not enough data".

    ``observations_required`` and ``shortfall`` are None exactly when
    ``reachable`` is False, which is the honest outcome for a candidate whose
    Sharpe already sits below the expected maximum of N noise trials: it does
    not need more data, it needs a different strategy.
    """

    reachable: bool
    observations_present: int
    observations_required: int | None
    shortfall: int | None
    target_dsr: float


@dataclass(frozen=True, slots=True)
class PBOResult:
    """CSCV output: probability of backtest overfitting."""

    pbo: float
    combinations_evaluated: int
    block_count: int
    strategies: int
    observations: int


def sharpe_ratio(returns: list[float], *, periods_per_year: int) -> float:
    """Annualized Sharpe ratio of a periodic return series (risk-free = 0).

    ``periods_per_year`` has no default on purpose: 252 for trading days, 365
    for crypto, 12 for monthly. Guessing it wrong silently rescales every
    Sharpe in the registry by up to 20%.
    """

    if len(returns) < 2:
        return 0.0
    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        return 0.0
    return statistics.fmean(returns) / stdev * math.sqrt(periods_per_year)


def deflated_sharpe_ratio(
    returns: list[float],
    *,
    trial_sharpes: Sequence[float],
    periods_per_year: int,
    effective_trials: int,
) -> DeflatedSharpeResult:
    """Probability that the true Sharpe exceeds the expected max of N noise trials.

    ``returns`` are the candidate strategy's periodic (e.g. daily) returns.
    ``trial_sharpes`` are the ANNUALIZED Sharpe ratios of every registered
    trial — the same numbers the registry stores — de-annualized here rather
    than by the caller, because feeding annualized variance into this formula
    inflates the expected maximum by ``periods_per_year`` and false-rejects
    everything. ``effective_trials`` is the correlation-adjusted trial count N;
    a flattering N voids the gate.
    """

    if effective_trials < 1:
        msg = "effective_trials must be at least 1"
        raise ValidationInputError(msg)
    if periods_per_year < 1:
        msg = "periods_per_year must be at least 1"
        raise ValidationInputError(msg)
    # Var(SR_annual / sqrt(p)) == Var(SR_annual) / p.
    trial_sharpe_variance = (
        statistics.pvariance(trial_sharpes) / periods_per_year if len(trial_sharpes) > 1 else 0.0
    )
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


def observations_required_for_dsr(
    returns: list[float],
    *,
    trial_sharpes: Sequence[float],
    periods_per_year: int,
    effective_trials: int,
    target_dsr: float = 0.95,
) -> DataRequirement:
    """How many observations this candidate would need to clear the DSR gate.

    "Not enough data" is a true statement that ends a conversation. This turns
    it into a quantity someone can go and collect, by inverting the DSR
    statistic for the sample size at which it would reach ``target_dsr``.

    The answer assumes the observed per-period Sharpe, skew and kurtosis hold
    over the additional data. That is a requirement, not a forecast: collecting
    the extra observations can just as easily reveal the Sharpe was noise,
    which is the outcome the gate exists to allow. When the candidate's Sharpe
    is already below the expected maximum of N noise trials, no sample size
    rescues it and ``reachable`` is False -- more data cannot fix having been
    beaten by chance.
    """

    measured = deflated_sharpe_ratio(
        returns,
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
        effective_trials=effective_trials,
    )
    present = len(returns)
    edge = measured.observed_sharpe - measured.expected_max_sharpe
    if edge <= 0.0:
        return DataRequirement(
            reachable=False,
            observations_present=present,
            observations_required=None,
            shortfall=None,
            target_dsr=target_dsr,
        )

    skewness = _skewness(returns)
    kurtosis = _kurtosis(returns)
    observed = measured.observed_sharpe
    mertens_term = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if mertens_term <= 0.0:
        msg = (
            "returns are too extreme or too short to invert the DSR "
            f"(Mertens variance term {mertens_term:.6f} <= 0)"
        )
        raise ValidationInputError(msg)

    threshold = statistics.NormalDist().inv_cdf(target_dsr)
    required = 1 + math.ceil((threshold * math.sqrt(mertens_term) / edge) ** 2)
    return DataRequirement(
        reachable=True,
        observations_present=present,
        observations_required=required,
        shortfall=max(0, required - present),
        target_dsr=target_dsr,
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

    # Every split needs each column's Sharpe over its half of the blocks.
    # Recomputing those from raw rows repeats the same additions across all
    # C(S, S/2) splits. Sum and sum-of-squares per block compose additively,
    # so accumulate them once and assemble each split from S/2 of them.
    block_sums: list[list[float]] = []
    block_squares: list[list[float]] = []
    for block in range(block_count):
        sums = [0.0] * strategies
        squares = [0.0] * strategies
        for row in performance_matrix[block * block_size : (block + 1) * block_size]:
            for column in range(strategies):
                value = row[column]
                sums[column] += value
                squares[column] += value * value
        block_sums.append(sums)
        block_squares.append(squares)

    half = block_count // 2
    split_size = block_size * half
    overfit_count = 0
    combos = list(combinations(range(block_count), half))
    for train_blocks in combos:
        train_set = frozenset(train_blocks)
        test_blocks = tuple(block for block in range(block_count) if block not in train_set)
        train_scores = _split_sharpes(block_sums, block_squares, train_blocks, split_size)
        test_scores = _split_sharpes(block_sums, block_squares, test_blocks, split_size)
        best_in_sample = max(range(strategies), key=lambda column: train_scores[column])
        best_out_of_sample = test_scores[best_in_sample]
        rank = sum(1 for score in test_scores if score <= best_out_of_sample)
        if rank / (strategies + 1) <= 0.5:
            overfit_count += 1

    return PBOResult(
        pbo=overfit_count / len(combos),
        combinations_evaluated=len(combos),
        block_count=block_count,
        strategies=strategies,
        observations=usable_rows,
    )


def _split_sharpes(
    block_sums: list[list[float]],
    block_squares: list[list[float]],
    blocks: tuple[int, ...],
    split_size: int,
) -> list[float]:
    """Per-column Sharpe over the union of ``blocks``, from block statistics."""

    strategies = len(block_sums[0])
    if split_size < 2:
        return [0.0] * strategies
    totals = [0.0] * strategies
    squares = [0.0] * strategies
    for block in blocks:
        block_total = block_sums[block]
        block_square = block_squares[block]
        for column in range(strategies):
            totals[column] += block_total[column]
            squares[column] += block_square[column]
    scores: list[float] = []
    for column in range(strategies):
        mean = totals[column] / split_size
        # Sample variance (ddof=1), matching statistics.stdev.
        variance = (squares[column] - totals[column] * mean) / (split_size - 1)
        scores.append(mean / math.sqrt(variance) if variance > 0.0 else 0.0)
    return scores


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
