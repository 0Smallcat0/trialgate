"""Qualification-gate statistics: Deflated Sharpe Ratio and CSCV/PBO.

Sources:
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (JPM 2014; preprint
  of July 31, 2014): the DSR, Eq. (2) p. 8, and the expected maximum Sharpe
  ratio SR_0, Eq. (1) p. 7 / Eq. (6) p. 12.
- Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
  Overfitting" (Journal of Computational Finance 2015).
- Lo, "The Statistics of Sharpe Ratios" (Financial Analysts Journal 58(4),
  2002), for serially correlated returns: eta(q), Eq. (20) p. 40, and the
  GMM variance, Eq. (14) p. 39; see trialgate.autocorrelation.

These statistics operate on measurement floats, not account money, so
float arithmetic is intentional here (unlike a Decimal trading path).
"""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from trialgate.autocorrelation import (
    AutocorrelationDiagnostics,
    andrews_bandwidth,
    diagnose,
    eta,
    sample_autocorrelations,
)

AutocorrelationMode = Literal["report", "adjust", "ignore"]
_AUTOCORRELATION_MODES = ("report", "adjust", "ignore")

_EULER_GAMMA = 0.5772156649015329


class ValidationInputError(ValueError):
    """Raised when gate statistics receive unusable inputs."""


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """DSR output with the inputs that produced it.

    ``deflated_sharpe_ratio`` is the value to judge: the IID statistic unless
    the call asked for ``autocorrelation="adjust"``. The other two DSR fields
    carry both readings whenever autocorrelation was examined (every mode but
    ``"ignore"``), so a report can show what the adjustment would change.
    """

    deflated_sharpe_ratio: float
    observed_sharpe: float
    expected_max_sharpe: float
    effective_trials: int
    observations: int
    iid_deflated_sharpe_ratio: float | None = None
    adjusted_deflated_sharpe_ratio: float | None = None
    autocorrelation: AutocorrelationDiagnostics | None = None
    autocorrelation_adjusted: bool = False


@dataclass(frozen=True, slots=True)
class AnnualizedSharpe:
    """An annualized Sharpe ratio both ways: sqrt(q) and Lo's eta(q).

    ``annualized_sharpe`` = eta(q) * SR uses the sample autocorrelations up to
    ``max_lag`` (zero beyond it); ``iid_annualized_sharpe`` is sharpe_ratio(),
    the sqrt(q) figure, which is right only when returns are uncorrelated.
    """

    annualized_sharpe: float
    iid_annualized_sharpe: float
    eta: float
    periods_per_year: int
    max_lag: int
    autocorrelations: tuple[float, ...]


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


def lo_annualized_sharpe(
    returns: list[float],
    *,
    periods_per_year: int,
    max_lag: int | None = None,
) -> AnnualizedSharpe:
    """Annualize with Lo's (2002) eta(q) instead of sqrt(q).

    SR(q) = eta(q) * SR, eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k),
    q = ``periods_per_year`` (Lo, Eq. (20) p. 40). Positive autocorrelation
    makes eta(q) < sqrt(q): the sqrt(q) figure overstates the annual Sharpe of
    smoothed returns.

    SR here divides by the T - 1 standard deviation, like ``sharpe_ratio``, so
    that ``annualized_sharpe`` equals ``iid_annualized_sharpe`` when every
    rho_k is zero. Lo's Eqs. (3b)-(4) p. 37 divide by the 1/T one, which is
    larger by sqrt(T / (T - 1)), a factor 1 + 1/(2T) to first order.

    ``max_lag`` is how many sample autocorrelations enter the sum. Lo's
    estimator uses all q - 1 ("The General Case", p. 42); pass
    ``max_lag=periods_per_year - 1`` for that, which is the right call for
    monthly data (11 lags). For daily data, 251
    sample autocorrelations are mostly noise and that noise is weighted by up
    to q - 1, so the default truncates at the Andrews (1991) bandwidth of the
    series -- one or two lags on uncorrelated returns, dozens on a strongly
    persistent one -- and treats the rest as zero.
    """

    if periods_per_year < 1:
        msg = "periods_per_year must be at least 1"
        raise ValidationInputError(msg)
    if len(returns) < 3:
        msg = "returns must contain at least 3 observations"
        raise ValidationInputError(msg)
    _require_finite("returns", returns)
    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        msg = "returns must not be constant"
        raise ValidationInputError(msg)
    ceiling = min(periods_per_year - 1, len(returns) - 1)
    if max_lag is None:
        lags = min(ceiling, max(1, math.floor(andrews_bandwidth(returns))))
    elif 0 <= max_lag <= ceiling:
        lags = max_lag
    else:
        msg = f"max_lag must be between 0 and {ceiling}, got {max_lag}"
        raise ValidationInputError(msg)
    rhos = sample_autocorrelations(returns, lags) if lags else []
    try:
        factor = eta(periods_per_year, rhos)
    except ValueError as error:
        raise ValidationInputError(str(error)) from error
    per_period = statistics.fmean(returns) / stdev
    return AnnualizedSharpe(
        annualized_sharpe=per_period * factor,
        iid_annualized_sharpe=sharpe_ratio(returns, periods_per_year=periods_per_year),
        eta=factor,
        periods_per_year=periods_per_year,
        max_lag=lags,
        autocorrelations=tuple(rhos),
    )


def deflated_sharpe_ratio(
    returns: list[float],
    *,
    trial_sharpes: Sequence[float],
    periods_per_year: int,
    effective_trials: int,
    autocorrelation: AutocorrelationMode = "report",
) -> DeflatedSharpeResult:
    """Probability that the true Sharpe exceeds the expected max of N noise trials.

    ``returns`` are the candidate strategy's periodic (e.g. daily) returns.
    ``trial_sharpes`` are the ANNUALIZED Sharpe ratios of every registered
    trial — the same numbers the registry stores — de-annualized here rather
    than by the caller, because feeding annualized variance into this formula
    inflates the expected maximum by ``periods_per_year`` and false-rejects
    everything. ``effective_trials`` is the correlation-adjusted trial count N;
    a flattering N voids the gate.

    The IID value is the DSR paper's Eq. (2) p. 8.

    ``autocorrelation`` decides what serial correlation in ``returns`` does.
    The PSR/DSR variance term assumes independent periods; positively
    autocorrelated returns make it too small and the DSR too high. In every
    mode but ``"ignore"`` the long-run variance of the Sharpe estimator is
    estimated (Lo 2002, GMM case, Eq. (14) p. 39; trialgate.autocorrelation)
    and the statistic recomputed on the equivalent number of independent
    observations:

        z_adj = (SR - SR_0) * sqrt((T - 1) / ratio) / sqrt(Mertens term)

    This adjustment is trialgate's extension of the DSR, not part of Bailey &
    Lopez de Prado's method; see trialgate.autocorrelation.sharpe_variance_ratio.

    ``"report"`` (default) judges the IID value -- bit-for-bit what earlier
    releases returned -- and carries the adjusted one alongside for a gate to
    warn with. ``"adjust"`` judges the adjusted value. ``"ignore"`` skips the
    estimate.
    """

    if effective_trials < 1:
        msg = "effective_trials must be at least 1"
        raise ValidationInputError(msg)
    _require_mode(autocorrelation)
    if periods_per_year < 1:
        msg = "periods_per_year must be at least 1"
        raise ValidationInputError(msg)
    observations = len(returns)
    if observations < 3:
        msg = "returns must contain at least 3 observations"
        raise ValidationInputError(msg)
    _require_finite("returns", returns)
    _require_finite("trial_sharpes", trial_sharpes)

    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        msg = "returns must not be constant"
        raise ValidationInputError(msg)
    if effective_trials > 1 and len(trial_sharpes) < 2:
        # One Sharpe has no spread, and a spread of zero would deflate by
        # nothing: N trials claimed, none charged for.
        msg = (
            f"effective_trials={effective_trials} needs the Sharpe ratios of at least "
            f"2 trials to estimate their spread; got {len(trial_sharpes)}"
        )
        raise ValidationInputError(msg)
    # Var(SR_annual / sqrt(p)) == Var(SR_annual) / p.
    trial_sharpe_variance = (
        statistics.pvariance(trial_sharpes) / periods_per_year if len(trial_sharpes) > 1 else 0.0
    )
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
    iid_dsr = normal.cdf(statistic)
    if autocorrelation == "ignore":
        return DeflatedSharpeResult(
            deflated_sharpe_ratio=iid_dsr,
            observed_sharpe=observed_sharpe,
            expected_max_sharpe=expected_max_sharpe,
            effective_trials=effective_trials,
            observations=observations,
        )

    diagnostics = diagnose(returns)
    adjusted_dsr = normal.cdf(
        (observed_sharpe - expected_max_sharpe)
        * math.sqrt((observations - 1.0) / diagnostics.variance_ratio)
        / denominator
    )
    adjusting = autocorrelation == "adjust"
    return DeflatedSharpeResult(
        deflated_sharpe_ratio=adjusted_dsr if adjusting else iid_dsr,
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max_sharpe,
        effective_trials=effective_trials,
        observations=observations,
        iid_deflated_sharpe_ratio=iid_dsr,
        adjusted_deflated_sharpe_ratio=adjusted_dsr,
        autocorrelation=diagnostics,
        autocorrelation_adjusted=adjusting,
    )


def observations_required_for_dsr(
    returns: list[float],
    *,
    trial_sharpes: Sequence[float],
    periods_per_year: int,
    effective_trials: int,
    target_dsr: float = 0.95,
    autocorrelation: AutocorrelationMode = "report",
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

    With ``autocorrelation="adjust"`` the requirement is in observations of
    the series as it is, autocorrelation included: it is the IID requirement
    scaled by the Sharpe estimator's variance ratio, since each observation
    carries 1 / ratio of an independent one. ``"report"`` and ``"ignore"``
    give the IID requirement.
    """

    if not 0.5 < target_dsr < 1.0:
        # At or below 0.5 any positive edge already clears the target at every
        # sample size, and squaring a negative threshold would report the
        # sample size of the mirror-image target instead.
        msg = f"target_dsr must be in (0.5, 1), got {target_dsr}"
        raise ValidationInputError(msg)
    _require_mode(autocorrelation)
    measured = deflated_sharpe_ratio(
        returns,
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
        effective_trials=effective_trials,
        autocorrelation="adjust" if autocorrelation == "adjust" else "ignore",
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
    squared = (threshold * math.sqrt(mertens_term) / edge) ** 2
    if measured.autocorrelation is not None:
        squared *= measured.autocorrelation.variance_ratio
    required = 1 + math.ceil(squared)
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
    for row in performance_matrix:
        _require_finite("performance_matrix", row)

    usable_rows = observations - (observations % block_count)
    block_size = usable_rows // block_count
    half = block_count // 2
    split_size = block_size * half
    if split_size < 2:
        msg = (
            f"each half of the split would hold {split_size} row, too few for a Sharpe "
            "ratio; supply at least 4 rows for block_count=2"
        )
        raise ValidationInputError(msg)

    # Every split needs each column's Sharpe over its half of the blocks.
    # Recomputing those from raw rows repeats the same additions across all
    # C(S, S/2) splits. Sum and sum-of-squares per block compose additively,
    # so accumulate them once and assemble each split from S/2 of them.
    # Each column is shifted by its first value first: the variance is
    # unchanged, but sum-of-squares minus square-of-sum no longer cancels
    # catastrophically, so a constant column scores exactly zero instead of a
    # rounding residue divided by a near-zero spread.
    shifts = list(performance_matrix[0])
    block_sums: list[list[float]] = []
    block_squares: list[list[float]] = []
    for block in range(block_count):
        sums = [0.0] * strategies
        squares = [0.0] * strategies
        for row in performance_matrix[block * block_size : (block + 1) * block_size]:
            for column in range(strategies):
                value = row[column] - shifts[column]
                sums[column] += value
                squares[column] += value * value
        block_sums.append(sums)
        block_squares.append(squares)

    overfit_count = 0
    combos = list(combinations(range(block_count), half))
    for train_blocks in combos:
        train_set = frozenset(train_blocks)
        test_blocks = tuple(block for block in range(block_count) if block not in train_set)
        train_scores = _split_sharpes(block_sums, block_squares, shifts, train_blocks, split_size)
        test_scores = _split_sharpes(block_sums, block_squares, shifts, test_blocks, split_size)
        best_in_sample = max(range(strategies), key=lambda column: train_scores[column])
        best_out_of_sample = test_scores[best_in_sample]
        # Ties rank against the chosen trial: duplicated columns (a parameter
        # that changes nothing) would otherwise lift it past the median and
        # lower PBO; this way PBO equals that of the distinct trials.
        rank = 1 + sum(1 for score in test_scores if score < best_out_of_sample)
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
    shifts: list[float],
    blocks: tuple[int, ...],
    split_size: int,
) -> list[float]:
    """Per-column Sharpe over the union of ``blocks``, from shifted block statistics."""

    strategies = len(block_sums[0])
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
        shifted_mean = totals[column] / split_size
        spread = squares[column] - totals[column] * shifted_mean
        # A column constant over this split but not equal to its shift (one
        # odd first bar, then flat) still cancels to a rounding residue rather
        # than zero. Summing n terms errs by at most about n * epsilon of the
        # sum of squares, so a spread inside that bound is no spread at all.
        if spread <= 4.0 * split_size * sys.float_info.epsilon * squares[column]:
            scores.append(0.0)
            continue
        # Sample variance (ddof=1), matching statistics.stdev.
        variance = spread / (split_size - 1)
        mean = shifts[column] + shifted_mean
        scores.append(mean / math.sqrt(variance))
    return scores


def _require_mode(mode: str) -> None:
    if mode not in _AUTOCORRELATION_MODES:
        msg = f"autocorrelation must be one of {_AUTOCORRELATION_MODES}, got {mode!r}"
        raise ValidationInputError(msg)


def _require_finite(name: str, values: Sequence[float]) -> None:
    # NaN compares false against every threshold, so it would not fail loudly
    # anywhere downstream: it would quietly decide argmax and rank instead.
    if not all(math.isfinite(value) for value in values):
        msg = f"{name} must contain only finite numbers (found NaN or infinity)"
        raise ValidationInputError(msg)


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
