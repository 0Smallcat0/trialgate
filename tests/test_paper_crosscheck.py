"""The statistics against the papers' own numbers, and against edge inputs.

The calibration tests check what the statistics claim to be; these check them
against the published worked example and against inputs where a correct
formula can still be computed wrongly: a constant column, a NaN from a
backtest's first bar, a trial count with no trial spread behind it.

Sources:
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (JPM 2014), the
  numerical example in section 4: N = 100 trials, V[SR_n] = 1/2 annualized,
  annualized SR = 2.5 over T = 1250 daily observations (250 per year),
  skew -3, kurtosis 10 -> SR_0 = 0.1132 per period, DSR = 0.9004.
- Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
  Overfitting" (J. Computational Finance 2017), section 2.
"""

from __future__ import annotations

import math
import random
import statistics
from itertools import combinations

import pytest

from trialgate import (
    ValidationInputError,
    deflated_sharpe_ratio,
    observations_required_for_dsr,
    probability_of_backtest_overfitting,
)

_PAPER_TRIAL_SHARPES = [math.sqrt(0.5), -math.sqrt(0.5)]  # population variance 1/2


def _paper_example_returns() -> list[float]:
    """1250 daily returns with SR = 2.5/sqrt(250), skew ~ -3, kurtosis ~ 10.

    Kurtosis is bounded below by skew**2 + 1, and the example sits exactly on
    that bound, which only a two-point distribution reaches: a fraction p with
    p(1 - p) = 1/13 at the low point. 105 of 1250 is the nearest whole count,
    giving skew -2.9994 and kurtosis 9.9965.
    """

    points = [-1.0] * 105 + [0.0] * 1145
    mean = statistics.fmean(points)
    stdev = statistics.stdev(points)
    per_period_sharpe = 2.5 / math.sqrt(250)
    return [((point - mean) / stdev + per_period_sharpe) * 0.01 for point in points]


def _population_moment(values: list[float], order: int) -> float:
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return sum(((value - mean) / spread) ** order for value in values) / len(values)


def test_expected_max_sharpe_reproduces_the_dsr_paper_example() -> None:
    result = deflated_sharpe_ratio(
        _paper_example_returns(),
        trial_sharpes=_PAPER_TRIAL_SHARPES,
        periods_per_year=250,
        effective_trials=100,
    )

    assert result.expected_max_sharpe == pytest.approx(0.1132, abs=5e-5)


def test_deflated_sharpe_reproduces_the_dsr_paper_example() -> None:
    returns = _paper_example_returns()
    assert _population_moment(returns, 3) == pytest.approx(-3.0, abs=1e-3)
    assert _population_moment(returns, 4) == pytest.approx(10.0, abs=1e-2)

    result = deflated_sharpe_ratio(
        returns,
        trial_sharpes=_PAPER_TRIAL_SHARPES,
        periods_per_year=250,
        effective_trials=100,
    )

    assert result.observed_sharpe == pytest.approx(2.5 / math.sqrt(250))
    assert result.deflated_sharpe_ratio == pytest.approx(0.9004, abs=5e-4)


def _reference_dsr(
    returns: list[float], trial_sharpes: list[float], periods_per_year: int, trials: int
) -> float:
    """DSR paper eqs. for SR_0 and PSR, written out from raw sums."""

    count = len(returns)
    mean = sum(returns) / count
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in returns) / (count - 1))
    sharpe = mean / sample_sd
    skew = _population_moment(returns, 3)
    kurtosis = _population_moment(returns, 4)
    trial_mean = sum(trial_sharpes) / len(trial_sharpes)
    variance = sum((value - trial_mean) ** 2 for value in trial_sharpes) / len(trial_sharpes)
    inverse = statistics.NormalDist().inv_cdf
    gamma = 0.5772156649015329
    benchmark = math.sqrt(variance / periods_per_year) * (
        (1 - gamma) * inverse(1 - 1 / trials) + gamma * inverse(1 - 1 / (trials * math.e))
    )
    statistic = (sharpe - benchmark) * math.sqrt(count - 1)
    statistic /= math.sqrt(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2)
    return statistics.NormalDist().cdf(statistic)


@pytest.mark.parametrize("seed", range(8))
def test_dsr_matches_a_reference_written_from_raw_sums(seed: int) -> None:
    rng = random.Random(seed)
    count = rng.randrange(60, 2000)
    # Student-t-like fat tails: a normal scaled by a random volatility regime.
    returns = [rng.gauss(0.0006, 0.01) * rng.choice((0.5, 1.0, 3.0)) for _ in range(count)]
    trial_sharpes = [rng.gauss(0.6, 0.8) for _ in range(rng.randrange(2, 40))]
    periods_per_year = rng.choice((12, 52, 252, 365))
    trials = rng.randrange(2, 3000)

    result = deflated_sharpe_ratio(
        returns,
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
        effective_trials=trials,
    )

    assert result.deflated_sharpe_ratio == pytest.approx(
        _reference_dsr(returns, trial_sharpes, periods_per_year, trials), abs=1e-12
    )


def test_dsr_never_rises_as_trials_accumulate() -> None:
    rng = random.Random(11)
    returns = [rng.gauss(0.0008, 0.01) for _ in range(800)]
    sharpes = [rng.gauss(0.5, 0.6) for _ in range(20)]
    values = [
        deflated_sharpe_ratio(
            returns, trial_sharpes=sharpes, periods_per_year=252, effective_trials=trials
        ).deflated_sharpe_ratio
        for trials in (2, 5, 10, 50, 100, 1000, 10000)
    ]

    assert values == sorted(values, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_claimed_trials_without_a_trial_spread_are_refused() -> None:
    """One Sharpe has no variance, so SR_0 would be 0: N trials, deflated by nothing."""

    rng = random.Random(2)
    returns = [rng.gauss(0.0008, 0.01) for _ in range(500)]

    with pytest.raises(ValidationInputError, match="at least 2 trials"):
        deflated_sharpe_ratio(
            returns, trial_sharpes=[1.4], periods_per_year=252, effective_trials=450
        )
    with pytest.raises(ValidationInputError, match="at least 2 trials"):
        deflated_sharpe_ratio(returns, trial_sharpes=[], periods_per_year=252, effective_trials=2)


def test_identical_trial_sharpes_are_a_legitimate_zero_spread() -> None:
    # Every trial scored the same, so selecting among them selected nothing.
    rng = random.Random(3)
    result = deflated_sharpe_ratio(
        [rng.gauss(0.0008, 0.01) for _ in range(500)],
        trial_sharpes=[1.1, 1.1, 1.1],
        periods_per_year=252,
        effective_trials=3,
    )

    assert result.expected_max_sharpe == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_dsr_refuses_non_finite_inputs(bad: float) -> None:
    rng = random.Random(4)
    returns = [rng.gauss(0.0008, 0.01) for _ in range(300)]

    with pytest.raises(ValidationInputError, match="finite"):
        deflated_sharpe_ratio(
            [bad, *returns], trial_sharpes=[0.5, 1.0], periods_per_year=252, effective_trials=5
        )
    with pytest.raises(ValidationInputError, match="finite"):
        deflated_sharpe_ratio(
            returns, trial_sharpes=[0.5, bad], periods_per_year=252, effective_trials=5
        )


@pytest.mark.parametrize("target", [0.05, 0.5, 1.0])
def test_data_requirement_refuses_targets_it_cannot_invert(target: float) -> None:
    # Below 0.5 the squared threshold would report the requirement for
    # 1 - target: asking for DSR >= 0.05 used to cost as much data as 0.95.
    rng = random.Random(5)
    with pytest.raises(ValidationInputError, match="target_dsr"):
        observations_required_for_dsr(
            [rng.gauss(0.0008, 0.01) for _ in range(400)],
            trial_sharpes=[1.0, 1.2, 0.8],
            periods_per_year=252,
            effective_trials=12,
            target_dsr=target,
        )


def _logit_pbo(matrix: list[list[float]], block_count: int) -> float:
    """PBO paper section 2, with the logit kept rather than simplified away."""

    strategies = len(matrix[0])
    size = len(matrix) // block_count
    blocks = [matrix[index * size : (index + 1) * size] for index in range(block_count)]

    def sharpes(rows: list[list[float]]) -> list[float]:
        scores = []
        for column in zip(*rows, strict=True):
            stdev = statistics.stdev(column)  # exact: a constant column gives 0.0
            scores.append(statistics.fmean(column) / stdev if stdev > 0.0 else 0.0)
        return scores

    logits = []
    for train in combinations(range(block_count), block_count // 2):
        train_rows = [row for index in train for row in blocks[index]]
        test_rows = [
            row for index in range(block_count) if index not in train for row in blocks[index]
        ]
        in_sample, out_of_sample = sharpes(train_rows), sharpes(test_rows)
        chosen = in_sample.index(max(in_sample))
        omega = sorted(out_of_sample).index(out_of_sample[chosen]) + 1
        omega_bar = omega / (strategies + 1)
        logits.append(math.log(omega_bar / (1 - omega_bar)))
    return sum(1 for value in logits if value <= 0.0) / len(logits)


@pytest.mark.parametrize(("seed", "block_count"), [(1, 4), (2, 6), (3, 8), (4, 10)])
def test_pbo_matches_the_logit_definition(seed: int, block_count: int) -> None:
    rng = random.Random(seed)
    drifts = [rng.gauss(0.0, 0.0006) for _ in range(9)]
    matrix = [[rng.gauss(drift, 0.01) for drift in drifts] for _ in range(block_count * 30)]

    assert probability_of_backtest_overfitting(
        matrix, block_count=block_count
    ).pbo == pytest.approx(_logit_pbo(matrix, block_count))


def test_pbo_is_invariant_to_column_order_and_positive_rescaling() -> None:
    # Sharpe ratios are scale-free and the in-sample winner does not depend on
    # where it sits, so neither change may move PBO.
    rng = random.Random(6)
    drifts = [rng.gauss(0.0, 0.0006) for _ in range(7)]
    matrix = [[rng.gauss(drift, 0.01) for drift in drifts] for _ in range(320)]
    order = [3, 0, 6, 1, 5, 2, 4]
    scales = [0.5, 2.0, 1.0, 7.0, 0.1, 3.0, 1.5]
    shuffled = [[row[column] for column in order] for row in matrix]
    rescaled = [[value * scale for value, scale in zip(row, scales, strict=True)] for row in matrix]

    baseline = probability_of_backtest_overfitting(matrix, block_count=8).pbo
    assert probability_of_backtest_overfitting(shuffled, block_count=8).pbo == baseline
    assert probability_of_backtest_overfitting(rescaled, block_count=8).pbo == baseline


def test_a_constant_return_column_scores_zero_not_a_rounding_residue() -> None:
    """A cash-like trial earning 0.169% every period has no Sharpe ratio.

    Summing squares and subtracting the squared sum left a residue of ~1e-20
    here, and dividing the mean by its square root scored this column at a
    Sharpe of ~2e7: it won every in-sample split and every out-of-sample rank,
    and PBO came out 0.0 for a matrix whose literal definition gives 0.4.
    """

    rng = random.Random(1)
    matrix = [[0.00169] + [rng.gauss(0.0, 0.01) for _ in range(5)] for _ in range(480)]

    pbo = probability_of_backtest_overfitting(matrix, block_count=8).pbo

    assert pbo == pytest.approx(_logit_pbo(matrix, 8))
    assert pbo > 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_pbo_refuses_non_finite_returns(bad: float) -> None:
    """A NaN from a backtest's first bar used to decide argmax and rank silently."""

    rng = random.Random(7)
    matrix = [[rng.gauss(0.0, 0.01) for _ in range(4)] for _ in range(160)]
    matrix[5][1] = bad

    with pytest.raises(ValidationInputError, match="finite"):
        probability_of_backtest_overfitting(matrix, block_count=8)


def test_pbo_refuses_splits_too_short_for_a_sharpe_ratio() -> None:
    # Two blocks of one row each: every Sharpe was scored 0.0, every split
    # ranked the chosen trial top, and PBO came out a passing 0.0.
    with pytest.raises(ValidationInputError, match="too few for a Sharpe"):
        probability_of_backtest_overfitting([[0.01, -0.02], [-0.03, 0.04]], block_count=2)
    result = probability_of_backtest_overfitting(
        [[0.01, -0.02], [-0.03, 0.04], [0.02, 0.01], [0.0, -0.01]], block_count=2
    )
    assert result.combinations_evaluated == 2


def test_a_column_constant_after_its_first_row_scores_zero_not_a_rounding_residue() -> None:
    """Shifting by the first row does not cancel a column that is constant elsewhere.

    A trial that pays an entry cost on its first bar and then earns a flat
    0.169% is constant over every split, but not equal to its first value, so
    the shifted sums still left a rounding residue and the column scored a
    Sharpe in the millions.
    """

    rng = random.Random(3)
    matrix = [[0.00169] + [rng.gauss(0.0, 0.01) for _ in range(5)] for _ in range(480)]
    matrix[0][0] = -0.0021

    pbo = probability_of_backtest_overfitting(matrix, block_count=8).pbo

    assert pbo == pytest.approx(_logit_pbo(matrix, 8))
