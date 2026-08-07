from __future__ import annotations

import math
import statistics

import pytest

from trialgate import (
    ValidationInputError,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    validation,
)


def _returns_with_drift() -> list[float]:
    return [0.01 if index % 2 == 0 else -0.004 for index in range(120)]


def test_sharpe_ratio_of_constant_returns_is_zero() -> None:
    assert sharpe_ratio([0.01] * 50, periods_per_year=252) == 0.0


def test_deflated_sharpe_shrinks_as_trials_accumulate() -> None:
    returns = _returns_with_drift()

    single_trial = deflated_sharpe_ratio(
        returns, trial_sharpes=[1.2], periods_per_year=252, effective_trials=1
    )
    many_trials = deflated_sharpe_ratio(
        returns,
        trial_sharpes=[0.2 * index for index in range(30)],
        periods_per_year=252,
        effective_trials=200,
    )

    assert single_trial.expected_max_sharpe == 0.0
    assert many_trials.expected_max_sharpe > 0.0
    assert many_trials.deflated_sharpe_ratio < single_trial.deflated_sharpe_ratio


def test_annualized_trial_sharpes_are_deannualized_internally() -> None:
    # Handing the formula annualized variance inflates the expected maximum by
    # periods_per_year and false-rejects everything; the conversion is the
    # library's job, not the caller's.
    sharpes = [0.4, 1.1, 1.9, 0.7]
    result = deflated_sharpe_ratio(
        _returns_with_drift(), trial_sharpes=sharpes, periods_per_year=252, effective_trials=4
    )

    spread = math.sqrt(statistics.pvariance(sharpes) / 252)
    normal = statistics.NormalDist()
    assert result.expected_max_sharpe == pytest.approx(
        spread
        * (
            (1.0 - 0.5772156649015329) * normal.inv_cdf(1.0 - 1.0 / 4)
            + 0.5772156649015329 * normal.inv_cdf(1.0 - 1.0 / (4 * math.e))
        )
    )


def test_deflated_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationInputError):
        deflated_sharpe_ratio(
            [0.01, 0.02], trial_sharpes=[1.0], periods_per_year=252, effective_trials=1
        )
    with pytest.raises(ValidationInputError):
        deflated_sharpe_ratio(
            _returns_with_drift(), trial_sharpes=[1.0], periods_per_year=0, effective_trials=1
        )
    with pytest.raises(ValidationInputError):
        deflated_sharpe_ratio(
            _returns_with_drift(), trial_sharpes=[1.0], periods_per_year=252, effective_trials=0
        )


def test_dsr_raises_instead_of_fabricating_certainty_on_extreme_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With population moments the Mertens term is bounded below by
    # (1 - skew*SR/2)^2 >= 0, so a breakdown needs a broken estimator; the
    # guard is defense-in-depth and must raise rather than clamp silently.
    monkeypatch.setattr(validation, "_kurtosis", lambda _values: 0.0)
    with pytest.raises(ValidationInputError, match="Mertens"):
        deflated_sharpe_ratio(
            [0.02, 0.019, 0.021, 0.02, 0.018, 0.022] * 10,
            trial_sharpes=[1.0],
            periods_per_year=252,
            effective_trials=1,
        )


def test_pbo_is_zero_when_one_strategy_dominates_everywhere() -> None:
    rows = []
    for index in range(80):
        wiggle = 0.001 if index % 2 == 0 else -0.001
        rows.append([0.02 + wiggle, wiggle, wiggle - 0.0005])
    result = probability_of_backtest_overfitting(rows, block_count=8)

    assert result.pbo == 0.0
    assert result.combinations_evaluated == math.comb(8, 4)


def test_pbo_detects_regime_flipping_strategies() -> None:
    # Two strategies that alternate dominance between the halves of the sample:
    # whatever wins in-sample tends to lose out-of-sample.
    rows = []
    for index in range(80):
        if index < 40:
            rows.append([0.01, -0.01])
        else:
            rows.append([-0.01, 0.01])
    result = probability_of_backtest_overfitting(rows, block_count=8)

    assert result.pbo >= 0.4


def test_pbo_with_sixteen_blocks_evaluates_all_symmetric_splits() -> None:
    rows = [[0.01 * ((index + column) % 3 - 1) for column in range(2)] for index in range(16)]
    result = probability_of_backtest_overfitting(rows, block_count=16)

    assert result.combinations_evaluated == math.comb(16, 8)


def test_pbo_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationInputError):
        probability_of_backtest_overfitting([[0.01, 0.02]] * 4, block_count=7)
    with pytest.raises(ValidationInputError):
        probability_of_backtest_overfitting([[0.01]] * 32, block_count=8)
    with pytest.raises(ValidationInputError):
        probability_of_backtest_overfitting([[0.01, 0.02]] * 4, block_count=8)


def test_a_single_trial_has_no_expected_maximum_to_beat() -> None:
    result = deflated_sharpe_ratio(
        _returns_with_drift(), trial_sharpes=[1.0], periods_per_year=365, effective_trials=1
    )

    assert result.expected_max_sharpe == 0.0
