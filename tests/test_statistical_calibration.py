"""Check the statistics against what they claim to be, not against themselves.

A judge library is worth nothing if its numbers are subtly wrong, and pinning
an implementation against a transcription of itself catches only typos. These
tests instead assert the defining properties:

- DSR is a probability, so on strategies that are true nulls it must be
  uniformly distributed — about 5% of pure noise should clear 0.95, no more.
- The expected maximum Sharpe of N noise trials is an analytic approximation
  of a quantity we can simulate directly.
- PBO of the best of many independent noise columns must sit near 0.5: the
  in-sample winner has no out-of-sample edge, so it lands in the bottom half
  about half the time.

Every seed is fixed, so these are deterministic despite being simulations.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from trialgate import deflated_sharpe_ratio, probability_of_backtest_overfitting


def test_dsr_is_calibrated_on_true_nulls() -> None:
    """~5% of zero-edge strategies should clear the 0.95 gate. Not 0%, not 30%."""

    rng = random.Random(4242)
    trials = 400
    cleared = 0
    for _ in range(trials):
        returns = [rng.gauss(0.0, 0.01) for _ in range(500)]
        result = deflated_sharpe_ratio(
            returns, trial_sharpes=[1.0], periods_per_year=252, effective_trials=1
        )
        if result.deflated_sharpe_ratio >= 0.95:
            cleared += 1

    # Binomial(400, 0.05) has mean 20, sd 4.4; this window is roughly +/- 3 sd.
    assert 0.02 <= cleared / trials <= 0.09, f"{cleared}/{trials} cleared the DSR gate"


def test_expected_max_sharpe_matches_a_direct_simulation() -> None:
    """The Bailey/Lopez de Prado approximation vs. simulating the maximum."""

    rng = random.Random(97)
    annualized = [0.9, 1.4, 0.2, 1.1, 0.6, 1.8, 0.4, 1.0]
    periods_per_year = 252
    spread = math.sqrt(statistics.pvariance(annualized) / periods_per_year)

    for effective_trials in (10, 100, 1000):
        result = deflated_sharpe_ratio(
            [rng.gauss(0.0005, 0.01) for _ in range(400)],
            trial_sharpes=annualized,
            periods_per_year=periods_per_year,
            effective_trials=effective_trials,
        )
        simulated = statistics.fmean(
            max(rng.gauss(0.0, spread) for _ in range(effective_trials)) for _ in range(2000)
        )

        assert result.expected_max_sharpe == pytest.approx(simulated, rel=0.06), (
            f"N={effective_trials}: approximation {result.expected_max_sharpe:.6f} "
            f"vs simulated {simulated:.6f}"
        )


def test_pbo_of_independent_noise_columns_is_near_one_half() -> None:
    """No column has an edge, so the in-sample winner is a coin flip afterwards."""

    rng = random.Random(31337)
    matrix = [[rng.gauss(0.0, 0.01) for _ in range(24)] for _ in range(480)]

    result = probability_of_backtest_overfitting(matrix, block_count=10)

    assert 0.35 <= result.pbo <= 0.65, f"PBO of pure noise was {result.pbo}"


def test_pbo_is_zero_when_the_edge_is_real_and_persistent() -> None:
    """The mirror image: a genuine, stable edge must never look overfit."""

    rng = random.Random(5)
    matrix = [
        [0.004 + rng.gauss(0.0, 0.002)] + [rng.gauss(0.0, 0.01) for _ in range(9)]
        for _ in range(480)
    ]

    result = probability_of_backtest_overfitting(matrix, block_count=10)

    assert result.pbo == 0.0
