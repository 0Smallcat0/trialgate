"""Check the stop-line gate against what it claims to be.

The closed form is an exact identity, so the useful test is not a transcription
of the formula back into the assertion -- that catches typos and nothing else.
These tests pin it against three independent things:

- a direct simulation of the process it claims to describe,
- the two limits where the answer is known analytically (zero log drift
  collapses to the reflection principle; a long horizon with positive drift
  collapses to the classic ruin probability),
- the monotonicity that makes the volatility solver's bisection valid.

Simulation undershoots the closed form on purpose: a discrete walk can step
over the barrier and back between observations, so it misses crossings the
continuous process makes. The tests assert that direction explicitly, because
a formula error would break the ordering, not just the tolerance.

Every seed is fixed, so these are deterministic despite being simulations.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from trialgate import (
    RiskInputError,
    assess_stop_line,
    max_volatility_for_stop,
    stop_breach_probability,
)


def _simulate_breach(
    *,
    annual_return: float,
    annual_volatility: float,
    stop_loss_fraction: float,
    horizon_years: float = 1.0,
    paths: int = 2500,
    steps: int = 504,
    seed: int = 11,
) -> float:
    """Fraction of simulated log-equity paths that ever touch the stop level."""

    rng = random.Random(seed)
    step_years = horizon_years / steps
    drift = (annual_return - annual_volatility**2 / 2.0) * step_years
    shock = annual_volatility * math.sqrt(step_years)
    barrier = math.log(1.0 - stop_loss_fraction)

    breached = 0
    for _ in range(paths):
        equity = 0.0
        for _ in range(steps):
            equity += drift + shock * rng.gauss(0.0, 1.0)
            if equity <= barrier:
                breached += 1
                break
    return breached / paths


@pytest.mark.parametrize(
    ("sharpe", "volatility"),
    [(1.0, 0.15), (0.5, 0.15), (-0.3, 0.20)],
)
def test_the_closed_form_tracks_a_direct_simulation(sharpe: float, volatility: float) -> None:
    closed_form = stop_breach_probability(
        annual_return=sharpe * volatility,
        annual_volatility=volatility,
        stop_loss_fraction=0.15,
        horizon_years=1.0,
    )
    simulated = _simulate_breach(
        annual_return=sharpe * volatility,
        annual_volatility=volatility,
        stop_loss_fraction=0.15,
    )

    assert simulated <= closed_form + 0.01, "discrete paths cannot beat the continuous barrier"
    assert closed_form - simulated <= 0.035, f"closed form {closed_form} vs simulated {simulated}"


def test_zero_log_drift_collapses_to_the_reflection_principle() -> None:
    """With mu - sigma^2/2 == 0 the answer is exactly 2 * Phi(-L / (sigma sqrt T))."""

    volatility = 0.20
    level = -math.log(1.0 - 0.15)
    expected = 2.0 * statistics.NormalDist().cdf(-level / volatility)

    observed = stop_breach_probability(
        annual_return=volatility**2 / 2.0,
        annual_volatility=volatility,
        stop_loss_fraction=0.15,
        horizon_years=1.0,
    )

    assert observed == pytest.approx(expected, abs=1e-12)


def test_a_long_horizon_collapses_to_the_classic_ruin_probability() -> None:
    """As T grows, a profitable strategy's breach probability -> exp(-2 mu L / sigma^2)."""

    volatility = 0.15
    annual_return = 0.20
    level = -math.log(1.0 - 0.15)
    drift = annual_return - volatility**2 / 2.0
    expected = math.exp(-2.0 * drift * level / volatility**2)

    observed = stop_breach_probability(
        annual_return=annual_return,
        annual_volatility=volatility,
        stop_loss_fraction=0.15,
        horizon_years=400.0,
    )

    assert observed == pytest.approx(expected, rel=1e-6)


def test_breach_probability_increases_with_volatility() -> None:
    """The property the volatility solver's bisection depends on."""

    probabilities = [
        stop_breach_probability(
            annual_return=1.0 * volatility,
            annual_volatility=volatility,
            stop_loss_fraction=0.15,
            horizon_years=1.0,
        )
        for volatility in (0.05, 0.10, 0.15, 0.25, 0.40, 0.80, 2.00)
    ]

    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.01
    assert probabilities[-1] > 0.90


def test_a_losing_strategy_does_not_overflow_the_reflected_term() -> None:
    """Negative drift makes exp(-2 mu L / sigma^2) enormous; the log-space path must hold."""

    observed = stop_breach_probability(
        annual_return=-2.0,
        annual_volatility=0.05,
        stop_loss_fraction=0.15,
        horizon_years=1.0,
    )

    assert observed == pytest.approx(1.0, abs=1e-9)


def test_the_volatility_solver_round_trips_to_the_target() -> None:
    for sharpe in (0.5, 1.0, 2.0):
        volatility = max_volatility_for_stop(
            sharpe_ratio=sharpe,
            stop_loss_fraction=0.15,
            horizon_years=1.0,
            max_breach_probability=0.10,
        )
        breach = stop_breach_probability(
            annual_return=sharpe * volatility,
            annual_volatility=volatility,
            stop_loss_fraction=0.15,
            horizon_years=1.0,
        )

        assert breach == pytest.approx(0.10, abs=1e-6)


def test_an_unconstrained_stop_line_reports_the_cap_not_an_invented_limit() -> None:
    """A deep stop over a few days binds nothing; the solver must not fabricate a cap."""

    volatility = max_volatility_for_stop(
        sharpe_ratio=3.0,
        stop_loss_fraction=0.90,
        horizon_years=0.01,
        max_breach_probability=0.50,
    )

    assert volatility == 10.0


def test_assess_stop_line_reads_both_moments_off_the_series() -> None:
    rng = random.Random(2026)
    daily_volatility = 0.15 / math.sqrt(252)
    returns = [rng.gauss(0.10 / 252, daily_volatility) for _ in range(1500)]

    result = assess_stop_line(returns, periods_per_year=252, stop_loss_fraction=0.15)

    assert result.annual_volatility == pytest.approx(0.15, rel=0.10)
    assert result.annual_return == pytest.approx(0.10, abs=0.06)
    assert result.max_breach_probability == 0.10
    assert result.passed is (result.breach_probability <= 0.10)


def test_volatility_scale_is_the_position_size_instruction() -> None:
    rng = random.Random(99)
    returns = [rng.gauss(0.0002, 0.02) for _ in range(1200)]

    result = assess_stop_line(returns, periods_per_year=252, stop_loss_fraction=0.15)
    resized = [value * result.volatility_scale for value in returns]
    after = assess_stop_line(resized, periods_per_year=252, stop_loss_fraction=0.15)

    assert after.breach_probability <= 0.10 + 1e-6
    assert after.breach_probability <= result.breach_probability


@pytest.mark.parametrize(
    "kwargs",
    [
        {"annual_volatility": 0.0},
        {"horizon_years": 0.0},
        {"stop_loss_fraction": 0.0},
        {"stop_loss_fraction": 1.0},
        {"stop_loss_fraction": -0.1},
    ],
)
def test_unusable_inputs_raise_instead_of_returning_a_number(kwargs: dict[str, float]) -> None:
    arguments: dict[str, float] = {
        "annual_return": 0.10,
        "annual_volatility": 0.15,
        "stop_loss_fraction": 0.15,
        "horizon_years": 1.0,
    }
    arguments.update(kwargs)

    with pytest.raises(RiskInputError):
        stop_breach_probability(**arguments)


def test_a_constant_series_is_rejected_rather_than_scored() -> None:
    with pytest.raises(RiskInputError, match="constant"):
        assess_stop_line([0.001] * 100, periods_per_year=252, stop_loss_fraction=0.15)
