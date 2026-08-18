"""Turning "not enough data" into a number someone can go and collect.

The inversion has to agree with the thing it inverts: at exactly the sample
size it reports, the DSR must land on the target. Anything looser would let a
plausible-looking figure drift away from the gate it is supposed to describe.
"""

from __future__ import annotations

import random

import pytest

from trialgate import deflated_sharpe_ratio, observations_required_for_dsr


def _series(*, mean: float, stdev: float, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, stdev) for _ in range(count)]


def test_the_reported_size_is_the_boundary_the_gate_actually_uses() -> None:
    """Straddle the reported n and the verdict must flip across it.

    The series is lengthened by whole repetitions rather than sliced to length.
    A truncated repetition double-counts its head and moves the mean -- an
    earlier version of this test did exactly that and blamed the inversion for
    an 8% Sharpe drop it had caused itself.
    """

    base = _series(mean=0.0008, stdev=0.010, count=400, seed=5)
    trial_sharpes = [1.0, 1.2, 0.8]
    requirement = observations_required_for_dsr(
        base, trial_sharpes=trial_sharpes, periods_per_year=252, effective_trials=12
    )

    assert requirement.reachable
    assert requirement.observations_required is not None
    enough = -(-requirement.observations_required // len(base))  # ceil, in repetitions
    assert enough >= 2, "pick a base short enough that the requirement bites"

    def dsr_at(repetitions: int) -> float:
        return deflated_sharpe_ratio(
            base * repetitions,
            trial_sharpes=trial_sharpes,
            periods_per_year=252,
            effective_trials=12,
        ).deflated_sharpe_ratio

    assert dsr_at(enough) >= 0.95
    assert dsr_at(enough - 1) < 0.95


def test_a_shortfall_is_reported_when_the_sample_is_too_small() -> None:
    returns = _series(mean=0.0008, stdev=0.010, count=400, seed=5)
    requirement = observations_required_for_dsr(
        returns, trial_sharpes=[1.0, 1.2, 0.8], periods_per_year=252, effective_trials=12
    )

    assert requirement.observations_present == 400
    assert requirement.observations_required is not None
    assert requirement.shortfall == max(0, requirement.observations_required - 400)


def test_an_already_qualified_candidate_needs_nothing_more() -> None:
    returns = _series(mean=0.0015, stdev=0.008, count=1500, seed=7)
    requirement = observations_required_for_dsr(
        returns, trial_sharpes=[1.0], periods_per_year=252, effective_trials=1
    )

    assert requirement.reachable
    assert requirement.shortfall == 0


def test_being_beaten_by_chance_is_not_a_sample_size_problem() -> None:
    """A Sharpe below the expected max of N noise trials is unreachable, not thin."""

    returns = _series(mean=0.00002, stdev=0.012, count=600, seed=3)
    requirement = observations_required_for_dsr(
        returns,
        trial_sharpes=[2.5, -2.0, 1.8, -1.5, 2.2],
        periods_per_year=252,
        effective_trials=500,
    )

    assert requirement.reachable is False
    assert requirement.observations_required is None
    assert requirement.shortfall is None


def test_more_assumed_trials_demand_more_data() -> None:
    returns = _series(mean=0.0008, stdev=0.010, count=400, seed=5)

    def required_for(effective_trials: int) -> int:
        requirement = observations_required_for_dsr(
            returns,
            trial_sharpes=[1.0, 1.2, 0.8],
            periods_per_year=252,
            effective_trials=effective_trials,
        )
        assert requirement.observations_required is not None
        return requirement.observations_required

    assert required_for(50) > required_for(2)


def test_a_lower_target_is_cheaper_to_reach() -> None:
    returns = _series(mean=0.0008, stdev=0.010, count=400, seed=5)

    strict = observations_required_for_dsr(
        returns,
        trial_sharpes=[1.0, 1.2, 0.8],
        periods_per_year=252,
        effective_trials=12,
        target_dsr=0.99,
    )
    lenient = observations_required_for_dsr(
        returns,
        trial_sharpes=[1.0, 1.2, 0.8],
        periods_per_year=252,
        effective_trials=12,
        target_dsr=0.90,
    )

    assert strict.observations_required is not None
    assert lenient.observations_required is not None
    assert strict.observations_required > lenient.observations_required


def test_a_constant_series_is_rejected_rather_than_scored() -> None:
    with pytest.raises(ValueError, match="constant"):
        observations_required_for_dsr(
            [0.001] * 200, trial_sharpes=[1.0], periods_per_year=252, effective_trials=5
        )
