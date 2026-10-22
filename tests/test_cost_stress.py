"""Gate 9: an edge that only exists at the fee it was tested at is not an edge."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trialgate import (
    GateCheck,
    RegistryTamperError,
    TrialRegistryError,
    append_trial,
    evaluate_registry,
    load_period_costs,
)

_NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
_START = datetime(2019, 1, 1, tzinfo=UTC)
_END = datetime(2025, 6, 30, tzinfo=UTC)
_PERIODS = 1000


def _register(
    path: Path,
    returns: list[float],
    costs: list[float] | None,
    *,
    label: str = "candidate",
) -> int:
    return append_trial(
        path,
        recorded_at=_NOW,
        code_version="test",
        strategy_id="drift",
        parameters={"label": label},
        universe=("SYNTH",),
        data_start=_START,
        data_end=_END,
        cost_assumptions={"round_trip_bps": "10"},
        metrics={},
        operator_note=label,
        period_returns=returns,
        period_costs=costs,
    ).trial_id


def _sweep(path: Path, *, daily_cost: float, costs_for_candidate: bool = True) -> int:
    """One configuration with a real edge, net of ``daily_cost``, and four without.

    The candidate's gross mean is 0.2% a day on 1% volatility, about a 3.2
    annual Sharpe before costs, so gate 4 passes it at N = 5 for every cost
    used below and only the extra charge decides gate 9.
    """

    rng = random.Random(7)
    gross = [rng.gauss(0.002, 0.01) for _ in range(_PERIODS)]
    costs = [daily_cost] * _PERIODS
    candidate = _register(
        path,
        [value - cost for value, cost in zip(gross, costs, strict=True)],
        costs if costs_for_candidate else None,
    )
    for other in range(4):
        noise = [rng.gauss(0.0, 0.01) for _ in range(_PERIODS)]
        _register(path, noise, [0.0001] * _PERIODS, label=f"noise {other}")
    return candidate


def _gate9(path: Path, candidate: int, *, cost_multiplier: float | None = 2.0) -> GateCheck | None:
    report = evaluate_registry(
        path,
        candidate_trial_id=candidate,
        periods_per_year=252,
        effective_trials=5,
        block_count=8,
        min_observations=100,
        cost_multiplier=cost_multiplier,
    )
    return next((c for c in report.checks if c.gate == "cost stress (gate 9)"), None)


# --- the registry stores them ------------------------------------------------


def test_costs_are_stored_beside_the_registry_and_read_back(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    with_costs = _register(registry, [0.01, -0.02, 0.005], [0.001, 0.0, 0.0005])
    without = _register(registry, [0.01, -0.02, 0.005], None, label="no costs")

    assert load_period_costs(registry, with_costs) == [0.001, 0.0, 0.0005]
    assert load_period_costs(registry, without) is None
    row = json.loads(registry.read_text(encoding="utf-8").splitlines()[0])
    assert "costs_sha256" in row
    with pytest.raises(TrialRegistryError, match="not registered"):
        load_period_costs(registry, 99)


def test_an_edited_cost_series_is_tampering(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    trial = _register(registry, [0.01, -0.02], [0.004, 0.004])
    costs_file = tmp_path / "registry.jsonl.costs" / f"{trial}.json"
    costs_file.write_text("[0.0, 0.0]", encoding="utf-8")

    with pytest.raises(RegistryTamperError, match="edited since it was registered"):
        load_period_costs(registry, trial)
    costs_file.unlink()
    with pytest.raises(TrialRegistryError, match="is missing"):
        load_period_costs(registry, trial)


@pytest.mark.parametrize(
    ("returns", "costs", "message"),
    [
        (None, [0.001, 0.001], "line up with period_returns"),
        ([0.01, 0.02, 0.03], [0.001, 0.001], "got 2 costs for 3 returns"),
        ([0.01, 0.02], [0.001, -0.001], "not negative"),
        ([0.01, 0.02], [0.001, math.nan], "finite"),
    ],
)
def test_costs_that_cannot_be_what_was_paid_are_refused(
    tmp_path: Path, returns: list[float] | None, costs: list[float], message: str
) -> None:
    registry = tmp_path / "registry.jsonl"
    with pytest.raises(TrialRegistryError, match=message):
        append_trial(
            registry,
            recorded_at=_NOW,
            code_version="test",
            strategy_id="drift",
            parameters={},
            universe=("SYNTH",),
            data_start=_START,
            data_end=_END,
            cost_assumptions={},
            metrics={},
            operator_note="",
            period_returns=returns,
            period_costs=costs,
        )
    assert not registry.exists()


# --- the gate ------------------------------------------------------------------


def test_an_edge_much_larger_than_its_costs_survives_paying_them_twice(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0001)

    check = _gate9(registry, candidate)

    assert check is not None
    assert check.passed
    assert check.threshold == 0.95
    assert "at 2x the costs" in check.detail


def test_an_edge_the_size_of_its_costs_does_not(tmp_path: Path) -> None:
    """Net of costs it passes gate 4; at twice the fee most of the edge is gone."""

    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.001)
    report = evaluate_registry(
        registry,
        candidate_trial_id=candidate,
        periods_per_year=252,
        effective_trials=5,
        block_count=8,
        min_observations=100,
    )
    dsr = next(c for c in report.checks if c.gate == "DSR (gate 4)")
    stress = next(c for c in report.checks if c.gate == "cost stress (gate 9)")

    assert dsr.passed
    assert not stress.passed
    assert stress.observed < dsr.observed
    assert not report.passed


def test_the_breakeven_multiple_is_where_the_mean_return_reaches_zero(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0005)
    returns = json.loads(
        (tmp_path / "registry.jsonl.returns" / f"{candidate}.json").read_text(encoding="utf-8")
    )
    breakeven = 1.0 + sum(returns) / (0.0005 * _PERIODS)

    check = _gate9(registry, candidate)

    assert check is not None
    assert f"reaches zero at {breakeven:.2f}x those costs" in check.detail
    stressed = [value - (breakeven - 1.0) * 0.0005 for value in returns]
    assert sum(stressed) == pytest.approx(0.0, abs=1e-12)


def test_a_losing_trial_is_already_past_breakeven(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _register(registry, [-0.01, 0.005] * 200, [0.001] * 400)
    _register(registry, [0.0, 0.001] * 200, [0.0] * 400, label="other")

    check = _gate9(registry, candidate)

    assert check is not None
    assert not check.passed
    assert "already at or below zero at the costs charged" in check.detail


def test_a_trial_without_costs_fails_and_says_how_to_supply_them(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0001, costs_for_candidate=False)

    check = _gate9(registry, candidate)

    assert check is not None
    assert not check.passed
    assert math.isnan(check.observed)
    assert "registered without period_costs" in check.detail
    assert "cost_multiplier=None" in check.detail


def test_a_trial_charged_nothing_has_no_cost_to_stress(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _register(registry, [0.002, -0.001] * 200, [0.0] * 400)
    _register(registry, [0.0, 0.001] * 200, [0.0] * 400, label="other")

    check = _gate9(registry, candidate)

    assert check is not None
    assert not check.passed
    assert "charged nothing in any period" in check.detail


def test_the_gate_is_skipped_only_when_asked(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0001, costs_for_candidate=False)

    assert _gate9(registry, candidate, cost_multiplier=None) is None


def test_a_higher_multiplier_can_only_make_the_verdict_harsher(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0004)

    observed = []
    for multiplier in (1.5, 2.0, 3.0, 5.0):
        check = _gate9(registry, candidate, cost_multiplier=multiplier)
        assert check is not None
        assert f"at {multiplier:g}x the costs" in check.detail
        observed.append(check.observed)

    assert observed == sorted(observed, reverse=True)


@pytest.mark.parametrize("multiplier", [1.0, 0.5, math.nan, math.inf])
def test_a_multiplier_that_charges_no_extra_cost_is_refused(
    tmp_path: Path, multiplier: float
) -> None:
    registry = tmp_path / "registry.jsonl"
    candidate = _sweep(registry, daily_cost=0.0001)

    with pytest.raises(ValueError, match="cost_multiplier"):
        _gate9(registry, candidate, cost_multiplier=multiplier)
