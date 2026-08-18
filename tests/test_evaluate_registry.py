"""The one-call path a stranger takes: register trials, then ask for a verdict."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trialgate import append_trial, evaluate_registry, load_trials

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
_START = datetime(2019, 1, 1, tzinfo=UTC)
_END = datetime(2025, 6, 30, tzinfo=UTC)


def _register_noise_sweep(
    path: Path, configs: int = 12, periods: int = 300, round_trip_bps: str = "25"
) -> None:
    """A random walk filtered by N different lookbacks: no edge exists."""

    rng = random.Random(11)
    prices = [100.0]
    for _ in range(periods):
        prices.append(prices[-1] * (1.0 + rng.gauss(0.0, 0.012)))
    raw = [prices[index + 1] / prices[index] - 1.0 for index in range(periods)]

    for config in range(configs):
        lookback = 2 + config
        returns = [
            raw[day] if day >= lookback and prices[day] - prices[day - lookback] > 0 else 0.0
            for day in range(periods)
        ]
        append_trial(
            path,
            recorded_at=_NOW,
            code_version="test",
            strategy_id="noise",
            parameters={"lookback": str(lookback)},
            universe=("SYNTH",),
            data_start=_START,
            data_end=_END,
            cost_assumptions={"round_trip_bps": round_trip_bps},
            metrics={},
            operator_note=f"config {config}",
            period_returns=returns,
        )


def test_a_noise_sweep_is_not_qualified(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path)
    trials = load_trials(registry_path)

    report = evaluate_registry(
        registry_path,
        candidate_trial_id=trials[-1].trial_id,
        periods_per_year=252,
        effective_trials=len(trials),
        block_count=8,
        min_observations=100,
    )

    assert not report.passed
    assert [check.gate for check in report.checks] == [
        "data floor (gate 2)",
        "PBO (gate 3)",
        "DSR (gate 4)",
        "declared costs (gate 8)",
    ]
    assert "NOT QUALIFIED" in report.summary()


def test_a_gross_of_cost_registry_fails_the_cost_gate(tmp_path: Path) -> None:
    """Returns that were never charged for execution are not returns."""

    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path, configs=4, round_trip_bps="0")

    report = evaluate_registry(
        registry_path,
        candidate_trial_id=1,
        periods_per_year=252,
        effective_trials=4,
        block_count=4,
        min_observations=100,
    )
    costs = next(check for check in report.checks if check.gate.startswith("declared costs"))

    assert not costs.passed
    assert costs.observed == 0.0
    assert costs.threshold == 4.0
    assert "priced execution at zero" in costs.detail


def test_the_cost_gate_can_be_waived_but_not_by_accident(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path, configs=4, round_trip_bps="0")

    report = evaluate_registry(
        registry_path,
        candidate_trial_id=1,
        periods_per_year=252,
        effective_trials=4,
        block_count=4,
        min_observations=100,
        require_declared_costs=False,
    )

    assert not any(check.gate.startswith("declared costs") for check in report.checks)


def test_the_stop_line_gate_is_absent_until_a_policy_is_given(tmp_path: Path) -> None:
    """No default stop line: the library measures a policy, it does not invent one."""

    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path, configs=4)

    without = evaluate_registry(
        registry_path,
        candidate_trial_id=1,
        periods_per_year=252,
        effective_trials=4,
        block_count=4,
        min_observations=100,
    )
    with_policy = evaluate_registry(
        registry_path,
        candidate_trial_id=1,
        periods_per_year=252,
        effective_trials=4,
        block_count=4,
        min_observations=100,
        stop_loss_fraction=0.15,
    )
    stop = next(
        check for check in with_policy.checks if check.gate.startswith("stop-line survivability")
    )

    assert not any(check.gate.startswith("stop-line") for check in without.checks)
    assert 0.0 <= stop.observed <= 1.0
    assert "survivable volatility" in stop.detail


def test_the_data_floor_reads_the_registered_series_length(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path, periods=300)

    report = evaluate_registry(
        registry_path,
        candidate_trial_id=1,
        periods_per_year=252,
        effective_trials=12,
        block_count=8,
        min_observations=1000,
    )
    floor = next(check for check in report.checks if check.gate.startswith("data floor"))

    assert floor.observed == 300.0
    assert not floor.passed


def test_an_unregistered_candidate_is_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path, configs=3)

    with pytest.raises(ValueError, match="not registered"):
        evaluate_registry(
            registry_path,
            candidate_trial_id=99,
            periods_per_year=252,
            effective_trials=3,
            block_count=8,
        )


def test_more_assumed_trials_can_only_make_the_verdict_harsher(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register_noise_sweep(registry_path)

    def dsr_for(effective_trials: int) -> float:
        report = evaluate_registry(
            registry_path,
            candidate_trial_id=1,
            periods_per_year=252,
            effective_trials=effective_trials,
            block_count=8,
            min_observations=100,
        )
        return next(check for check in report.checks if check.gate.startswith("DSR")).observed

    assert dsr_for(12) < dsr_for(1)
