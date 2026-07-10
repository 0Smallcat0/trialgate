from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trialgate import (
    TrialRegistryError,
    append_trial,
    config_hash_for,
    load_trials,
    trial_count,
)

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)


def test_trial_registry_assigns_monotonic_ids_and_round_trips(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"

    first = append_trial(
        registry_path,
        recorded_at=_NOW,
        config_hash="hash-a",
        code_version="abc1234",
        strategy_id="daily_trend_ensemble",
        parameters={"lookbacks": "20,65,150,200"},
        universe=("BTCUSDT", "ETHUSDT"),
        data_start=_NOW - timedelta(days=400),
        data_end=_NOW - timedelta(days=10),
        cost_assumptions={"fee_bps": "10"},
        metrics={"annualized_sharpe": "1.1"},
        operator_note="first",
    )
    second = append_trial(
        registry_path,
        recorded_at=_NOW,
        config_hash="hash-a",
        code_version="abc1234",
        strategy_id="daily_trend_ensemble",
        parameters={"lookbacks": "20,65,150,200"},
        universe=("BTCUSDT", "ETHUSDT"),
        data_start=_NOW - timedelta(days=400),
        data_end=_NOW - timedelta(days=10),
        cost_assumptions={"fee_bps": "20"},
        metrics={"annualized_sharpe": "0.9"},
        operator_note="cost stress",
    )

    assert (first.trial_id, second.trial_id) == (1, 2)
    assert trial_count(registry_path) == 2
    loaded = load_trials(registry_path)
    assert loaded == (first, second)


def test_config_hash_is_deterministic_and_content_sensitive() -> None:
    snapshot_a = {"strategy": {"name": "daily_trend_ensemble"}, "version": "1"}
    snapshot_b = {"version": "1", "strategy": {"name": "daily_trend_ensemble"}}
    snapshot_c = {"strategy": {"name": "large_liquid_trend_15"}, "version": "1"}

    assert config_hash_for(snapshot_a) == config_hash_for(snapshot_b)
    assert config_hash_for(snapshot_a) != config_hash_for(snapshot_c)


def test_append_trial_rejects_naive_timestamps(tmp_path: Path) -> None:
    with pytest.raises(TrialRegistryError, match="timezone-aware"):
        append_trial(
            tmp_path / "registry.jsonl",
            recorded_at=datetime(2026, 7, 2, 12, 0),
            config_hash="hash-a",
            code_version="abc1234",
            strategy_id="daily_trend_ensemble",
            parameters={},
            universe=("BTCUSDT",),
            data_start=_NOW - timedelta(days=400),
            data_end=_NOW - timedelta(days=10),
            cost_assumptions={},
            metrics={},
            operator_note="",
        )


def test_append_trial_rejects_empty_strategy_and_universe(tmp_path: Path) -> None:
    with pytest.raises(TrialRegistryError, match="strategy_id"):
        append_trial(
            tmp_path / "registry.jsonl",
            recorded_at=_NOW,
            config_hash="hash-a",
            code_version="abc1234",
            strategy_id="  ",
            parameters={},
            universe=("BTCUSDT",),
            data_start=_NOW - timedelta(days=400),
            data_end=_NOW - timedelta(days=10),
            cost_assumptions={},
            metrics={},
            operator_note="",
        )
    with pytest.raises(TrialRegistryError, match="universe"):
        append_trial(
            tmp_path / "registry.jsonl",
            recorded_at=_NOW,
            config_hash="hash-a",
            code_version="abc1234",
            strategy_id="daily_trend_ensemble",
            parameters={},
            universe=(),
            data_start=_NOW - timedelta(days=400),
            data_end=_NOW - timedelta(days=10),
            cost_assumptions={},
            metrics={},
            operator_note="",
        )


def test_load_trials_rejects_corrupt_lines(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(TrialRegistryError, match="not valid JSON"):
        load_trials(registry_path)


def test_missing_registry_counts_zero(tmp_path: Path) -> None:
    assert trial_count(tmp_path / "absent.jsonl") == 0
    assert load_trials(tmp_path / "absent.jsonl") == ()
