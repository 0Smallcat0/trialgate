from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trialgate import (
    RegistryTamperError,
    TrialRegistryError,
    append_trial,
    performance_matrix,
    probability_of_backtest_overfitting,
)

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
_START = _NOW - timedelta(days=400)
_END = _NOW - timedelta(days=10)


def _append(
    path: Path,
    note: str,
    returns: list[float] | None,
    *,
    strategy_id: str = "trend",
    data_end: datetime = _END,
) -> None:
    append_trial(
        path,
        recorded_at=_NOW,
        config_hash="hash-a",
        code_version="abc1234",
        strategy_id=strategy_id,
        parameters={},
        universe=("BTCUSDT",),
        data_start=_START,
        data_end=data_end,
        cost_assumptions={},
        metrics={},
        operator_note=note,
        period_returns=returns,
    )


def _series(seed: int, periods: int = 40) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.01) for _ in range(periods)]


def test_matrix_is_assembled_from_the_registry_in_registration_order(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    first, second = _series(1), _series(2)
    _append(registry_path, "a", first)
    _append(registry_path, "b", second)

    matrix = performance_matrix(registry_path)

    assert len(matrix) == 40
    assert [row[0] for row in matrix] == first
    assert [row[1] for row in matrix] == second


def test_matrix_feeds_pbo_directly(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    for seed in range(6):
        _append(registry_path, f"config {seed}", _series(seed, periods=200))

    result = probability_of_backtest_overfitting(performance_matrix(registry_path), block_count=8)

    assert result.strategies == 6
    assert 0.0 <= result.pbo <= 1.0


def test_unregistered_returns_make_a_matrix_impossible(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "with returns", _series(1))
    _append(registry_path, "the one that got away", None)

    with pytest.raises(TrialRegistryError, match="flattering subset"):
        performance_matrix(registry_path)


def test_strategy_id_selects_a_family_without_dropping_its_members(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "a", _series(1), strategy_id="trend")
    _append(registry_path, "b", _series(2), strategy_id="trend")
    _append(registry_path, "unrelated", None, strategy_id="carry")

    matrix = performance_matrix(registry_path, strategy_id="trend")

    assert len(matrix[0]) == 2
    with pytest.raises(TrialRegistryError, match="flattering subset"):
        performance_matrix(registry_path, strategy_id="carry")


def test_misaligned_columns_are_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "a", _series(1, periods=40))
    _append(registry_path, "b", _series(2, periods=39))

    with pytest.raises(TrialRegistryError, match="aligned columns"):
        performance_matrix(registry_path)


def test_differing_data_spans_are_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "a", _series(1))
    _append(registry_path, "b", _series(2), data_end=_END - timedelta(days=30))

    with pytest.raises(TrialRegistryError, match="aligned columns"):
        performance_matrix(registry_path)


def test_editing_a_stored_series_is_detected(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "a", _series(1))
    _append(registry_path, "b", _series(2))

    series_path = tmp_path / "registry.jsonl.returns" / "1.json"
    doctored = json.loads(series_path.read_text(encoding="utf-8"))
    doctored[0] = 0.99
    series_path.write_text(json.dumps(doctored), encoding="utf-8")

    with pytest.raises(RegistryTamperError, match="edited since it was"):
        performance_matrix(registry_path)


def test_deleting_a_stored_series_is_detected(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "a", _series(1))
    (tmp_path / "registry.jsonl.returns" / "1.json").unlink()

    with pytest.raises(TrialRegistryError, match="is missing"):
        performance_matrix(registry_path)


def test_empty_registry_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrialRegistryError, match="no registered trials"):
        performance_matrix(tmp_path / "absent.jsonl")


def test_short_series_are_rejected_at_registration(tmp_path: Path) -> None:
    with pytest.raises(TrialRegistryError, match="at least 2 observations"):
        _append(tmp_path / "registry.jsonl", "too short", [0.01])
