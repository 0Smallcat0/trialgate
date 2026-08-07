from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trialgate import (
    RegistryTamperError,
    TrialRecord,
    append_trial,
    load_trials,
    verify_chain,
)
from trialgate.__main__ import main

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
_GENESIS = "0" * 64


def _append(path: Path, note: str, sharpe: str = "1.1") -> TrialRecord:
    return append_trial(
        path,
        recorded_at=_NOW,
        config_hash="hash-a",
        code_version="abc1234",
        strategy_id="daily_trend_ensemble",
        parameters={"lookbacks": "20,65,150,200"},
        universe=("BTCUSDT",),
        data_start=_NOW - timedelta(days=400),
        data_end=_NOW - timedelta(days=10),
        cost_assumptions={"fee_bps": "10"},
        metrics={"annualized_sharpe": sharpe},
        operator_note=note,
    )


def _write_legacy(path: Path, trial_id: int, note: str) -> None:
    """A record in the pre-chain format: no prev_hash field at all."""

    record = TrialRecord(
        trial_id=trial_id,
        recorded_at=_NOW,
        config_hash="hash-legacy",
        code_version="0000000",
        strategy_id="daily_trend_ensemble",
        parameters={},
        universe=("BTCUSDT",),
        data_start=_NOW - timedelta(days=400),
        data_end=_NOW - timedelta(days=10),
        cost_assumptions={},
        metrics={"annualized_sharpe": "0.8"},
        operator_note=note,
    )
    assert "prev_hash" not in record.to_json_dict()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def _rewrite_line(path: Path, index: int, mutate: str | None) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutate is None:
        del lines[index]
    else:
        row = json.loads(lines[index])
        row["operator_note"] = mutate
        lines[index] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_records_chain_to_their_predecessor(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"

    first = _append(registry_path, "first")
    second = _append(registry_path, "second")

    assert first.prev_hash == _GENESIS
    assert second.prev_hash == first.entry_hash
    assert verify_chain(registry_path).head == second.entry_hash
    assert load_trials(registry_path) == (first, second)


def test_absent_registry_verifies_as_empty(tmp_path: Path) -> None:
    report = verify_chain(tmp_path / "absent.jsonl")

    assert report.ok
    assert (report.total_trials, report.sealed_trials, report.head) == (0, 0, _GENESIS)


def test_editing_an_earlier_trial_breaks_the_chain(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "second")

    _rewrite_line(registry_path, 0, "retconned")

    with pytest.raises(RegistryTamperError, match="breaks the hash chain"):
        load_trials(registry_path)


def test_deleting_a_trial_breaks_the_chain(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "embarrassing", sharpe="0.1")
    _append(registry_path, "flattering", sharpe="2.4")

    _rewrite_line(registry_path, 0, None)

    with pytest.raises(RegistryTamperError, match="breaks the hash chain"):
        load_trials(registry_path)


def test_reordering_trials_breaks_the_chain(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "second")
    _append(registry_path, "third")

    lines = registry_path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RegistryTamperError, match="breaks the hash chain"):
        load_trials(registry_path)


def test_appending_to_a_tampered_registry_is_refused(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "second")
    _rewrite_line(registry_path, 0, "retconned")

    with pytest.raises(RegistryTamperError):
        _append(registry_path, "laundering append")


def test_first_chained_append_seals_the_legacy_history(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _write_legacy(registry_path, 1, "pre-chain trial")
    _write_legacy(registry_path, 2, "pre-chain trial")

    # Legacy-only registries load: nothing to verify, nothing sealed.
    assert len(load_trials(registry_path)) == 2
    assert verify_chain(registry_path).sealed_trials == 0

    _append(registry_path, "first chained trial")

    report = verify_chain(registry_path)
    assert (report.total_trials, report.sealed_trials) == (3, 2)

    _rewrite_line(registry_path, 0, "retconned legacy trial")
    with pytest.raises(RegistryTamperError, match="breaks the hash chain"):
        load_trials(registry_path)


def test_tail_edits_are_not_detectable_without_an_external_anchor(tmp_path: Path) -> None:
    """Documents the known limit: nothing anchors the last record locally."""

    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "last")

    _rewrite_line(registry_path, 1, "quietly rewritten")

    trials = load_trials(registry_path)
    assert trials[1].operator_note == "quietly rewritten"
    report = verify_chain(registry_path)
    assert report.ok
    # ...which is exactly why the report never claims the tail is sealed.
    assert report.sealed_trials == 1
    assert "truncation is undetectable" in report.summary()


def test_verify_chain_reports_a_broken_chain_instead_of_raising(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "second")
    _rewrite_line(registry_path, 0, "retconned")

    report = verify_chain(registry_path)

    assert not report.ok
    assert report.problem is not None
    assert "void" in report.summary()


def test_cli_exit_status_reflects_chain_integrity(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _append(registry_path, "first")
    _append(registry_path, "second")

    assert main(["verify", str(registry_path)]) == 0

    _rewrite_line(registry_path, 0, "retconned")
    assert main(["verify", str(registry_path)]) == 1
