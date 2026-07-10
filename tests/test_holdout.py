from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trialgate import (
    HoldoutViolationError,
    initialize_holdout,
    load_holdout,
    require_data_allowed,
    spend_holdout,
)

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)


def test_holdout_lock_is_single_use(tmp_path: Path) -> None:
    lock_path = tmp_path / "holdout.json"
    holdout_start = _NOW - timedelta(days=365)

    state = initialize_holdout(lock_path, holdout_start=holdout_start, locked_at=_NOW)
    assert state.spent is False

    # Regular runs must stay strictly before the holdout.
    require_data_allowed(lock_path, data_end=holdout_start - timedelta(days=1))
    with pytest.raises(HoldoutViolationError, match="locked holdout"):
        require_data_allowed(lock_path, data_end=holdout_start + timedelta(days=1))

    spent = spend_holdout(lock_path, spent_at=_NOW)
    assert spent.spent is True
    with pytest.raises(HoldoutViolationError, match="already spent"):
        spend_holdout(lock_path, spent_at=_NOW + timedelta(hours=1))
    # Even after the spend, holdout data stays off-limits to regular runs.
    with pytest.raises(HoldoutViolationError, match="SPENT"):
        require_data_allowed(lock_path, data_end=holdout_start + timedelta(days=1))

    # Re-initializing never resets an existing lock.
    again = initialize_holdout(lock_path, holdout_start=_NOW - timedelta(days=30), locked_at=_NOW)
    assert again.holdout_start == holdout_start
    assert again.spent is True


def test_spending_an_uninitialized_holdout_fails(tmp_path: Path) -> None:
    with pytest.raises(HoldoutViolationError, match="never initialized"):
        spend_holdout(tmp_path / "holdout.json", spent_at=_NOW)


def test_missing_lock_allows_all_data(tmp_path: Path) -> None:
    assert load_holdout(tmp_path / "holdout.json") is None
    require_data_allowed(tmp_path / "holdout.json", data_end=_NOW)


def test_holdout_rejects_naive_timestamps(tmp_path: Path) -> None:
    with pytest.raises(HoldoutViolationError, match="timezone-aware"):
        initialize_holdout(
            tmp_path / "holdout.json",
            holdout_start=datetime(2025, 7, 2),
            locked_at=_NOW,
        )


def test_corrupt_lock_raises_instead_of_unlocking(tmp_path: Path) -> None:
    lock_path = tmp_path / "holdout.json"
    lock_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(HoldoutViolationError, match="corrupt"):
        load_holdout(lock_path)
