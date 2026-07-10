"""Single-use final holdout lock: iterated out-of-sample is not out-of-sample.

Gate 5 of the six-gate qualification standard. The most recent slice of data
(typically ~12 months) is locked at first backtest; unlocking is a single-use,
logged, irreversible event. After the spend, holdout data stays off-limits to
regular runs forever (Arnott, Harvey & Markowitz 2019).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class HoldoutViolationError(ValueError):
    """Raised when a run would touch locked or already-spent holdout data."""


@dataclass(frozen=True, slots=True)
class HoldoutState:
    """Current holdout lock state."""

    holdout_start: datetime
    locked_at: datetime
    spent: bool
    spent_at: datetime | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "holdout_start": self.holdout_start.isoformat(),
            "locked_at": self.locked_at.isoformat(),
            "spent": self.spent,
            "spent_at": self.spent_at.isoformat() if self.spent_at else None,
        }


def initialize_holdout(
    path: str | Path,
    *,
    holdout_start: datetime,
    locked_at: datetime,
) -> HoldoutState:
    """Create the lock once; loading the existing lock on later calls."""

    _require_utc("holdout_start", holdout_start)
    _require_utc("locked_at", locked_at)
    existing = load_holdout(path)
    if existing is not None:
        return existing
    state = HoldoutState(
        holdout_start=holdout_start,
        locked_at=locked_at,
        spent=False,
        spent_at=None,
    )
    _write(path, state)
    return state


def load_holdout(path: str | Path) -> HoldoutState | None:
    """Load the lock state, or None when no lock exists yet."""

    lock_path = Path(path)
    if not lock_path.exists():
        return None
    try:
        row = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"holdout lock is corrupt: {lock_path}"
        raise HoldoutViolationError(msg) from exc
    if not isinstance(row, dict):
        msg = f"holdout lock must be a JSON object: {lock_path}"
        raise HoldoutViolationError(msg)
    spent_at_raw = row.get("spent_at")
    return HoldoutState(
        holdout_start=_parse_utc(str(row["holdout_start"])),
        locked_at=_parse_utc(str(row["locked_at"])),
        spent=bool(row["spent"]),
        spent_at=_parse_utc(str(spent_at_raw)) if spent_at_raw else None,
    )


def require_data_allowed(path: str | Path, *, data_end: datetime) -> None:
    """Reject any regular run whose data reaches into the holdout period."""

    _require_utc("data_end", data_end)
    state = load_holdout(path)
    if state is None:
        return
    if data_end >= state.holdout_start:
        spent_note = (
            "the holdout has already been SPENT; re-testing against it is void"
            if state.spent
            else "unlock it only through the single-use spend procedure"
        )
        msg = (
            f"data_end {data_end.isoformat()} reaches into the locked holdout "
            f"(starts {state.holdout_start.isoformat()}); {spent_note}"
        )
        raise HoldoutViolationError(msg)


def spend_holdout(path: str | Path, *, spent_at: datetime) -> HoldoutState:
    """Mark the single-use holdout as spent. A second spend attempt fails.

    The spend is recorded BEFORE the qualification run executes: if the run
    crashes, the holdout stays spent (conservative by doctrine).
    """

    _require_utc("spent_at", spent_at)
    state = load_holdout(path)
    if state is None:
        msg = "cannot spend a holdout that was never initialized"
        raise HoldoutViolationError(msg)
    if state.spent:
        msg = (
            f"holdout was already spent at "
            f"{state.spent_at.isoformat() if state.spent_at else 'unknown'}; "
            "a second use is void (iterated OOS is not OOS)"
        )
        raise HoldoutViolationError(msg)
    spent_state = HoldoutState(
        holdout_start=state.holdout_start,
        locked_at=state.locked_at,
        spent=True,
        spent_at=spent_at,
    )
    _write(path, spent_state)
    return spent_state


def _write(path: str | Path, state: HoldoutState) -> None:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(state.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        msg = f"holdout timestamps must be timezone-aware: {raw}"
        raise HoldoutViolationError(msg)
    return value.astimezone(UTC)


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        msg = f"{name} must be timezone-aware UTC"
        raise HoldoutViolationError(msg)
