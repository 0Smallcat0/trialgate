"""Append-only trial registry: unregistered backtests are void.

Gate 1 of the six-gate qualification standard. Every backtest execution of
any variant must be recorded here; the registry maintains the running trial
count N used by the DSR gate. A backtest that does not disclose its trial
count is worthless by construction (Bailey & Lopez de Prado 2014).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class TrialRegistryError(ValueError):
    """Raised when the trial registry cannot be read or appended safely."""


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One registered backtest execution."""

    trial_id: int
    recorded_at: datetime
    config_hash: str
    code_version: str
    strategy_id: str
    parameters: dict[str, str]
    universe: tuple[str, ...]
    data_start: datetime
    data_end: datetime
    cost_assumptions: dict[str, str]
    metrics: dict[str, str]
    operator_note: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "recorded_at": self.recorded_at.isoformat(),
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "strategy_id": self.strategy_id,
            "parameters": dict(self.parameters),
            "universe": list(self.universe),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "cost_assumptions": dict(self.cost_assumptions),
            "metrics": dict(self.metrics),
            "operator_note": self.operator_note,
        }


def config_hash_for(snapshot: dict[str, object]) -> str:
    """Deterministic SHA-256 of a JSON-serializable config snapshot."""

    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def trial_count(path: str | Path) -> int:
    """Current registered trial count N."""

    return len(load_trials(path))


def load_trials(path: str | Path) -> tuple[TrialRecord, ...]:
    """Load every registered trial, oldest first."""

    registry_path = Path(path)
    if not registry_path.exists():
        return ()
    records: list[TrialRecord] = []
    for line_number, line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{registry_path}:{line_number} is not valid JSON"
            raise TrialRegistryError(msg) from exc
        records.append(_record_from_row(row, registry_path, line_number))
    return tuple(records)


def append_trial(
    path: str | Path,
    *,
    recorded_at: datetime,
    config_hash: str,
    code_version: str,
    strategy_id: str,
    parameters: dict[str, str],
    universe: tuple[str, ...],
    data_start: datetime,
    data_end: datetime,
    cost_assumptions: dict[str, str],
    metrics: dict[str, str],
    operator_note: str,
) -> TrialRecord:
    """Append one trial record and return it with its assigned monotonic id."""

    _require_utc("recorded_at", recorded_at)
    _require_utc("data_start", data_start)
    _require_utc("data_end", data_end)
    if not strategy_id.strip():
        msg = "strategy_id must not be empty"
        raise TrialRegistryError(msg)
    if not universe:
        msg = "universe must not be empty"
        raise TrialRegistryError(msg)

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_trials(registry_path)
    record = TrialRecord(
        trial_id=len(existing) + 1,
        recorded_at=recorded_at,
        config_hash=config_hash,
        code_version=code_version,
        strategy_id=strategy_id,
        parameters=parameters,
        universe=universe,
        data_start=data_start,
        data_end=data_end,
        cost_assumptions=cost_assumptions,
        metrics=metrics,
        operator_note=operator_note,
    )
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
    return record


def _record_from_row(row: object, path: Path, line_number: int) -> TrialRecord:
    if not isinstance(row, dict):
        msg = f"{path}:{line_number} must be a JSON object"
        raise TrialRegistryError(msg)
    try:
        return TrialRecord(
            trial_id=int(row["trial_id"]),
            recorded_at=_parse_utc(str(row["recorded_at"])),
            config_hash=str(row["config_hash"]),
            code_version=str(row["code_version"]),
            strategy_id=str(row["strategy_id"]),
            parameters={str(k): str(v) for k, v in dict(row["parameters"]).items()},
            universe=tuple(str(v) for v in row["universe"]),
            data_start=_parse_utc(str(row["data_start"])),
            data_end=_parse_utc(str(row["data_end"])),
            cost_assumptions={str(k): str(v) for k, v in dict(row["cost_assumptions"]).items()},
            metrics={str(k): str(v) for k, v in dict(row["metrics"]).items()},
            operator_note=str(row["operator_note"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"{path}:{line_number} is not a valid trial record: {exc}"
        raise TrialRegistryError(msg) from exc


def _parse_utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        msg = f"trial timestamps must be timezone-aware: {raw}"
        raise TrialRegistryError(msg)
    return value.astimezone(UTC)


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        msg = f"{name} must be timezone-aware UTC"
        raise TrialRegistryError(msg)
