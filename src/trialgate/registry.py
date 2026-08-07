"""Append-only trial registry: unregistered backtests are void.

Gate 1 of the six-gate qualification standard. Every backtest execution of
any variant must be recorded here; the registry maintains the running trial
count N used by the DSR gate. A backtest that does not disclose its trial
count is worthless by construction (Bailey & Lopez de Prado 2014).

Each appended record carries ``prev_hash``: the SHA-256 chain value over
every preceding line of the file as written on disk. Deleting, editing, or
reordering an earlier trial breaks verification at the first record written
afterwards, which is what makes N auditable rather than merely asserted.

The chain does not prove that trials were registered at all, and it cannot
detect truncation of the tail — a local file has no external anchor, so
dropping the last k records leaves a shorter but consistent chain. Commit
the registry after every run to close that.

Legacy records carry no ``prev_hash`` but are still hashed into the chain,
so the first chained append seals the history behind it. Migration is never
required; rewriting old records to add hashes would itself be tampering.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from trialgate.holdout import require_data_allowed

_GENESIS = "0" * 64


class TrialRegistryError(ValueError):
    """Raised when the trial registry cannot be read or appended safely."""


class RegistryTamperError(TrialRegistryError):
    """Raised when the registry's hash chain does not match its contents.

    The registry has been edited, reordered, or had records removed since the
    recorded trial was appended. Any trial count or gate verdict derived from
    it is void.
    """


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One registered backtest execution.

    ``prev_hash`` is the chain value over every line preceding this one; it is
    ``None`` for legacy records written before chaining. ``entry_hash`` is the
    chain value including this line, computed at load time and never stored
    (storing it inside the line it covers would be self-referential).
    """

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
    prev_hash: str | None = None
    entry_hash: str | None = None
    returns_sha256: str | None = None
    returns_count: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
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
        if self.prev_hash is not None:
            row["prev_hash"] = self.prev_hash
        if self.returns_sha256 is not None:
            row["returns_sha256"] = self.returns_sha256
            row["returns_count"] = self.returns_count
        return row


@dataclass(frozen=True, slots=True)
class ChainReport:
    """Audit result for a registry's hash chain.

    ``sealed_trials`` is the trust boundary: trials 1..sealed_trials cannot be
    altered without breaking the chain. Anything after it can.
    """

    path: str
    ok: bool
    total_trials: int
    sealed_trials: int
    head: str
    problem: str | None

    def summary(self) -> str:
        if not self.ok:
            return (
                f"FAIL  {self.path}\n"
                f"      {self.problem}\n"
                "      Trial count and every gate verdict derived from this "
                "registry are void."
            )
        lines = [
            f"OK    {self.path}",
            f"      {self.total_trials} trials, sealed against edits: "
            + (f"1-{self.sealed_trials}" if self.sealed_trials else "none"),
            f"      head: {self.head}",
        ]
        if self.total_trials > self.sealed_trials:
            lines.append(
                f"      trials {self.sealed_trials + 1}-{self.total_trials} are still "
                "editable and truncation is undetectable; commit this file after "
                "every run"
            )
        return "\n".join(lines)


def config_hash_for(snapshot: Mapping[str, object]) -> str:
    """Deterministic SHA-256 of a JSON-serializable config snapshot."""

    return _sha256(json.dumps(dict(snapshot), sort_keys=True, separators=(",", ":")))


def trial_count(path: str | Path) -> int:
    """Current registered trial count N. Raises if the chain is broken."""

    return len(load_trials(path))


def verify_chain(path: str | Path) -> ChainReport:
    """Audit the registry's hash chain without raising.

    Returns a report suitable for a third party: whether the chain is intact,
    how many trials are cryptographically sealed against edits, and the
    current head. Use this to check a registry someone handed you.
    """

    registry_path = Path(path)
    try:
        records, head = _scan(registry_path)
    except TrialRegistryError as exc:
        return ChainReport(
            path=str(registry_path),
            ok=False,
            total_trials=0,
            sealed_trials=0,
            head=_GENESIS,
            problem=str(exc),
        )
    anchored = [record for record in records if record.prev_hash is not None]
    # Everything before the last anchored record is fixed by that record's
    # prev_hash. The anchored record itself, and anything after it, is not.
    return ChainReport(
        path=str(registry_path),
        ok=True,
        total_trials=len(records),
        sealed_trials=anchored[-1].trial_id - 1 if anchored else 0,
        head=head,
        problem=None,
    )


def load_trials(path: str | Path) -> tuple[TrialRecord, ...]:
    """Load every registered trial, oldest first.

    Verifies the hash chain and raises RegistryTamperError on any mismatch.
    There is deliberately no way to skip the check.
    """

    records, _ = _scan(path)
    return records


def append_trial(
    path: str | Path,
    *,
    recorded_at: datetime,
    code_version: str,
    strategy_id: str,
    parameters: dict[str, str],
    universe: tuple[str, ...],
    data_start: datetime,
    data_end: datetime,
    cost_assumptions: dict[str, str],
    metrics: dict[str, str],
    operator_note: str,
    period_returns: Sequence[float] | None = None,
    config_hash: str | None = None,
    holdout: str | Path | None = None,
) -> TrialRecord:
    """Append one trial record and return it with its id and chain hashes.

    Pass ``period_returns`` — this configuration's per-period (e.g. daily)
    return series — to make the trial usable by gate 3. The series is stored
    beside the registry and its SHA-256 goes into the chained record, so it is
    as tamper-evident as the record itself, and ``performance_matrix()`` can
    later assemble the CSCV input from the registry instead of from whatever
    columns the operator feels like handing over.

    Pass ``holdout`` — the path to the lock file — to enforce gate 5 here, at
    the only place that sees every run. Registering a trial whose ``data_end``
    reaches into the locked period then raises instead of relying on the
    operator to remember a separate check.

    ``config_hash`` defaults to a hash of ``parameters``. Pass it explicitly
    only when the configuration is bigger than the parameters you log.

    Refuses to append to a registry whose chain is already broken: extending a
    tampered log would launder it.
    """

    _require_utc("recorded_at", recorded_at)
    _require_utc("data_start", data_start)
    _require_utc("data_end", data_end)
    if not strategy_id.strip():
        msg = "strategy_id must not be empty"
        raise TrialRegistryError(msg)
    if not universe:
        msg = "universe must not be empty"
        raise TrialRegistryError(msg)

    if period_returns is not None and len(period_returns) < 2:
        msg = "period_returns must contain at least 2 observations"
        raise TrialRegistryError(msg)
    if holdout is not None:
        require_data_allowed(holdout, data_end=data_end)

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing, head = _scan(registry_path)
    trial_id = len(existing) + 1
    returns_text = (
        None if period_returns is None else json.dumps([float(v) for v in period_returns])
    )
    record = TrialRecord(
        trial_id=trial_id,
        recorded_at=recorded_at,
        config_hash=config_hash if config_hash is not None else config_hash_for(parameters),
        code_version=code_version,
        strategy_id=strategy_id,
        parameters=parameters,
        universe=universe,
        data_start=data_start,
        data_end=data_end,
        cost_assumptions=cost_assumptions,
        metrics=metrics,
        operator_note=operator_note,
        prev_hash=head,
        returns_sha256=None if returns_text is None else _sha256(returns_text),
        returns_count=None if period_returns is None else len(period_returns),
    )
    if returns_text is not None:
        returns_path = _returns_path(registry_path, trial_id)
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        returns_path.write_text(returns_text, encoding="utf-8")
    line = json.dumps(record.to_json_dict(), sort_keys=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return replace(record, entry_hash=_chain_step(head, line))


def performance_matrix(
    path: str | Path,
    *,
    strategy_id: str | None = None,
) -> list[list[float]]:
    """Assemble the T x N CSCV input from the registry itself.

    One column per registered trial, in registration order. This is the point:
    gate 3 is then computed over exactly the trials that were registered, so
    dropping the unflattering configurations from the comparison is not a thing
    an operator can do — there is no matrix to hand over.

    Raises if any selected trial has no stored returns (the matrix would be a
    subset of the registry), if the series disagree on length or data span, or
    if a stored series no longer matches the hash in its chained record.
    """

    registry_path = Path(path)
    trials = [
        trial
        for trial in load_trials(registry_path)
        if strategy_id is None or trial.strategy_id == strategy_id
    ]
    if not trials:
        msg = f"no registered trials in {registry_path}" + (
            f" for strategy_id {strategy_id!r}" if strategy_id else ""
        )
        raise TrialRegistryError(msg)

    missing = [trial.trial_id for trial in trials if trial.returns_sha256 is None]
    if missing:
        msg = (
            f"{len(missing)} of {len(trials)} registered trials have no stored "
            f"period_returns (trial ids {missing[:5]}{'...' if len(missing) > 5 else ''}); "
            "a matrix built from the rest would be a flattering subset of the "
            "registry, which voids gate 3. Re-register them with period_returns, "
            "or pass strategy_id to select a family that has them."
        )
        raise TrialRegistryError(msg)

    first = trials[0]
    for trial in trials[1:]:
        if (trial.returns_count, trial.data_start, trial.data_end) != (
            first.returns_count,
            first.data_start,
            first.data_end,
        ):
            msg = (
                f"trial {trial.trial_id} spans "
                f"{trial.data_start.date()}..{trial.data_end.date()} with "
                f"{trial.returns_count} periods, but trial {first.trial_id} spans "
                f"{first.data_start.date()}..{first.data_end.date()} with "
                f"{first.returns_count}; CSCV needs aligned columns"
            )
            raise TrialRegistryError(msg)

    columns = [_load_returns(registry_path, trial) for trial in trials]
    return [[column[period] for column in columns] for period in range(len(columns[0]))]


def _returns_path(registry_path: Path, trial_id: int) -> Path:
    return registry_path.parent / f"{registry_path.name}.returns" / f"{trial_id}.json"


def _load_returns(registry_path: Path, trial: TrialRecord) -> list[float]:
    returns_path = _returns_path(registry_path, trial.trial_id)
    if not returns_path.exists():
        msg = f"trial {trial.trial_id} records stored returns, but {returns_path} is missing"
        raise TrialRegistryError(msg)
    text = returns_path.read_text(encoding="utf-8")
    if _sha256(text) != trial.returns_sha256:
        msg = (
            f"{returns_path} does not match the SHA-256 recorded for trial "
            f"{trial.trial_id}; the return series has been edited since it was "
            "registered, so every gate verdict derived from it is void"
        )
        raise RegistryTamperError(msg)
    return [float(value) for value in json.loads(text)]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan(path: str | Path) -> tuple[tuple[TrialRecord, ...], str]:
    """Parse every line, threading and verifying the hash chain through it."""

    registry_path = Path(path)
    if not registry_path.exists():
        return (), _GENESIS
    records: list[TrialRecord] = []
    running = _GENESIS
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
        record = _record_from_row(row, registry_path, line_number)
        if record.prev_hash is not None and record.prev_hash != running:
            msg = (
                f"{registry_path}:{line_number} breaks the hash chain: trial "
                f"{record.trial_id} was appended after a history hashing to "
                f"{record.prev_hash[:12]}..., but the preceding {line_number - 1} "
                f"line(s) hash to {running[:12]}.... Earlier trials have been "
                "edited, reordered, or removed."
            )
            raise RegistryTamperError(msg)
        running = _chain_step(running, line)
        records.append(replace(record, entry_hash=running))
    return tuple(records), running


def _chain_step(prev: str, line: str) -> str:
    return hashlib.sha256(f"{prev}\n{line}".encode()).hexdigest()


def _record_from_row(row: object, path: Path, line_number: int) -> TrialRecord:
    if not isinstance(row, dict):
        msg = f"{path}:{line_number} must be a JSON object"
        raise TrialRegistryError(msg)
    prev_hash_raw = row.get("prev_hash")
    returns_hash_raw = row.get("returns_sha256")
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
            prev_hash=str(prev_hash_raw) if prev_hash_raw is not None else None,
            returns_sha256=None if returns_hash_raw is None else str(returns_hash_raw),
            returns_count=None if returns_hash_raw is None else int(row["returns_count"]),
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
