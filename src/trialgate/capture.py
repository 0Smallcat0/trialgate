"""Register trials where they are produced, instead of from memory.

Gate 1 holds only if every backtest reaches the registry, and the backtests
that matter most are the ones nobody types up: the overnight hyperopt, the
Optuna study, the agent loop that tried 300 variants before lunch. This module
reads them off the optimizer that ran them.

- ``import_freqtrade_results()`` reads freqtrade's own hyperopt log
  (``user_data/hyperopt_results/*.fthypt``, one JSON line per evaluated epoch)
  and registers each epoch as a trial, with the daily return series that gate 3
  needs. Epochs already imported are skipped, so it is safe to run after every
  hyperopt.
- ``OptunaRecorder`` is an Optuna callback that registers each trial as the
  study finishes it; ``import_optuna_study()`` registers a study that already
  ran. Optuna itself is not imported: anything shaped like a ``FrozenTrial``
  works, and trialgate keeps no dependencies.

Each imported record carries ``metrics["capture_source"]`` -- where it came
from -- which is what makes a second import of the same file a no-op rather
than a doubled trial count.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeGuard

from trialgate.holdout import require_data_allowed
from trialgate.registry import TrialRecord, TrialRegistryError, append_trial, load_trials

CAPTURE_SOURCE = "capture_source"
_FREQTRADE_TIME = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class ImportReport:
    """What one import added to the registry, and what it had already seen."""

    registry: str
    appended: tuple[TrialRecord, ...]
    already_registered: int
    sources: tuple[str, ...]

    def summary(self) -> str:
        first = self.appended[0].trial_id if self.appended else None
        last = self.appended[-1].trial_id if self.appended else None
        added = (
            f"registered {len(self.appended)} new trials ({first}-{last})"
            if self.appended
            else "registered 0 new trials"
        )
        return (
            f"{added} into {self.registry} from {len(self.sources)} file(s); "
            f"{self.already_registered} were already registered"
        )


def import_freqtrade_results(
    results: str | Path,
    registry: str | Path,
    *,
    code_version: str = "unrecorded",
    holdout: str | Path | None = None,
    recorded_at: datetime | None = None,
) -> ImportReport:
    """Register every epoch of every freqtrade hyperopt run under ``results``.

    ``results`` is a ``.fthypt`` file or the directory freqtrade writes them to
    (``user_data/hyperopt_results``). Every line is one evaluated parameter
    set, including the losing ones and the ones freqtrade scored as too few
    trades -- they were all tried, so they all count toward N. Re-running the
    hyperopt writes a new file, and importing the directory again picks up
    only the epochs not yet registered.

    Each trial's ``period_returns`` is the daily return of the account: the
    profit freqtrade books on each day a trade closes (its own
    ``daily_profit``), over the account's equity at the start of that day,
    with every day of the backtest range present and a zero on days nothing
    closed. Open positions are not marked to market, as in freqtrade's daily
    report. Judge these with ``periods_per_year=365``: crypto trades every day.

    ``cost_assumptions`` records the fee freqtrade charged on each side, read
    from the run's own trades. A run whose epochs never traded leaves it
    empty, which gate 8 then reports.

    ``period_costs`` (gate 9) is the fee each day's closed trades paid, over
    the same equity: every fill's amount times its price times that side's
    fee, booked on the day the trade closes, as its profit is. Funding fees
    are not execution and are left in the returns. An epoch whose results
    carry a trade count but no trade list registers no costs.

    Passing ``holdout`` checks every epoch against the gate-5 lock before
    anything is written, so a hyperopt that reached into the locked period is
    refused whole rather than half-imported.
    """

    files = _fthypt_files(Path(results))
    registry_path = Path(registry)
    seen = _registered_sources(registry_path)
    pending: list[dict[str, Any]] = []
    skipped = 0
    for file in files:
        epochs = _read_fthypt(file)
        fees = _run_fees(epochs)
        for line_number, epoch in epochs:
            source = f"freqtrade:{file.name}#{_epoch_number(epoch, line_number)}"
            if source in seen:
                skipped += 1
                continue
            seen.add(source)
            pending.append(
                _freqtrade_trial(
                    epoch,
                    source=source,
                    where=f"{file}:{line_number}",
                    fees=fees,
                    code_version=code_version,
                    recorded_at=recorded_at,
                )
            )

    if holdout is not None:
        for trial in pending:
            require_data_allowed(holdout, data_end=trial["data_end"])
    appended = tuple(append_trial(registry_path, **trial) for trial in pending)
    return ImportReport(
        registry=str(registry_path),
        appended=appended,
        already_registered=skipped,
        sources=tuple(str(file) for file in files),
    )


class TrialLike(Protocol):
    """The parts of an ``optuna.trial.FrozenTrial`` this module reads."""

    @property
    def number(self) -> int: ...

    @property
    def params(self) -> dict[str, Any]: ...

    @property
    def user_attrs(self) -> dict[str, Any]: ...

    @property
    def values(self) -> list[float] | None: ...

    @property
    def datetime_complete(self) -> datetime | None: ...


class StudyLike(Protocol):
    """The parts of an ``optuna.Study`` this module reads."""

    @property
    def study_name(self) -> str: ...

    def get_trials(self, deepcopy: bool = ...) -> Sequence[Any]: ...


ReturnsGetter = Callable[[Any], Sequence[float] | None]


class OptunaRecorder:
    """Optuna callback: ``study.optimize(objective, callbacks=[recorder])``.

    Registers each finished trial -- complete or pruned; a pruned trial was
    looked at and rejected, which is a try -- with its parameters, its
    objective value and, if the objective stored it, its return series.
    Failed trials (the objective raised) produced no result to select and are
    not registered.

    The return series comes from ``trial.user_attrs["period_returns"]`` by
    default: call ``trial.set_user_attr("period_returns", list(returns))`` in
    the objective. Pass ``returns=`` to read it from somewhere else instead.
    Gate 9's cost series comes the same way from ``"period_costs"``, or
    ``costs=``: what trading cost in each period, as a fraction of equity.

    Everything the trial does not know -- the strategy, the data range, the
    universe, the costs -- is fixed for the study and given here once.

    A pruned trial has no full return series, so a study that prunes still
    counts its pruned trials in N but cannot build gate 3's matrix; attach
    the series before pruning, or do not prune backtests. Appends are
    serialized, so ``n_jobs > 1`` is safe; several processes writing one
    registry are not.
    """

    def __init__(
        self,
        registry: str | Path,
        *,
        strategy_id: str,
        universe: tuple[str, ...],
        data_start: datetime,
        data_end: datetime,
        cost_assumptions: dict[str, str],
        code_version: str = "unrecorded",
        returns: ReturnsGetter | None = None,
        holdout: str | Path | None = None,
        costs: ReturnsGetter | None = None,
    ) -> None:
        self.registry = Path(registry)
        self.strategy_id = strategy_id
        self.universe = universe
        self.data_start = data_start
        self.data_end = data_end
        self.cost_assumptions = dict(cost_assumptions)
        self.code_version = code_version
        self.returns = returns or _user_attr_returns
        self.costs = costs or _user_attr_costs
        self.holdout = holdout
        if holdout is not None:
            # Fail before the first trial runs, not after the hundredth.
            require_data_allowed(holdout, data_end=data_end)
        self._lock = threading.Lock()
        self._seen: set[str] | None = None

    def __call__(self, study: StudyLike, trial: Any) -> None:
        self.record(study.study_name, trial)

    def record(self, study_name: str, trial: Any) -> TrialRecord | None:
        """Register one trial; returns None when it is not registrable or already in."""

        state = _state_name(trial)
        if state not in {"COMPLETE", "PRUNED"}:
            return None
        source = f"optuna:{study_name}#{trial.number}"
        with self._lock:
            if self._seen is None:
                self._seen = _registered_sources(self.registry)
            if source in self._seen:
                return None
            record = self._append(study_name, trial, state, source)
            self._seen.add(source)
            return record

    def _append(self, study_name: str, trial: Any, state: str, source: str) -> TrialRecord:
        values = trial.values if state == "COMPLETE" else None
        metrics = {
            CAPTURE_SOURCE: source,
            "state": state.lower(),
        }
        if values:
            metrics["objective"] = ",".join(f"{value:.10g}" for value in values)
        recorded = trial.datetime_complete
        returns = self.returns(trial)
        return append_trial(
            self.registry,
            recorded_at=(
                recorded.astimezone(UTC)
                if isinstance(recorded, datetime) and recorded.tzinfo is not None
                else _assume_local_as_utc(recorded)
            ),
            code_version=self.code_version,
            strategy_id=self.strategy_id,
            parameters={str(k): _param_text(v) for k, v in sorted(trial.params.items())},
            universe=self.universe,
            data_start=self.data_start,
            data_end=self.data_end,
            cost_assumptions=self.cost_assumptions,
            metrics=metrics,
            operator_note=f"optuna study {study_name} trial {trial.number} ({state.lower()})",
            period_returns=returns,
            # A pruned trial can carry costs for the periods it ran and no
            # full return series; the costs mean nothing without the returns.
            period_costs=None if returns is None else self.costs(trial),
            holdout=self.holdout,
        )


def import_optuna_study(
    study: StudyLike,
    registry: str | Path,
    *,
    strategy_id: str,
    universe: tuple[str, ...],
    data_start: datetime,
    data_end: datetime,
    cost_assumptions: dict[str, str],
    code_version: str = "unrecorded",
    returns: ReturnsGetter | None = None,
    holdout: str | Path | None = None,
    costs: ReturnsGetter | None = None,
) -> ImportReport:
    """Register a study that already ran, e.g. one loaded from its storage.

    Same rules as ``OptunaRecorder``; trials the recorder already registered
    are skipped, so the two can be mixed.
    """

    recorder = OptunaRecorder(
        registry,
        strategy_id=strategy_id,
        universe=universe,
        data_start=data_start,
        data_end=data_end,
        cost_assumptions=cost_assumptions,
        code_version=code_version,
        returns=returns,
        holdout=holdout,
        costs=costs,
    )
    appended: list[TrialRecord] = []
    skipped = 0
    for trial in study.get_trials(deepcopy=False):
        record = recorder.record(study.study_name, trial)
        if record is not None:
            appended.append(record)
        elif _state_name(trial) in {"COMPLETE", "PRUNED"}:
            skipped += 1
    return ImportReport(
        registry=str(recorder.registry),
        appended=tuple(appended),
        already_registered=skipped,
        sources=(f"optuna:{study.study_name}",),
    )


# --- freqtrade ---------------------------------------------------------------


def _fthypt_files(results: Path) -> list[Path]:
    if results.is_file():
        return [results]
    if results.is_dir():
        files = sorted(results.glob("*.fthypt"))
        if files:
            return files
        msg = f"no .fthypt files in {results}; freqtrade writes them to user_data/hyperopt_results"
        raise TrialRegistryError(msg)
    msg = f"{results} does not exist"
    raise TrialRegistryError(msg)


def _read_fthypt(file: Path) -> list[tuple[int, dict[str, Any]]]:
    epochs: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            epoch = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{file}:{line_number} is not a freqtrade hyperopt epoch (invalid JSON)"
            raise TrialRegistryError(msg) from exc
        if not isinstance(epoch, dict) or not isinstance(epoch.get("results_metrics"), dict):
            msg = f"{file}:{line_number} has no results_metrics; is this a .fthypt file?"
            raise TrialRegistryError(msg)
        epochs.append((line_number, epoch))
    return epochs


def _epoch_number(epoch: Mapping[str, Any], line_number: int) -> int:
    raw = epoch.get("current_epoch")
    return int(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else line_number


def _run_fees(epochs: Iterable[tuple[int, Mapping[str, Any]]]) -> dict[str, str]:
    """The per-side fee the run charged, from the first trade that records one.

    The fee is the exchange's (or the config's override) and is the same for
    every epoch of a run, so an epoch that never traded was still priced at it.
    """

    for _, epoch in epochs:
        for trade in epoch["results_metrics"].get("trades") or []:
            if not isinstance(trade, Mapping):
                continue
            fee_open, fee_close = trade.get("fee_open"), trade.get("fee_close")
            if _is_number(fee_open) and _is_number(fee_close):
                return {
                    "fee_open": f"{float(fee_open):.10g}",
                    "fee_close": f"{float(fee_close):.10g}",
                    "charged": "true",
                }
    return {}


def _freqtrade_trial(
    epoch: Mapping[str, Any],
    *,
    source: str,
    where: str,
    fees: dict[str, str],
    code_version: str,
    recorded_at: datetime | None,
) -> dict[str, Any]:
    metrics = epoch["results_metrics"]
    try:
        start = _freqtrade_time(metrics["backtest_start"])
        end = _freqtrade_time(metrics["backtest_end"])
        balance = float(metrics["starting_balance"])
        strategy = str(metrics["strategy_name"])
        pairs = tuple(str(pair) for pair in metrics.get("pairlist") or ())
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"{where}: results_metrics is missing a field trialgate needs ({exc})"
        raise TrialRegistryError(msg) from exc
    daily_fees = _daily_fees(metrics, start, end, where)
    returns, costs = _daily_returns(
        metrics.get("daily_profit") or [], start, end, balance, where, daily_fees
    )
    run_ts = metrics.get("backtest_run_start_ts")
    ran_at = datetime.fromtimestamp(float(run_ts), tz=UTC) if _is_number(run_ts) else None
    reported = {
        CAPTURE_SOURCE: source,
        "loss": _number_text(epoch.get("loss")),
        "total_trades": _number_text(metrics.get("total_trades")),
        "profit_total": _number_text(metrics.get("profit_total")),
        "sharpe": _number_text(metrics.get("sharpe")),
        "max_drawdown_account": _number_text(metrics.get("max_drawdown_account")),
        "timeframe": str(metrics.get("timeframe", "")),
    }
    params = epoch.get("params_dict") or {}
    return {
        "recorded_at": recorded_at or ran_at or datetime.now(tz=UTC),
        "code_version": code_version,
        "strategy_id": strategy,
        "parameters": {str(k): _param_text(v) for k, v in sorted(dict(params).items())},
        "universe": pairs or ("unknown",),
        "data_start": start,
        "data_end": end,
        "cost_assumptions": dict(fees),
        "metrics": {key: value for key, value in reported.items() if value != ""},
        "operator_note": f"freqtrade hyperopt {source.removeprefix('freqtrade:')}",
        "period_returns": returns,
        "period_costs": costs,
    }


def _daily_fees(
    metrics: Mapping[str, Any], start: datetime, end: datetime, where: str
) -> list[float] | None:
    """Fee paid per backtest day, booked on each trade's close day; None if unknown."""

    first = start.date()
    days = (end.date() - first).days + 1
    fees = [0.0] * days
    trades = metrics.get("trades")
    if not isinstance(trades, list):
        traded = metrics.get("total_trades")
        return None if not _is_number(traded) or traded else fees
    for trade in trades:
        if not isinstance(trade, Mapping):
            continue
        paid = _trade_fee(trade)
        if paid is None:
            return None
        try:
            closed = _freqtrade_time(trade["close_date"]).date()
        except (KeyError, ValueError) as exc:
            msg = f"{where}: a trade has no readable close_date ({exc})"
            raise TrialRegistryError(msg) from exc
        offset = (closed - first).days
        if not 0 <= offset < days:
            msg = f"{where}: a trade closed on {closed}, outside the backtest {first}..{end.date()}"
            raise TrialRegistryError(msg)
        fees[offset] += paid
    return fees


def _trade_fee(trade: Mapping[str, Any]) -> float | None:
    """What one trade paid in fees: each fill's amount x price x that side's fee."""

    fee_open, fee_close = trade.get("fee_open"), trade.get("fee_close")
    if not (_is_number(fee_open) and _is_number(fee_close)):
        return None
    orders = trade.get("orders")
    if isinstance(orders, list) and orders:
        paid = 0.0
        for order in orders:
            if not isinstance(order, Mapping):
                return None
            amount, price = order.get("amount"), order.get("safe_price")
            if not (_is_number(amount) and _is_number(price)):
                return None
            side_fee = fee_open if order.get("ft_is_entry") else fee_close
            paid += abs(float(amount)) * float(price) * float(side_fee)
        return paid
    amount, opened, closed = trade.get("amount"), trade.get("open_rate"), trade.get("close_rate")
    if not (_is_number(amount) and _is_number(opened) and _is_number(closed)):
        return None
    size = abs(float(amount))
    return size * float(opened) * float(fee_open) + size * float(closed) * float(fee_close)


def _daily_returns(
    daily_profit: Sequence[Any],
    start: datetime,
    end: datetime,
    balance: float,
    where: str,
    fees: list[float] | None = None,
) -> tuple[list[float], list[float] | None]:
    if balance <= 0.0:
        msg = f"{where}: starting_balance must be positive, got {balance}"
        raise TrialRegistryError(msg)
    first, last = start.date(), end.date()
    days = (last - first).days + 1
    if days < 2:
        msg = f"{where}: the backtest spans {days} day(s); a return series needs at least 2"
        raise TrialRegistryError(msg)
    profit = [0.0] * days
    for row in daily_profit:
        try:
            day = date.fromisoformat(str(row[0]))
            value = float(row[1])
        except (IndexError, TypeError, ValueError) as exc:
            msg = f"{where}: daily_profit row {row!r} is not [date, profit]"
            raise TrialRegistryError(msg) from exc
        offset = (day - first).days
        if not 0 <= offset < days:
            msg = f"{where}: daily_profit has {day}, outside the backtest {first}..{last}"
            raise TrialRegistryError(msg)
        if math.isfinite(value):
            profit[offset] += value
    returns: list[float] = []
    costs: list[float] = []
    equity = balance
    for index, value in enumerate(profit):
        if equity <= 0.0:
            msg = f"{where}: the account is at or below zero by {first + timedelta(days=index)}"
            raise TrialRegistryError(msg)
        returns.append(value / equity)
        if fees is not None:
            costs.append(fees[index] / equity)
        equity += value
    return returns, (costs if fees is not None else None)


def _freqtrade_time(raw: object) -> datetime:
    text = str(raw)
    try:
        value = datetime.strptime(text, _FREQTRADE_TIME)
    except ValueError:
        value = datetime.fromisoformat(text)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# --- shared ------------------------------------------------------------------


def _registered_sources(registry: Path) -> set[str]:
    return {
        trial.metrics[CAPTURE_SOURCE]
        for trial in load_trials(registry)
        if CAPTURE_SOURCE in trial.metrics
    }


def _user_attr_returns(trial: Any) -> Sequence[float] | None:
    raw = getattr(trial, "user_attrs", {}).get("period_returns")
    return None if raw is None else [float(value) for value in raw]


def _user_attr_costs(trial: Any) -> Sequence[float] | None:
    raw = getattr(trial, "user_attrs", {}).get("period_costs")
    return None if raw is None else [float(value) for value in raw]


def _state_name(trial: Any) -> str:
    state = getattr(trial, "state", None)
    return str(getattr(state, "name", state)).upper()


def _assume_local_as_utc(value: object) -> datetime:
    # Optuna stores naive local timestamps. Converting through the local zone is
    # the only reading of a naive time that matches what the clock showed.
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.now(tz=UTC)


def _param_text(value: object) -> str:
    return f"{value:.10g}" if isinstance(value, float) else str(value)


def _number_text(value: object) -> str:
    return f"{float(value):.10g}" if _is_number(value) else ""


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)
