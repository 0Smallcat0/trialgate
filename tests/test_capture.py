"""Trials registered off the optimizer that ran them, not off anyone's memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from trialgate import (
    HoldoutViolationError,
    evaluate_registry,
    import_freqtrade_results,
    import_optuna_study,
    initialize_holdout,
    load_trials,
    performance_matrix,
)
from trialgate.capture import OptunaRecorder
from trialgate.registry import TrialRegistryError

FIXTURES = Path(__file__).parent / "fixtures"


def _epoch(
    number: int,
    daily_profit: list[tuple[str, float]],
    *,
    start: str = "2024-01-01 00:00:00",
    end: str = "2024-01-10 00:00:00",
    trades: list[dict[str, Any]] | None = None,
    is_best: bool = False,
) -> dict[str, Any]:
    """One .fthypt line, with the fields freqtrade's hyperopt writes and this reads."""

    return {
        "loss": -0.1 * number,
        "params_dict": {"buy_rsi": 20 + number, "sell_rsi": 70.5},
        "results_metrics": {
            "strategy_name": "SampleStrategy",
            "pairlist": ["BTC/USDT", "ETH/USDT"],
            "backtest_start": start,
            "backtest_end": end,
            "starting_balance": 1000.0,
            "daily_profit": [list(row) for row in daily_profit],
            "trades": trades if trades is not None else [],
            "total_trades": len(trades or []),
            "profit_total": sum(value for _, value in daily_profit) / 1000.0,
            "sharpe": 1.5,
            "max_drawdown_account": 0.02,
            "timeframe": "5m",
            "backtest_run_start_ts": 1_760_000_000,
        },
        "current_epoch": number,
        "is_best": is_best,
        "fthypt_fileversion": 2,
    }


def _write_fthypt(path: Path, epochs: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(epoch) + "\n" for epoch in epochs), encoding="utf-8")
    return path


_TRADE = {"fee_open": 0.001, "fee_close": 0.001, "profit_abs": 5.0}


def test_every_epoch_becomes_a_trial_with_its_daily_returns(tmp_path: Path) -> None:
    results = tmp_path / "hyperopt_results"
    _write_fthypt(
        results / "strategy_SampleStrategy_2026-10-01_10-00-00.fthypt",
        [
            _epoch(1, [("2024-01-03", 10.0), ("2024-01-04", -20.0)], trades=[_TRADE]),
            _epoch(2, [], trades=[]),
            _epoch(3, [("2024-01-10", 5.0)], trades=[_TRADE], is_best=True),
        ],
    )
    registry = tmp_path / "registry.jsonl"

    report = import_freqtrade_results(results, registry, code_version="abc123")

    trials = load_trials(registry)
    assert len(report.appended) == len(trials) == 3
    first = trials[0]
    assert first.strategy_id == "SampleStrategy"
    assert first.universe == ("BTC/USDT", "ETH/USDT")
    assert first.parameters == {"buy_rsi": "21", "sell_rsi": "70.5"}
    assert first.data_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert first.code_version == "abc123"
    assert first.returns_count == 10  # every day of 2024-01-01..2024-01-10
    assert first.metrics["capture_source"].endswith(".fthypt#1")

    matrix = performance_matrix(registry)
    column = [row[0] for row in matrix]
    # 10 on a 1000 balance, then -20 on the 1010 that was there at the start of the day.
    assert column[2] == pytest.approx(0.01)
    assert column[3] == pytest.approx(-20.0 / 1010.0)
    assert sum(1 for value in column if value != 0.0) == 2
    # An epoch that never traded is still a trial, priced at the run's fee.
    assert [row[1] for row in matrix] == [0.0] * 10
    assert trials[1].cost_assumptions == {
        "fee_open": "0.001",
        "fee_close": "0.001",
        "charged": "true",
    }


def test_importing_again_adds_only_new_epochs(tmp_path: Path) -> None:
    results = tmp_path / "hyperopt_results"
    _write_fthypt(results / "strategy_S_2026-10-01.fthypt", [_epoch(1, []), _epoch(2, [])])
    registry = tmp_path / "registry.jsonl"
    import_freqtrade_results(results, registry)

    _write_fthypt(results / "strategy_S_2026-10-02.fthypt", [_epoch(1, [])])
    second = import_freqtrade_results(results, registry)

    assert len(second.appended) == 1
    assert second.already_registered == 2
    assert len(load_trials(registry)) == 3
    assert "0 new" in import_freqtrade_results(results, registry).summary()


def test_a_hyperopt_into_the_locked_holdout_is_refused_whole(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.json"
    initialize_holdout(
        holdout,
        holdout_start=datetime(2024, 1, 5, tzinfo=UTC),
        locked_at=datetime(2023, 12, 1, tzinfo=UTC),
    )
    results = _write_fthypt(tmp_path / "run.fthypt", [_epoch(1, []), _epoch(2, [])])
    registry = tmp_path / "registry.jsonl"

    with pytest.raises(HoldoutViolationError):
        import_freqtrade_results(results, registry, holdout=holdout)
    assert not registry.exists()


def test_a_file_that_is_not_a_hyperopt_log_is_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.fthypt"
    bogus.write_text('{"loss": 1}\n', encoding="utf-8")
    with pytest.raises(TrialRegistryError, match="results_metrics"):
        import_freqtrade_results(bogus, tmp_path / "registry.jsonl")
    (tmp_path / "empty").mkdir()
    with pytest.raises(TrialRegistryError, match="no .fthypt"):
        import_freqtrade_results(tmp_path / "empty", tmp_path / "r.jsonl")
    with pytest.raises(TrialRegistryError, match="does not exist"):
        import_freqtrade_results(tmp_path / "missing", tmp_path / "r.jsonl")


def test_a_profit_outside_the_backtest_range_is_an_error(tmp_path: Path) -> None:
    results = _write_fthypt(tmp_path / "run.fthypt", [_epoch(1, [("2023-12-31", 1.0)])])
    with pytest.raises(TrialRegistryError, match="outside the backtest"):
        import_freqtrade_results(results, tmp_path / "registry.jsonl")


def test_a_real_freqtrade_hyperopt_log_goes_through_every_gate(tmp_path: Path) -> None:
    """A .fthypt written by freqtrade itself, not by this test file.

    freqtrade 2026.9, its bundled SampleStrategy, ``hyperopt --spaces buy sell
    --epochs 8 --timeframe 4h --timerange 20240101-20240701 -p BTC/USDT
    --random-state 42`` on Bybit candles.
    """

    registry = tmp_path / "registry.jsonl"
    report = import_freqtrade_results(FIXTURES / "freqtrade", registry)
    trials = load_trials(registry)
    assert len(report.appended) == len(trials) == 8
    assert {trial.strategy_id for trial in trials} == {"SampleStrategy"}
    assert {trial.returns_count for trial in trials} == {183}  # 2024-01-01..2024-07-01
    assert trials[0].cost_assumptions["fee_open"] == "0.001"

    # The epoch freqtrade itself would export: the lowest loss.
    best = min(trials, key=lambda trial: float(trial.metrics["loss"]))
    gates = evaluate_registry(
        registry,
        candidate_trial_id=best.trial_id,
        periods_per_year=365,
        effective_trials=len(trials),
        block_count=4,
        min_observations=10,
    )
    assert [check.gate for check in gates.checks][-1] == "declared costs (gate 8)"
    assert gates.checks[-1].passed

    # Each epoch's booked profit, as freqtrade summed it, is what the series compounds to.
    epochs = [
        json.loads(line)
        for line in next((FIXTURES / "freqtrade").glob("*.fthypt"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for trial, column in zip(trials, zip(*performance_matrix(registry), strict=True), strict=True):
        final = 1.0
        for value in column:
            final *= 1.0 + value
        metrics = epochs[trial.trial_id - 1]["results_metrics"]
        assert final - 1.0 == pytest.approx(metrics["profit_total"], abs=1e-9)


# --- Optuna -----------------------------------------------------------------


class _State(Enum):
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3


@dataclass
class _Trial:
    """Shaped like optuna.trial.FrozenTrial, for the attributes trialgate reads."""

    number: int
    params: dict[str, Any]
    state: _State
    values: list[float] | None
    user_attrs: dict[str, Any] = field(default_factory=dict)
    datetime_complete: datetime | None = None


@dataclass
class _Study:
    study_name: str
    trials: list[_Trial]

    def get_trials(self, deepcopy: bool = True) -> list[_Trial]:
        return list(self.trials)


def _recorder(registry: Path, **kwargs: Any) -> OptunaRecorder:
    return OptunaRecorder(
        registry,
        strategy_id="rsi",
        universe=("SPY",),
        data_start=datetime(2015, 1, 1, tzinfo=UTC),
        data_end=datetime(2024, 12, 31, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "10"},
        **kwargs,
    )


def test_the_callback_registers_complete_and_pruned_trials_but_not_failed(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.jsonl"
    recorder = _recorder(registry)
    study = _Study("s1", [])
    finished = datetime(2026, 10, 21, 9, 30, tzinfo=UTC)
    for trial in (
        _Trial(0, {"window": 10}, _State.COMPLETE, [1.2], {"period_returns": [0.01, -0.02]}),
        _Trial(1, {"window": 20}, _State.PRUNED, None),
        _Trial(2, {"window": 30}, _State.FAIL, None),
        _Trial(3, {"window": 0.25}, _State.COMPLETE, [0.4], {}, finished),
    ):
        recorder(study, trial)

    trials = load_trials(registry)
    assert [t.metrics["capture_source"] for t in trials] == [
        "optuna:s1#0",
        "optuna:s1#1",
        "optuna:s1#3",
    ]
    assert trials[0].returns_count == 2
    assert trials[0].metrics["objective"] == "1.2"
    assert trials[1].metrics["state"] == "pruned"
    assert trials[2].parameters == {"window": "0.25"}
    assert trials[2].recorded_at == finished


def test_backfilling_a_study_skips_what_the_callback_already_registered(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    study = _Study("s2", [_Trial(n, {"x": n}, _State.COMPLETE, [float(n)]) for n in range(4)])
    recorder = _recorder(registry)
    recorder(study, study.trials[0])
    recorder(study, study.trials[1])

    report = import_optuna_study(
        study,
        registry,
        strategy_id="rsi",
        universe=("SPY",),
        data_start=datetime(2015, 1, 1, tzinfo=UTC),
        data_end=datetime(2024, 12, 31, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "10"},
    )

    assert len(report.appended) == 2
    assert report.already_registered == 2
    assert len(load_trials(registry)) == 4


def test_a_study_over_the_locked_holdout_fails_before_the_first_trial(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.json"
    initialize_holdout(
        holdout,
        holdout_start=datetime(2024, 1, 1, tzinfo=UTC),
        locked_at=datetime(2023, 12, 1, tzinfo=UTC),
    )
    with pytest.raises(HoldoutViolationError):
        _recorder(tmp_path / "registry.jsonl", holdout=holdout)


def test_the_callback_works_inside_a_real_optuna_study(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    registry = tmp_path / "registry.jsonl"

    def objective(trial: Any) -> float:
        window = trial.suggest_int("window", 2, 50)
        returns = [((day * window) % 7 - 3) / 1000.0 for day in range(40)]
        trial.set_user_attr("period_returns", returns)
        return sum(returns)

    study = optuna.create_study(study_name="real", direction="maximize")
    study.optimize(objective, n_trials=12, callbacks=[_recorder(registry)])

    trials = load_trials(registry)
    assert len(trials) == 12
    assert all(trial.returns_count == 40 for trial in trials)
    assert len(performance_matrix(registry)[0]) == 12
