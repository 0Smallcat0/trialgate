"""Gate 6: the stop condition is written down before the first live period."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trialgate import (
    LiveBandsError,
    append_trial,
    check_live,
    compute_live_bands,
    load_live_bands,
    lock_live_bands,
)

_NOW = datetime(2026, 10, 21, tzinfo=UTC)
_DAILY_SD = 0.012
_SHARPE_ONE = 1.0 / 252**0.5 * _DAILY_SD


def _series(count: int, mean: float, seed: int, sd: float = _DAILY_SD) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(count)]


def _registry_with(path: Path, returns: list[float]) -> int:
    record = append_trial(
        path,
        recorded_at=_NOW,
        code_version="test",
        strategy_id="trend",
        parameters={"lookback": "20"},
        universe=("SPY",),
        data_start=datetime(2018, 1, 1, tzinfo=UTC),
        data_end=datetime(2025, 12, 31, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "10"},
        metrics={},
        operator_note="candidate",
        period_returns=returns,
    )
    return record.trial_id


def _crosses(bands_path: Path, live: list[float]) -> bool:
    return check_live(bands_path, live).status == "STOP"


@pytest.fixture
def locked(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "registry.jsonl"
    trial_id = _registry_with(registry, _series(1500, _SHARPE_ONE, seed=1))
    bands = tmp_path / "live_bands.json"
    lock_live_bands(
        bands, registry=registry, candidate_trial_id=trial_id, horizon=60, locked_at=_NOW
    )
    return registry, bands


def test_the_false_alarm_rate_is_joint_over_the_horizon_not_per_period() -> None:
    backtest = _series(1500, _SHARPE_ONE, seed=2)
    bands = compute_live_bands(backtest, horizon=60, resamples=1000, seed=3)

    # Fresh draws from the process the backtest came from: the strategy is
    # exactly what it claimed. About 5% of them may cross; a per-period 5%
    # band would stop most of them over 60 periods.
    crossed = 0
    for seed in range(400):
        live = _series(60, _SHARPE_ONE, seed=1000 + seed)
        crossed += _first_crossing(bands, live) is not None
    assert 0.01 <= crossed / 400 <= 0.11
    assert 0.0 < bands.measured_false_alarm_rate <= 0.08
    # Running maxima only grow, so their bounds do too.
    assert list(bands.max_drawdown) == sorted(bands.max_drawdown)
    assert list(bands.max_losing_streak) == sorted(bands.max_losing_streak)


def _first_crossing(bands: object, live: list[float]) -> int | None:
    from trialgate.live import BandSet

    assert isinstance(bands, BandSet)
    equity = peak = 1.0
    drawdown = 0.0
    streak = longest = 0
    for index, value in enumerate(live):
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
        streak = streak + 1 if value < 0.0 else 0
        longest = max(longest, streak)
        if (
            equity - 1.0 < bands.min_cumulative_return[index]
            or drawdown > bands.max_drawdown[index]
            or longest > bands.max_losing_streak[index]
        ):
            return index
    return None


def test_a_crash_the_backtest_never_showed_is_a_stop(locked: tuple[Path, Path]) -> None:
    _, bands = locked
    live = [0.002] * 10 + [-0.05] * 5

    verdict = check_live(bands, live)

    assert verdict.status == "STOP"
    assert verdict.breach_period is not None and verdict.breach_period <= 15
    assert verdict.summary().startswith("STOP  live deviation (gate 6): period")
    assert "Stop trading it." in verdict.summary()


def test_inside_the_bands_is_pending_until_the_horizon_and_then_a_pass(
    locked: tuple[Path, Path],
) -> None:
    _, bands = locked
    calm = [0.0005, -0.0004] * 40

    assert check_live(bands, calm[:30]).status == "PENDING"
    final = check_live(bands, calm)
    assert final.status == "PASS"
    assert final.periods_checked == 60
    # The pass states how little a quarter of live data can prove.
    assert "no edge at all -- also stay inside" in final.summary()


def test_the_lock_is_set_once(locked: tuple[Path, Path]) -> None:
    registry, bands = locked
    again = lock_live_bands(
        bands, registry=registry, candidate_trial_id=1, horizon=60, locked_at=_NOW
    )
    assert again == load_live_bands(bands)
    with pytest.raises(LiveBandsError, match="not re-chosen"):
        lock_live_bands(
            bands,
            registry=registry,
            candidate_trial_id=1,
            horizon=60,
            locked_at=_NOW,
            false_alarm_rate=0.01,
        )


def test_an_edited_band_does_not_survive_recomputation(locked: tuple[Path, Path]) -> None:
    registry, bands = locked
    row = json.loads(bands.read_text(encoding="utf-8"))
    row["max_drawdown"] = [value * 2 for value in row["max_drawdown"]]
    bands.write_text(json.dumps(row), encoding="utf-8")

    assert check_live(bands, [0.0] * 5).status == "PENDING"  # the file alone cannot tell
    with pytest.raises(LiveBandsError, match="edited since it was locked"):
        check_live(bands, [0.0] * 5, registry=registry)


def test_bands_come_from_the_registered_series_only(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    append_trial(
        registry,
        recorded_at=_NOW,
        code_version="test",
        strategy_id="trend",
        parameters={},
        universe=("SPY",),
        data_start=datetime(2018, 1, 1, tzinfo=UTC),
        data_end=datetime(2025, 12, 31, tzinfo=UTC),
        cost_assumptions={"round_trip_bps": "10"},
        metrics={"annualized_sharpe": "2.0"},
        operator_note="no returns stored",
    )
    with pytest.raises(ValueError, match="without period_returns"):
        lock_live_bands(
            tmp_path / "bands.json",
            registry=registry,
            candidate_trial_id=1,
            horizon=30,
            locked_at=_NOW,
        )


def test_a_too_short_backtest_or_a_naive_timestamp_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LiveBandsError, match="too few"):
        compute_live_bands([0.01, -0.01, 0.0], horizon=10)
    registry = tmp_path / "registry.jsonl"
    _registry_with(registry, _series(300, 0.0, seed=4))
    with pytest.raises(LiveBandsError, match="UTC"):
        lock_live_bands(
            tmp_path / "b.json",
            registry=registry,
            candidate_trial_id=1,
            horizon=10,
            locked_at=datetime(2026, 1, 1),
        )
