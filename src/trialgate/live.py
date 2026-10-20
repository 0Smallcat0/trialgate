"""Gate 6: write the stop condition down before going live, then hold to it.

Money is lost after the backtest, not in it: the strategy goes live, runs
below what the backtest promised, and keeps running because nobody decided in
advance how far below was too far. By the time the question is asked, the
answer is being chosen by someone who wants to keep going.

``lock_live_bands()`` answers it beforehand, from the candidate's own
registered return series. It block-bootstraps that series into paths as long
as the live trial will run and, for every period of it, bounds three things a
live account shows: cumulative return (from below), drawdown from the running
peak, and the longest run of losing periods (from above). The bounds are
calibrated jointly, across all three statistics and every period, so a
strategy that really is what its backtest says crosses one of them over the
whole horizon with the stated probability (default 5%) -- not 5% per period,
which over 90 periods would stop nearly everything. The bands go into a lock
file that refuses to be rewritten.

``check_live()`` puts the live (or paper) returns against them: STOP at the
first crossing, PENDING while inside them with periods still to run, PASS
once the horizon is complete inside them.

What this catches is a live series that does not look like a draw from the
backtest: look-ahead and other bugs, costs and fills worse than modeled, and
an edge that was luck. What it cannot catch is the backtest itself being
wrong in a way the live series shares. A pass is a necessary condition, like
every other gate.

Bootstrap: circular blocks (Politis & Romano 1992) of length ``ceil(T^(1/3))``
by default. Hall, Horowitz & Jing (1995) give T^(1/3) as the optimal rate for
variance estimation and T^(1/4) for a one-sided distribution function; the
bands are quantiles, but the shorter blocks would keep less of the serial
correlation, so gate 6 takes the longer of the two. Blocks keep the serial
correlation an IID resample would destroy, which would draw the bands too
narrow. The false-alarm rate is
calibrated on one set of resamples and measured on a second, independent set;
the lock file records both.
"""

from __future__ import annotations

import json
import math
import random
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from trialgate.registry import TrialRecord, TrialRegistryError, _load_returns, load_trials

DEFAULT_FALSE_ALARM_RATE = 0.05
DEFAULT_RESAMPLES = 2000

LiveStatus = Literal["STOP", "PENDING", "PASS"]


class LiveBandsError(ValueError):
    """Raised when bands cannot be locked, loaded, or trusted."""


@dataclass(frozen=True, slots=True)
class BandSet:
    """The bands for one return series: what ``compute_live_bands()`` returns.

    Element ``d`` of each band applies after ``d + 1`` live periods.
    """

    backtest_periods: int
    horizon: int
    false_alarm_rate: float
    measured_false_alarm_rate: float
    power_if_no_edge: float
    block_length: int
    resamples: int
    seed: int
    min_cumulative_return: tuple[float, ...]
    max_drawdown: tuple[float, ...]
    max_losing_streak: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LiveBands:
    """The stop condition, fixed before the first live period.

    Element ``d`` of each band applies after ``d + 1`` live periods.
    """

    strategy_id: str
    candidate_trial_id: int
    returns_sha256: str
    backtest_periods: int
    horizon: int
    false_alarm_rate: float
    measured_false_alarm_rate: float
    power_if_no_edge: float
    block_length: int
    resamples: int
    seed: int
    locked_at: datetime
    min_cumulative_return: tuple[float, ...]
    max_drawdown: tuple[float, ...]
    max_losing_streak: tuple[int, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "candidate_trial_id": self.candidate_trial_id,
            "returns_sha256": self.returns_sha256,
            "backtest_periods": self.backtest_periods,
            "horizon": self.horizon,
            "false_alarm_rate": self.false_alarm_rate,
            "measured_false_alarm_rate": self.measured_false_alarm_rate,
            "power_if_no_edge": self.power_if_no_edge,
            "block_length": self.block_length,
            "resamples": self.resamples,
            "seed": self.seed,
            "locked_at": self.locked_at.isoformat(),
            "min_cumulative_return": list(self.min_cumulative_return),
            "max_drawdown": list(self.max_drawdown),
            "max_losing_streak": list(self.max_losing_streak),
        }


@dataclass(frozen=True, slots=True)
class LiveCheck:
    """Gate 6's verdict on the live returns so far."""

    status: LiveStatus
    periods_checked: int
    horizon: int
    breach: str | None
    breach_period: int | None
    observed: float | None
    bound: float | None
    cumulative_return: float
    drawdown: float
    losing_streak: int
    false_alarm_rate: float
    power_if_no_edge: float

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def summary(self) -> str:
        gate = "live deviation (gate 6)"
        state = (
            f"cumulative {self.cumulative_return:+.2%}, drawdown {self.drawdown:.2%}, "
            f"longest losing run {self.losing_streak}"
        )
        if self.status == "STOP":
            assert self.observed is not None and self.bound is not None
            if self.breach == "losing streak":
                crossed = f"{self.observed:.0f} losing periods in a row > {self.bound:.0f}"
            elif self.breach == "drawdown":
                crossed = f"drawdown {self.observed:.2%} > {self.bound:.2%}"
            else:
                crossed = f"cumulative return {self.observed:+.2%} < {self.bound:+.2%}"
            return (
                f"STOP  {gate}: period {self.breach_period} of {self.horizon}, {crossed}\n"
                f"      a strategy that is what its backtest says crosses a band "
                f"{self.false_alarm_rate:.0%} of the time over the horizon; this is the "
                "condition written down before going live. Stop trading it."
            )
        if self.status == "PASS":
            return (
                f"PASS  {gate}: {self.horizon} of {self.horizon} periods inside the "
                f"locked bands ({state})\n"
                f"      the same returns with their mean removed -- no edge at all -- also "
                f"stay inside {1.0 - self.power_if_no_edge:.0%} of the time over this horizon"
            )
        return (
            f"PENDING  {gate}: {self.periods_checked} of {self.horizon} periods inside "
            f"the locked bands so far ({state})"
        )


def lock_live_bands(
    path: str | Path,
    *,
    registry: str | Path,
    candidate_trial_id: int,
    horizon: int,
    locked_at: datetime,
    false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    block_length: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> LiveBands:
    """Compute gate 6's bands from the registered candidate and lock them.

    ``horizon`` -- how many live periods to watch before trusting the
    strategy -- has no default: it is a policy, like the stop line, and the
    old procedural form of this gate asked for at least three months.
    Choose it by ``power_if_no_edge`` on the result: how often the same
    returns with their mean removed would be stopped within the horizon. A
    Sharpe-1 daily strategy whose edge vanishes is caught only about one time
    in ten over 90 days; that is what a quarter of live data can tell you, and
    a PASS says so rather than implying more.

    Calling this again with the same arguments returns the existing lock.
    Calling it with different ones raises: choosing the bands after seeing
    some live results is the thing the lock exists to prevent.
    """

    if locked_at.tzinfo is None or locked_at.utcoffset() != UTC.utcoffset(locked_at):
        msg = "locked_at must be timezone-aware UTC"
        raise LiveBandsError(msg)
    returns, trial = _candidate_returns(registry, candidate_trial_id)
    bands = compute_live_bands(
        returns,
        horizon=horizon,
        false_alarm_rate=false_alarm_rate,
        block_length=block_length,
        resamples=resamples,
        seed=seed,
    )
    locked = LiveBands(
        strategy_id=trial.strategy_id,
        candidate_trial_id=trial.trial_id,
        returns_sha256=str(trial.returns_sha256),
        backtest_periods=bands.backtest_periods,
        horizon=bands.horizon,
        false_alarm_rate=bands.false_alarm_rate,
        measured_false_alarm_rate=bands.measured_false_alarm_rate,
        power_if_no_edge=bands.power_if_no_edge,
        block_length=bands.block_length,
        resamples=bands.resamples,
        seed=bands.seed,
        locked_at=locked_at,
        min_cumulative_return=bands.min_cumulative_return,
        max_drawdown=bands.max_drawdown,
        max_losing_streak=bands.max_losing_streak,
    )

    lock_path = Path(path)
    existing = load_live_bands(lock_path) if lock_path.exists() else None
    if existing is not None:
        if _same_bands(existing, locked):
            return existing
        msg = (
            f"{lock_path} already locks gate 6 for trial {existing.candidate_trial_id} "
            f"(horizon {existing.horizon}, false alarm {existing.false_alarm_rate:g}), "
            f"locked {existing.locked_at.isoformat()}; bands are set once, before the "
            "live results, and are not re-chosen"
        )
        raise LiveBandsError(msg)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(locked.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return locked


def load_live_bands(path: str | Path) -> LiveBands:
    lock_path = Path(path)
    try:
        row = json.loads(lock_path.read_text(encoding="utf-8"))
        bands = LiveBands(
            strategy_id=str(row["strategy_id"]),
            candidate_trial_id=int(row["candidate_trial_id"]),
            returns_sha256=str(row["returns_sha256"]),
            backtest_periods=int(row["backtest_periods"]),
            horizon=int(row["horizon"]),
            false_alarm_rate=float(row["false_alarm_rate"]),
            measured_false_alarm_rate=float(row["measured_false_alarm_rate"]),
            power_if_no_edge=float(row["power_if_no_edge"]),
            block_length=int(row["block_length"]),
            resamples=int(row["resamples"]),
            seed=int(row["seed"]),
            locked_at=datetime.fromisoformat(str(row["locked_at"])).astimezone(UTC),
            min_cumulative_return=tuple(float(v) for v in row["min_cumulative_return"]),
            max_drawdown=tuple(float(v) for v in row["max_drawdown"]),
            max_losing_streak=tuple(int(v) for v in row["max_losing_streak"]),
        )
    except FileNotFoundError as exc:
        msg = f"no gate 6 lock at {lock_path}; lock_live_bands() writes it before going live"
        raise LiveBandsError(msg) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        msg = f"{lock_path} is not a gate 6 lock file ({exc})"
        raise LiveBandsError(msg) from exc
    lengths = {
        len(bands.min_cumulative_return),
        len(bands.max_drawdown),
        len(bands.max_losing_streak),
    }
    if lengths != {bands.horizon}:
        msg = f"{lock_path} has bands of length {sorted(lengths)} for a horizon of {bands.horizon}"
        raise LiveBandsError(msg)
    return bands


def check_live(
    path: str | Path,
    live_returns: Sequence[float],
    *,
    registry: str | Path | None = None,
) -> LiveCheck:
    """Judge the live returns so far against the locked bands.

    ``live_returns`` are per period, at the backtest's frequency, net of the
    costs actually paid, starting from the first live period. Periods beyond
    the horizon are not judged.

    Pass ``registry`` to recompute the bands from the registered candidate
    and confirm the lock file still holds what was locked; the computation is
    deterministic in its recorded seed, so an edited band does not survive it.
    """

    bands = load_live_bands(path)
    if registry is not None:
        returns, trial = _candidate_returns(registry, bands.candidate_trial_id)
        if trial.returns_sha256 != bands.returns_sha256:
            msg = (
                f"trial {bands.candidate_trial_id}'s returns in {registry} are not the ones "
                f"{path} was locked from"
            )
            raise LiveBandsError(msg)
        recomputed = compute_live_bands(
            returns,
            horizon=bands.horizon,
            false_alarm_rate=bands.false_alarm_rate,
            block_length=bands.block_length,
            resamples=bands.resamples,
            seed=bands.seed,
        )
        if not _same_numbers(recomputed, bands):
            msg = (
                f"{path} does not hold the bands its own seed and registered returns "
                "produce; it has been edited since it was locked"
            )
            raise LiveBandsError(msg)

    values = [float(value) for value in live_returns]
    if any(not math.isfinite(value) or value <= -1.0 for value in values):
        msg = "live returns must be finite and above -100%"
        raise LiveBandsError(msg)
    judged = values[: bands.horizon]
    equity = peak = 1.0
    drawdown = 0.0
    streak = longest = 0
    for index, value in enumerate(judged):
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
        streak = streak + 1 if value < 0.0 else 0
        longest = max(longest, streak)
        cumulative = equity - 1.0
        crossing: tuple[str, float, float] | None = None
        if cumulative < bands.min_cumulative_return[index]:
            crossing = ("cumulative return", cumulative, bands.min_cumulative_return[index])
        elif drawdown > bands.max_drawdown[index]:
            crossing = ("drawdown", drawdown, bands.max_drawdown[index])
        elif longest > bands.max_losing_streak[index]:
            crossing = ("losing streak", float(longest), float(bands.max_losing_streak[index]))
        if crossing is not None:
            return LiveCheck(
                status="STOP",
                periods_checked=index + 1,
                horizon=bands.horizon,
                breach=crossing[0],
                breach_period=index + 1,
                observed=crossing[1],
                bound=crossing[2],
                cumulative_return=cumulative,
                drawdown=drawdown,
                losing_streak=longest,
                false_alarm_rate=bands.false_alarm_rate,
                power_if_no_edge=bands.power_if_no_edge,
            )
    return LiveCheck(
        status="PASS" if len(judged) >= bands.horizon else "PENDING",
        periods_checked=len(judged),
        horizon=bands.horizon,
        breach=None,
        breach_period=None,
        observed=None,
        bound=None,
        cumulative_return=equity - 1.0,
        drawdown=drawdown,
        losing_streak=longest,
        false_alarm_rate=bands.false_alarm_rate,
        power_if_no_edge=bands.power_if_no_edge,
    )


def compute_live_bands(
    returns: Sequence[float],
    *,
    horizon: int,
    false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    block_length: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BandSet:
    """The bands for one return series, without touching a registry or a file.

    ``lock_live_bands()`` adds the provenance and writes them.
    """

    series = [float(value) for value in returns]
    periods = len(series)
    if horizon < 1:
        msg = "horizon must be at least 1 period"
        raise LiveBandsError(msg)
    if not 0.0 < false_alarm_rate < 0.5:
        msg = "false_alarm_rate must be in (0, 0.5)"
        raise LiveBandsError(msg)
    length = block_length if block_length is not None else math.ceil(periods ** (1.0 / 3.0))
    if length < 1:
        msg = "block_length must be at least 1"
        raise LiveBandsError(msg)
    if periods < 2 * length:
        msg = f"{periods} backtest periods is too few to resample in blocks of {length}"
        raise LiveBandsError(msg)
    if resamples < 200:
        msg = "resamples must be at least 200 to locate a tail band"
        raise LiveBandsError(msg)
    if any(not math.isfinite(value) or value <= -1.0 for value in series):
        msg = "backtest returns must be finite and above -100%"
        raise LiveBandsError(msg)

    rng = random.Random(seed)
    calibration = _paths(series, horizon, length, resamples, rng)
    holdout = _paths(series, horizon, length, resamples, rng)
    mean = math.fsum(series) / periods
    no_edge = _paths([value - mean for value in series], horizon, length, resamples, rng)

    sorted_columns = (
        [sorted(column) for column in calibration[0]],
        [sorted(column) for column in calibration[1]],
        [sorted(column) for column in calibration[2]],
    )
    # Largest order statistic k whose bands the calibration paths cross at
    # most false_alarm_rate of the time. Crossing rates only rise with k.
    low, high = 0, int(false_alarm_rate * resamples)
    while low < high:
        middle = (low + high + 1) // 2
        if _crossing_rate(calibration, _bands_at(sorted_columns, middle)) <= false_alarm_rate:
            low = middle
        else:
            high = middle - 1
    bands = _bands_at(sorted_columns, low)
    cumulative, drawdown, streak = bands
    return BandSet(
        backtest_periods=periods,
        horizon=horizon,
        false_alarm_rate=false_alarm_rate,
        measured_false_alarm_rate=_crossing_rate(holdout, bands),
        power_if_no_edge=_crossing_rate(no_edge, bands),
        block_length=length,
        resamples=resamples,
        seed=seed,
        min_cumulative_return=tuple(cumulative),
        max_drawdown=tuple(drawdown),
        max_losing_streak=tuple(int(value) for value in streak),
    )


Columns = tuple[list[array[float]], list[array[float]], list[array[float]]]
Bands = tuple[list[float], list[float], list[float]]


def _paths(
    series: list[float], horizon: int, length: int, resamples: int, rng: random.Random
) -> Columns:
    """Per-period columns of the three running statistics over ``resamples`` paths."""

    periods = len(series)
    cumulative = [array("d", bytes(8 * resamples)) for _ in range(horizon)]
    drawdown = [array("d", bytes(8 * resamples)) for _ in range(horizon)]
    streak = [array("d", bytes(8 * resamples)) for _ in range(horizon)]
    for path in range(resamples):
        equity = peak = 1.0
        worst = 0.0
        run = longest = 0
        start = 0
        for step in range(horizon):
            if step % length == 0:
                start = int(rng.random() * periods)
            value = series[(start + step % length) % periods]
            equity *= 1.0 + value
            peak = max(peak, equity)
            worst = max(worst, 1.0 - equity / peak)
            run = run + 1 if value < 0.0 else 0
            longest = max(longest, run)
            cumulative[step][path] = equity - 1.0
            drawdown[step][path] = worst
            streak[step][path] = float(longest)
    return cumulative, drawdown, streak


def _bands_at(
    sorted_columns: tuple[list[list[float]], list[list[float]], list[list[float]]], k: int
) -> Bands:
    cumulative, drawdown, streak = sorted_columns
    return (
        [column[k] for column in cumulative],
        [column[-1 - k] for column in drawdown],
        [column[-1 - k] for column in streak],
    )


def _crossing_rate(columns: Columns, bands: Bands) -> float:
    cumulative, drawdown, streak = columns
    low, high_drawdown, high_streak = bands
    resamples = len(cumulative[0])
    crossed = bytearray(resamples)
    for step in range(len(cumulative)):
        c, d, s = cumulative[step], drawdown[step], streak[step]
        floor, ceiling, longest = low[step], high_drawdown[step], high_streak[step]
        for path in range(resamples):
            if not crossed[path] and (c[path] < floor or d[path] > ceiling or s[path] > longest):
                crossed[path] = 1
    return sum(crossed) / resamples


def _candidate_returns(registry: str | Path, trial_id: int) -> tuple[list[float], TrialRecord]:
    registry_path = Path(registry)
    for trial in load_trials(registry_path):
        if trial.trial_id == trial_id:
            if trial.returns_sha256 is None:
                msg = (
                    f"trial {trial_id} was registered without period_returns; gate 6 "
                    "draws its bands from the registered series, not from a series handed in"
                )
                raise TrialRegistryError(msg)
            return _load_returns(registry_path, trial), trial
    msg = f"trial {trial_id} is not registered in {registry_path}"
    raise TrialRegistryError(msg)


def _same_numbers(computed: BandSet, bands: LiveBands) -> bool:
    return (
        computed.backtest_periods == bands.backtest_periods
        and computed.max_losing_streak == bands.max_losing_streak
        and _close(computed.min_cumulative_return, bands.min_cumulative_return)
        and _close(computed.max_drawdown, bands.max_drawdown)
    )


def _close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(left, right, strict=True)
    )


def _same_bands(existing: LiveBands, new: LiveBands) -> bool:
    return (
        existing.candidate_trial_id == new.candidate_trial_id
        and existing.returns_sha256 == new.returns_sha256
        and existing.horizon == new.horizon
        and existing.false_alarm_rate == new.false_alarm_rate
        and existing.block_length == new.block_length
        and existing.resamples == new.resamples
        and existing.seed == new.seed
    )
