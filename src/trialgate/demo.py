"""A runnable proof that a backtest alone proves nothing.

``python -m trialgate demo`` optimizes a four-parameter strategy over a PURE
RANDOM WALK — data with no edge in it, by construction — reports the Sharpe
ratio that the best configuration achieves in-sample, and then puts that
configuration through the gates.

It doubles as the integration example: everything below is what you would do
with your own returns, and the only lines you would change are the ones that
generate ``prices``. Nothing here is special-cased for the demo.
"""

from __future__ import annotations

import random
import statistics
import tempfile
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from trialgate.gates import evaluate_registry
from trialgate.registry import append_trial, verify_chain
from trialgate.validation import sharpe_ratio

PERIODS = 2000
PERIODS_PER_YEAR = 252
# Four knobs at the resolution someone actually sweeps them. The size of this
# grid is the point: the best of 450 tries on noise is not the same object as
# one try, and that difference is what the gates are measuring.
ENTRY_LOOKBACKS = (5, 10, 15, 20, 30, 40, 65, 100, 150, 200)
EXIT_LOOKBACKS = (3, 5, 10, 20, 40)
VOL_WINDOWS = (10, 20, 60)
VOL_THRESHOLDS = (0.7, 1.0, 1.4)
# Charged on every position change, and declared to the registry as the same
# number. Gate 8 rejects a registry whose trials priced execution at zero, so
# a demo that skipped this would be demonstrating the wrong thing.
ROUND_TRIP_BPS = 10.0
# The demo's own hard stop, for gate 7. A real one is a policy decision; this
# is here so the gate has something to measure against.
STOP_LOSS_FRACTION = 0.15


def _random_walk(seed: int) -> list[float]:
    """Geometric random walk with zero drift. There is no signal in here."""

    rng = random.Random(seed)
    prices = [100.0]
    for _ in range(PERIODS):
        prices.append(prices[-1] * (1.0 + rng.gauss(0.0, 0.012)))
    return prices


def _rolling_volatility(raw: list[float], window: int) -> list[float]:
    return [
        statistics.pstdev(raw[max(0, day - window) : day]) if day > 1 else 0.0
        for day in range(PERIODS)
    ]


def _strategy_returns(
    prices: list[float],
    raw: list[float],
    volatility: list[float],
    baseline: float,
    *,
    entry_lookback: int,
    exit_lookback: int,
    vol_window: int,
    vol_threshold: float,
) -> tuple[list[float], list[float]]:
    """Trend-follow with a volatility filter: four knobs, no edge to find.

    Returns are net of execution: each side of a round trip is charged when the
    position changes, so what reaches the registry is what an account would
    have kept. The second list is that charge, period by period, for gate 9.
    """

    warmup = max(entry_lookback, exit_lookback, vol_window)
    one_way_cost = ROUND_TRIP_BPS / 2.0 / 10_000.0
    returns = [0.0] * PERIODS
    costs = [0.0] * PERIODS
    holding = False
    for day in range(warmup, PERIODS):
        previous = holding
        if volatility[day] > vol_threshold * baseline:
            holding = False
        elif holding:
            holding = prices[day] >= prices[day - exit_lookback]
        else:
            holding = prices[day] > prices[day - entry_lookback]
        if holding:
            returns[day] = raw[day]
        if holding != previous:
            returns[day] -= one_way_cost
            costs[day] = one_way_cost
    return returns, costs


def run_demo() -> int:
    """Print the demonstration. Always returns 0: the rejection is the point."""

    prices = _random_walk(seed=20260807)
    raw = [prices[index + 1] / prices[index] - 1.0 for index in range(PERIODS)]
    baseline = statistics.pstdev(raw)
    volatility = {window: _rolling_volatility(raw, window) for window in VOL_WINDOWS}
    grid = list(product(ENTRY_LOOKBACKS, EXIT_LOOKBACKS, VOL_WINDOWS, VOL_THRESHOLDS))

    print("trialgate demo — optimizing 4 parameters over a PURE RANDOM WALK")
    print(f"{PERIODS} periods of zero-drift noise, {len(grid)} configurations, no edge exists.\n")

    with tempfile.TemporaryDirectory() as workspace:
        registry = Path(workspace) / "registry.jsonl"

        best_sharpe = float("-inf")
        best_trial_id = 0
        best_label = ""
        for entry, exit_, window, threshold in grid:
            returns, costs = _strategy_returns(
                prices,
                raw,
                volatility[window],
                baseline,
                entry_lookback=entry,
                exit_lookback=exit_,
                vol_window=window,
                vol_threshold=threshold,
            )
            annualized = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)

            # THIS is the integration point: every backtest you run, including
            # the quick ones, gets registered together with its return series.
            record = append_trial(
                registry,
                recorded_at=datetime.now(tz=UTC),
                code_version="demo",
                strategy_id="trend_with_vol_filter",
                parameters={
                    "entry_lookback": str(entry),
                    "exit_lookback": str(exit_),
                    "vol_window": str(window),
                    "vol_threshold": str(threshold),
                },
                universe=("SYNTHETIC",),
                data_start=datetime(2018, 1, 1, tzinfo=UTC),
                data_end=datetime(2025, 12, 31, tzinfo=UTC),
                cost_assumptions={"round_trip_bps": f"{ROUND_TRIP_BPS:g}"},
                metrics={"annualized_sharpe": f"{annualized:.4f}"},
                operator_note=f"entry={entry} exit={exit_} vol={window}/{threshold}",
                period_returns=returns,
                period_costs=costs,
            )
            if annualized > best_sharpe:
                best_sharpe = annualized
                best_trial_id = record.trial_id
                best_label = record.operator_note

        print(f"Best in-sample annualized Sharpe: {best_sharpe:.2f}   ({best_label})")
        print("On data that is definitionally random. This is what a single")
        print("backtest buys you, and it is why one is not evidence.\n")

        print(f"Registered trials (the honest N): {len(grid)}")
        print(verify_chain(registry).summary(), "\n")

        report = evaluate_registry(
            registry,
            candidate_trial_id=best_trial_id,
            periods_per_year=PERIODS_PER_YEAR,
            effective_trials=len(grid),
            strategy_id="trend_with_vol_filter",
            stop_loss_fraction=STOP_LOSS_FRACTION,
        )
        print(report.summary())

    print("\nTo run this against your own strategy, change three things:")
    print("  1. replace _random_walk() with your price or return data")
    print("  2. replace _strategy_returns() with your backtest")
    print("  3. point the registry at a real path so N accumulates across runs")
    print("Everything else — registration, the matrix, the gates — is unchanged.")
    # NOT QUALIFIED is this demo succeeding, so the exit status is 0. Only
    # `trialgate verify` reports a verdict through its exit code.
    return 0
