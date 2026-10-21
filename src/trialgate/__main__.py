"""Command line: audit a registry, fill it from an optimizer, judge, and watch live.

The point of a tamper-evident registry is that someone who does not trust you
can check it. That person should not have to write Python:

    trialgate verify state/registry.jsonl

Exit status is 0 when the chain is intact and 1 when it is not, so this drops
straight into CI or a pre-commit hook.

A freqtrade user goes from hyperopt to a verdict without writing Python either:

    trialgate import-freqtrade user_data/hyperopt_results --registry state/registry.jsonl
    trialgate evaluate state/registry.jsonl --trial 412 --periods-per-year 365 \\
        --effective-trials 500
    trialgate live-lock state/registry.jsonl --trial 412 --horizon 90 \\
        --out state/live_bands.json
    trialgate live-check state/live_bands.json live_returns.txt

``evaluate`` exits 1 when the trial is not qualified and ``live-check`` exits 1
on STOP, so both can gate a deploy script.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from trialgate.registry import verify_chain

# What a wrong path, a wrong trial id or a refused lock raises (every trialgate
# error is a ValueError). The message is the whole answer; the traceback above
# it only hid it.
_USER_ERRORS = (OSError, ValueError)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trialgate",
        description="Tell a backtest that is skill from one that is luck: register every "
        "trial, judge the candidate, then watch it live.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    verify = subcommands.add_parser(
        "verify",
        help="check the registry hash chain and report what it proves",
    )
    verify.add_argument("registry", help="path to the registry JSONL file")

    subcommands.add_parser(
        "demo",
        help="optimize 4 parameters over a pure random walk and watch the gates reject it",
    )

    freqtrade = subcommands.add_parser(
        "import-freqtrade",
        help="register every freqtrade hyperopt epoch (.fthypt) as a trial",
    )
    freqtrade.add_argument("results", help="a .fthypt file or user_data/hyperopt_results")
    freqtrade.add_argument("--registry", required=True, help="registry JSONL to append to")
    freqtrade.add_argument("--holdout", help="gate 5 lock file to enforce")
    freqtrade.add_argument("--code-version", default="unrecorded", help="e.g. a git commit")

    evaluate = subcommands.add_parser("evaluate", help="judge one registered trial")
    evaluate.add_argument("registry")
    evaluate.add_argument("--trial", type=int, required=True, help="candidate trial id")
    evaluate.add_argument("--periods-per-year", type=int, required=True)
    evaluate.add_argument(
        "--effective-trials",
        type=int,
        required=True,
        help="how many independent tries; the registered N is the conservative choice",
    )
    evaluate.add_argument("--strategy", help="judge only trials with this strategy_id")
    evaluate.add_argument("--stop-loss", type=float, help="hard stop as a fraction, for gate 7")
    evaluate.add_argument(
        "--autocorrelation", choices=("report", "adjust", "ignore"), default="report"
    )
    stress = evaluate.add_mutually_exclusive_group()
    stress.add_argument(
        "--cost-multiplier",
        type=float,
        default=2.0,
        help="gate 9 charges the recorded costs this many times over (default 2)",
    )
    stress.add_argument(
        "--no-cost-stress",
        action="store_true",
        help="skip gate 9, for a registry whose trials carry no period costs",
    )

    lock = subcommands.add_parser("live-lock", help="gate 6: lock the live bands before going live")
    lock.add_argument("registry")
    lock.add_argument("--trial", type=int, required=True, help="the trial going live")
    lock.add_argument("--horizon", type=int, required=True, help="live periods to watch")
    lock.add_argument("--out", required=True, help="where to write the lock file")
    lock.add_argument("--false-alarm", type=float, default=0.05)
    lock.add_argument("--seed", type=int, default=0)

    check = subcommands.add_parser("live-check", help="gate 6: judge live returns so far")
    check.add_argument("bands", help="the lock file live-lock wrote")
    check.add_argument("returns", help="live returns: a JSON list, or one number per line")
    check.add_argument("--registry", help="recompute the bands from it to confirm the lock")

    args = parser.parse_args(argv)
    try:
        return _run(args)
    except _USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.command == "demo":
        from trialgate.demo import run_demo

        return run_demo()
    if args.command == "import-freqtrade":
        return _import_freqtrade(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "live-lock":
        return _live_lock(args)
    if args.command == "live-check":
        return _live_check(args)

    # verify_chain() reads an absent file as an empty registry, which is right
    # for a log about to be started and wrong for one being audited: a mistyped
    # path printed OK and exited 0.
    if not Path(args.registry).is_file():
        print(f"FAIL  {args.registry}\n      no registry file at this path", file=sys.stderr)
        return 1
    report = verify_chain(args.registry)
    print(report.summary(), file=sys.stdout if report.ok else sys.stderr)
    return 0 if report.ok else 1


def _import_freqtrade(args: argparse.Namespace) -> int:
    from trialgate.capture import CAPTURE_SOURCE, import_freqtrade_results
    from trialgate.registry import load_trials

    report = import_freqtrade_results(
        args.results, args.registry, code_version=args.code_version, holdout=args.holdout
    )
    print(report.summary())
    imported = [
        trial
        for trial in load_trials(args.registry)
        if trial.metrics.get(CAPTURE_SOURCE, "").startswith("freqtrade:")
        and "loss" in trial.metrics
    ]
    if imported:
        best = min(imported, key=lambda trial: float(trial.metrics["loss"]))
        print(
            f"freqtrade's own pick (lowest loss) is trial {best.trial_id}, "
            f"{best.metrics[CAPTURE_SOURCE].removeprefix('freqtrade:')}. Judge it with:\n"
            f"  trialgate evaluate {args.registry} --trial {best.trial_id} "
            f"--periods-per-year 365 --effective-trials {len(imported)}"
            + (f" --strategy {best.strategy_id}" if " " not in best.strategy_id else "")
        )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from trialgate.gates import evaluate_registry

    report = evaluate_registry(
        args.registry,
        candidate_trial_id=args.trial,
        periods_per_year=args.periods_per_year,
        effective_trials=args.effective_trials,
        strategy_id=args.strategy,
        stop_loss_fraction=args.stop_loss,
        autocorrelation=args.autocorrelation,
        cost_multiplier=None if args.no_cost_stress else args.cost_multiplier,
    )
    print(report.summary())
    return 0 if report.passed else 1


def _live_lock(args: argparse.Namespace) -> int:
    from trialgate.live import lock_live_bands

    bands = lock_live_bands(
        args.out,
        registry=args.registry,
        candidate_trial_id=args.trial,
        horizon=args.horizon,
        locked_at=datetime.now(tz=UTC),
        false_alarm_rate=args.false_alarm,
        seed=args.seed,
    )
    last = bands.horizon - 1
    print(
        f"LOCKED  gate 6 for trial {bands.candidate_trial_id} -> {args.out}\n"
        f"        over {bands.horizon} live periods, STOP if cumulative return falls "
        f"below {bands.min_cumulative_return[last]:+.2%}, drawdown exceeds "
        f"{bands.max_drawdown[last]:.2%}, or {bands.max_losing_streak[last]} losing periods "
        "run in a row (tighter bands apply earlier; see the file)\n"
        f"        false alarm on a strategy that is what its backtest says: "
        f"{bands.measured_false_alarm_rate:.1%} (target {bands.false_alarm_rate:.0%})\n"
        f"        the same returns with no edge are stopped {bands.power_if_no_edge:.0%} of the "
        "time; a longer horizon raises that"
    )
    return 0


def _live_check(args: argparse.Namespace) -> int:
    from trialgate.live import check_live

    text = Path(args.returns).read_text(encoding="utf-8").strip()
    returns: list[float] = []
    if text.startswith("["):
        returns = [float(value) for value in json.loads(text)]
    else:
        # One return per line; a CSV whose last column is the return, with or
        # without a header row, reads the same way.
        rows = [line for line in text.splitlines() if line.strip() and line.lstrip()[0] != "#"]
        for index, line in enumerate(rows):
            try:
                returns.append(float(line.split(",")[-1]))
            except ValueError:
                if index != 0:
                    raise
    verdict = check_live(args.bands, returns, registry=args.registry)
    print(verdict.summary(), file=sys.stderr if verdict.status == "STOP" else sys.stdout)
    return 1 if verdict.status == "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
