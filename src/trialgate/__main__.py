"""Command line audit for a trial registry.

The point of a tamper-evident registry is that someone who does not trust you
can check it. That person should not have to write Python:

    trialgate verify state/registry.jsonl

Exit status is 0 when the chain is intact and 1 when it is not, so this drops
straight into CI or a pre-commit hook.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from trialgate.registry import verify_chain


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trialgate",
        description="Audit an append-only trial registry.",
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

    args = parser.parse_args(argv)

    if args.command == "demo":
        from trialgate.demo import run_demo

        return run_demo()

    report = verify_chain(args.registry)
    print(report.summary(), file=sys.stdout if report.ok else sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
