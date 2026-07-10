"""Threshold gates and the combined qualification report.

The six-gate qualification standard (numbering follows the reference
deployment, a live crypto signal system):

1. Trial registry (structural)      -> trialgate.registry
2. Data floor (statistical)         -> judged here
3. PBO <= 0.05 (statistical)        -> computed by trialgate.validation, judged here
4. DSR >= 0.95 (statistical)        -> computed by trialgate.validation, judged here
5. Single-use holdout (structural)  -> trialgate.holdout
6. Paper trading with measured real costs (procedural) -> out of library
   scope on purpose: it is a live measurement over months, not a computation.

Passing every gate is a NECESSARY condition for trusting a backtest, not a
profit guarantee: in a verified 2022 case study the least-overfit strategy
still lost ~35% in two months.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MIN_OBSERVATIONS = 1000
DEFAULT_MAX_PBO = 0.05
DEFAULT_MIN_DSR = 0.95


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One threshold verdict."""

    gate: str
    passed: bool
    observed: float
    threshold: float
    comparison: str

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status}  {self.gate}: {self.observed:g} {self.comparison} {self.threshold:g}"


@dataclass(frozen=True, slots=True)
class GateReport:
    """Combined verdict over the statistical gates (2-4)."""

    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        lines = [check.describe() for check in self.checks]
        lines.append("QUALIFIED" if self.passed else "NOT QUALIFIED")
        return "\n".join(lines)


def evaluate_gates(
    *,
    observations: int,
    pbo: float,
    deflated_sharpe: float,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    max_pbo: float = DEFAULT_MAX_PBO,
    min_dsr: float = DEFAULT_MIN_DSR,
) -> GateReport:
    """Judge the statistical gates (2-4) against their thresholds.

    Feed the honest numbers: ``observations`` from the actual data span,
    ``pbo`` from probability_of_backtest_overfitting() over ALL registered
    trials, ``deflated_sharpe`` from deflated_sharpe_ratio() using the
    registry's full trial count. Flattering inputs void the report.
    """

    checks = (
        GateCheck(
            gate="data floor (gate 2)",
            passed=observations >= min_observations,
            observed=float(observations),
            threshold=float(min_observations),
            comparison=">=",
        ),
        GateCheck(
            gate="PBO (gate 3)",
            passed=pbo <= max_pbo,
            observed=pbo,
            threshold=max_pbo,
            comparison="<=",
        ),
        GateCheck(
            gate="DSR (gate 4)",
            passed=deflated_sharpe >= min_dsr,
            observed=deflated_sharpe,
            threshold=min_dsr,
            comparison=">=",
        ),
    )
    return GateReport(checks=checks)
