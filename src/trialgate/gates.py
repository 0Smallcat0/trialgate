"""Threshold gates and the combined qualification report.

The qualification standard (numbering follows the crypto signal system this
library was extracted from):

1. Trial registry (structural)      -> trialgate.registry
2. Data floor (statistical)         -> judged here
3. PBO <= 0.05 (statistical)        -> computed by trialgate.validation, judged here
4. DSR >= 0.95 (statistical)        -> computed by trialgate.validation, judged here
5. Single-use holdout (structural)  -> trialgate.holdout
6. Paper trading with measured real costs (procedural) -> out of library
   scope on purpose: it is a live measurement over months, not a computation.
7. Stop-line survivability (statistical) -> computed by trialgate.risk, judged here
8. Declared costs (structural)      -> judged here

Gates 7 and 8 arrived in 0.5.0 from the same observation: a strategy can clear
every statistical gate and still be untradeable, either because its own
volatility trips the operator's hard stop before the edge pays (7), or because
the returns being judged were never charged for execution in the first place
(8). Gate 8 is on by default and will fail registries written before 0.5.0
that recorded ``cost_assumptions={}`` -- that is the intended verdict on a
gross-of-cost backtest, not an upgrade accident.

Passing every gate is a NECESSARY condition for trusting a backtest, not a
profit guarantee: in a verified 2022 case study the least-overfit strategy
still lost ~35% in two months.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from trialgate.registry import load_trials, performance_matrix
from trialgate.risk import assess_stop_line
from trialgate.validation import (
    AutocorrelationMode,
    DeflatedSharpeResult,
    ValidationInputError,
    deflated_sharpe_ratio,
    lo_annualized_sharpe,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

DEFAULT_MIN_OBSERVATIONS = 1000
DEFAULT_MAX_PBO = 0.05
DEFAULT_MIN_DSR = 0.95
DEFAULT_MAX_STOP_BREACH = 0.10


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One threshold verdict.

    ``detail`` carries the instruction a bare number cannot: gate 7 fails with
    a volatility to size down to, gate 8 fails with the trials that were never
    charged for execution. It defaults to empty so pre-0.5.0 construction
    keeps working.
    """

    gate: str
    passed: bool
    observed: float
    threshold: float
    comparison: str
    detail: str = ""

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = f"{status}  {self.gate}: {self.observed:g} {self.comparison} {self.threshold:g}"
        return f"{line}  -- {self.detail}" if self.detail else line


@dataclass(frozen=True, slots=True)
class GateReport:
    """Combined verdict over the judgeable gates."""

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
    stop_breach: float | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    max_pbo: float = DEFAULT_MAX_PBO,
    min_dsr: float = DEFAULT_MIN_DSR,
    max_stop_breach: float = DEFAULT_MAX_STOP_BREACH,
) -> GateReport:
    """Judge the statistical gates against their thresholds.

    Feed the honest numbers: ``observations`` from the actual data span,
    ``pbo`` from probability_of_backtest_overfitting() over ALL registered
    trials, ``deflated_sharpe`` from deflated_sharpe_ratio() using the
    registry's full trial count. Flattering inputs void the report.

    ``stop_breach`` (gate 7) is optional here because it needs a stop-loss
    policy this function is not given; supply it from
    trialgate.risk.stop_breach_probability() or use evaluate_registry(), which
    derives both moments from the registered return series.
    """

    checks = _checks(observations, pbo, deflated_sharpe, min_observations, max_pbo, min_dsr)
    if stop_breach is not None:
        checks = (*checks, _stop_check(stop_breach, max_stop_breach, ""))
    return GateReport(checks=checks)


def evaluate_registry(
    path: str | Path,
    *,
    candidate_trial_id: int,
    periods_per_year: int,
    effective_trials: int,
    strategy_id: str | None = None,
    block_count: int = 16,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    max_pbo: float = DEFAULT_MAX_PBO,
    min_dsr: float = DEFAULT_MIN_DSR,
    stop_loss_fraction: float | None = None,
    horizon_years: float = 1.0,
    max_stop_breach: float = DEFAULT_MAX_STOP_BREACH,
    require_declared_costs: bool = True,
    autocorrelation: AutocorrelationMode = "report",
) -> GateReport:
    """Judge the gates for one registered trial, using only registered data.

    Every input the operator could otherwise curate comes out of the registry:
    the CSCV matrix is assembled from all registered trials, the candidate's
    return series is a column of that matrix, the trial Sharpe spread that the
    DSR deflates by is computed from those same columns rather than read from
    the self-reported ``metrics`` strings, and gate 7's volatility and drift
    are moments of that same column.

    ``effective_trials`` is required, with no default, because it is the last
    number the library cannot derive: trials from a nested parameter grid are
    correlated, so the honest count is somewhere below the registered N and
    nobody agrees on how to find it. Passing the registered N is the
    conservative choice -- it can only reject, never wave something through.
    Whatever you pass, you had to type it.

    ``stop_loss_fraction`` is likewise required to activate gate 7 and has no
    default, for the opposite reason: a stop line is a policy, and defaulting
    one would be inventing the operator's risk appetite rather than measuring
    against it. Omitting it skips gate 7 instead of guessing.

    ``autocorrelation`` governs gate 4 when the candidate's returns are
    serially correlated, which the DSR formula assumes away. The default,
    ``"report"``, keeps the IID verdict and, when the Ljung-Box test finds
    the correlation significant, attaches a warning with the
    autocorrelation-adjusted DSR, the effective sample size, and the
    annualized Sharpe by Lo's eta(q) (Lo 2002, Eq. (20) p. 40). ``"adjust"``
    judges gate 4 on the adjusted DSR instead; ``"ignore"`` skips the
    estimate. The adjusted DSR is trialgate's extension of Bailey & Lopez de
    Prado's statistic, not their method; see trialgate.autocorrelation.
    """

    trials = [
        trial
        for trial in load_trials(path)
        if strategy_id is None or trial.strategy_id == strategy_id
    ]
    trial_ids = [trial.trial_id for trial in trials]
    if candidate_trial_id not in trial_ids:
        msg = (
            f"trial {candidate_trial_id} is not registered"
            + (f" under strategy_id {strategy_id!r}" if strategy_id else "")
            + f"; registered ids are {trial_ids[:10]}{'...' if len(trial_ids) > 10 else ''}"
        )
        raise ValueError(msg)

    matrix = performance_matrix(path, strategy_id=strategy_id)
    candidate_column = trial_ids.index(candidate_trial_id)
    candidate_returns = [row[candidate_column] for row in matrix]
    trial_sharpes = [
        sharpe_ratio([row[column] for row in matrix], periods_per_year=periods_per_year)
        for column in range(len(matrix[0]))
    ]

    pbo = probability_of_backtest_overfitting(matrix, block_count=block_count)
    dsr = deflated_sharpe_ratio(
        candidate_returns,
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
        effective_trials=effective_trials,
        autocorrelation=autocorrelation,
    )
    checks = _checks(
        len(matrix),
        pbo.pbo,
        dsr.deflated_sharpe_ratio,
        min_observations,
        max_pbo,
        min_dsr,
        _autocorrelation_detail(dsr, candidate_returns, periods_per_year, min_dsr),
    )

    if stop_loss_fraction is not None:
        stop = assess_stop_line(
            candidate_returns,
            periods_per_year=periods_per_year,
            stop_loss_fraction=stop_loss_fraction,
            horizon_years=horizon_years,
            max_breach_probability=max_stop_breach,
        )
        detail = (
            f"ran at {stop.annual_volatility:.1%} annualized volatility against a "
            f"-{stop_loss_fraction:.0%} stop; survivable volatility is "
            f"{stop.survivable_volatility:.1%} (size x{stop.volatility_scale:.2f})"
        )
        checks = (*checks, _stop_check(stop.breach_probability, max_stop_breach, detail))

    if require_declared_costs:
        undeclared = [
            trial.trial_id
            for trial in trials
            if not _declares_positive_cost(trial.cost_assumptions)
        ]
        shown = ", ".join(str(trial_id) for trial_id in undeclared[:5])
        checks = (
            *checks,
            GateCheck(
                gate="declared costs (gate 8)",
                passed=not undeclared,
                observed=float(len(trials) - len(undeclared)),
                threshold=float(len(trials)),
                comparison=">=",
                detail=(
                    ""
                    if not undeclared
                    else f"{len(undeclared)} trials priced execution at zero "
                    "or left it out of their returns: "
                    f"{shown}{'...' if len(undeclared) > 5 else ''}"
                ),
            ),
        )

    return GateReport(checks=checks)


def _autocorrelation_detail(
    dsr: DeflatedSharpeResult, returns: list[float], periods_per_year: int, min_dsr: float
) -> str:
    """Gate 4's warning (``"report"``) or provenance (``"adjust"``); empty otherwise."""

    diagnostics = dsr.autocorrelation
    if diagnostics is None or dsr.adjusted_deflated_sharpe_ratio is None:
        return ""
    if not dsr.autocorrelation_adjusted and not diagnostics.significant:
        return ""
    test = (
        f"lag-1 autocorrelation {diagnostics.first_order_autocorrelation:+.2f}, "
        f"Ljung-Box Q({diagnostics.ljung_box_lags}) p={diagnostics.ljung_box_p_value:.2g}"
    )
    effective = (
        f"{diagnostics.effective_observations:.0f} effective of {dsr.observations} observations"
    )
    if dsr.autocorrelation_adjusted:
        return (
            f"autocorrelation-adjusted on {effective} ({test}); "
            f"IID DSR {dsr.iid_deflated_sharpe_ratio:g}"
        )
    verdict = "PASS" if dsr.adjusted_deflated_sharpe_ratio >= min_dsr else "FAIL"
    try:
        annual = lo_annualized_sharpe(returns, periods_per_year=periods_per_year)
    except ValidationInputError:
        # The truncated autocorrelations can sum to an impossible q-period
        # variance on a short, strongly negative series; the warning above
        # stands without the annualized figure rather than failing the report.
        sharpe = ""
    else:
        sharpe = (
            f"; annualized Sharpe {annual.iid_annualized_sharpe:.2f} by "
            f"sqrt({periods_per_year}) is {annual.annualized_sharpe:.2f} by Lo's eta(q)"
        )
    return (
        f"WARNING returns are serially correlated ({test}), which this DSR assumes "
        f"away: adjusted for it, DSR is {dsr.adjusted_deflated_sharpe_ratio:g} on "
        f"{effective} (would {verdict}){sharpe}. "
        'Pass autocorrelation="adjust" to judge the adjusted value'
    )


def _stop_check(breach: float, max_stop_breach: float, detail: str) -> GateCheck:
    return GateCheck(
        gate="stop-line survivability (gate 7)",
        passed=breach <= max_stop_breach,
        observed=breach,
        threshold=max_stop_breach,
        comparison="<=",
        detail=detail,
    )


_CHARGED_FLAGS = frozenset({"charged", "deducted", "net_of_costs"})
_FALSE_WORDS = frozenset({"", "0", "f", "false", "n", "no", "none"})


def _declares_positive_cost(assumptions: Mapping[str, str]) -> bool:
    """True when at least one cost assumption parses to a strictly positive number
    and no assumption says the returns were not charged it.

    A missing key and an explicit "0" are treated the same on purpose. Both
    describe a backtest that was never charged for execution, and only one of
    them looks like an oversight.

    A cost that is declared and then flagged as left out -- ``charged``,
    ``deducted`` or ``net_of_costs`` set to false -- is the same case a third
    way: the number says what execution would cost, and the returns being
    judged never paid it. The flags are yes/no answers, not prices, so they do
    not count as the positive cost either.
    """

    normalized = {
        str(key).strip().lower(): str(raw).strip().lower() for key, raw in assumptions.items()
    }
    if any(normalized[flag] in _FALSE_WORDS for flag in _CHARGED_FLAGS & normalized.keys()):
        return False
    for key, raw in normalized.items():
        if key in _CHARGED_FLAGS:
            continue
        try:
            if float(raw) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _checks(
    observations: int,
    pbo: float,
    deflated_sharpe: float,
    min_observations: int,
    max_pbo: float,
    min_dsr: float,
    dsr_detail: str = "",
) -> tuple[GateCheck, ...]:
    return (
        GateCheck(
            gate="data floor (gate 2)",
            passed=observations >= min_observations,
            observed=float(observations),
            threshold=float(min_observations),
            comparison=">=",
            detail=(
                ""
                if observations >= min_observations
                else f"{min_observations - observations} more observations needed"
            ),
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
            detail=dsr_detail,
        ),
    )
