"""trialgate: an anti-overfitting validation gate for backtested strategies.

A strategy earns belief by surviving verification, not by looking good
in-sample. This package provides the mechanical parts of a qualification
standard: a tamper-evident append-only trial registry, CSCV/PBO and Deflated
Sharpe Ratio statistics, a single-use holdout lock, a stop-line survivability
check, and threshold gates that combine them into a verdict.
"""

from trialgate.autocorrelation import AutocorrelationDiagnostics
from trialgate.gates import (
    DEFAULT_MAX_PBO,
    DEFAULT_MAX_STOP_BREACH,
    DEFAULT_MIN_DSR,
    DEFAULT_MIN_OBSERVATIONS,
    GateCheck,
    GateReport,
    evaluate_gates,
    evaluate_registry,
)
from trialgate.holdout import (
    HoldoutState,
    HoldoutViolationError,
    initialize_holdout,
    load_holdout,
    require_data_allowed,
    spend_holdout,
)
from trialgate.registry import (
    ChainReport,
    RegistryTamperError,
    TrialRecord,
    TrialRegistryError,
    append_trial,
    config_hash_for,
    load_trials,
    performance_matrix,
    trial_count,
    verify_chain,
)
from trialgate.risk import (
    RiskInputError,
    StopLineResult,
    assess_stop_line,
    max_volatility_for_stop,
    stop_breach_probability,
)
from trialgate.validation import (
    AnnualizedSharpe,
    AutocorrelationMode,
    DataRequirement,
    DeflatedSharpeResult,
    PBOResult,
    ValidationInputError,
    deflated_sharpe_ratio,
    lo_annualized_sharpe,
    observations_required_for_dsr,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

__version__ = "0.5.1"

__all__ = [
    "DEFAULT_MAX_PBO",
    "DEFAULT_MAX_STOP_BREACH",
    "DEFAULT_MIN_DSR",
    "DEFAULT_MIN_OBSERVATIONS",
    "AnnualizedSharpe",
    "AutocorrelationDiagnostics",
    "AutocorrelationMode",
    "ChainReport",
    "DataRequirement",
    "DeflatedSharpeResult",
    "GateCheck",
    "GateReport",
    "HoldoutState",
    "HoldoutViolationError",
    "PBOResult",
    "RegistryTamperError",
    "RiskInputError",
    "StopLineResult",
    "TrialRecord",
    "TrialRegistryError",
    "ValidationInputError",
    "append_trial",
    "assess_stop_line",
    "config_hash_for",
    "deflated_sharpe_ratio",
    "evaluate_gates",
    "evaluate_registry",
    "initialize_holdout",
    "load_holdout",
    "load_trials",
    "lo_annualized_sharpe",
    "max_volatility_for_stop",
    "observations_required_for_dsr",
    "performance_matrix",
    "probability_of_backtest_overfitting",
    "require_data_allowed",
    "sharpe_ratio",
    "spend_holdout",
    "stop_breach_probability",
    "trial_count",
    "verify_chain",
]
