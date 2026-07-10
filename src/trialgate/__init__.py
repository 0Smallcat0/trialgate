"""trialgate: an anti-overfitting validation gate for backtested strategies.

A strategy earns belief by surviving verification, not by looking good
in-sample. This package provides the mechanical parts of a six-gate
qualification standard: an append-only trial registry, CSCV/PBO and
Deflated Sharpe Ratio statistics, a single-use holdout lock, and threshold
gates that combine them into a verdict.
"""

from trialgate.gates import (
    DEFAULT_MAX_PBO,
    DEFAULT_MIN_DSR,
    DEFAULT_MIN_OBSERVATIONS,
    GateCheck,
    GateReport,
    evaluate_gates,
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
    TrialRecord,
    TrialRegistryError,
    append_trial,
    config_hash_for,
    load_trials,
    trial_count,
)
from trialgate.validation import (
    DeflatedSharpeResult,
    PBOResult,
    ValidationInputError,
    deflated_sharpe_ratio,
    non_annualized_sharpe_variance,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MAX_PBO",
    "DEFAULT_MIN_DSR",
    "DEFAULT_MIN_OBSERVATIONS",
    "DeflatedSharpeResult",
    "GateCheck",
    "GateReport",
    "HoldoutState",
    "HoldoutViolationError",
    "PBOResult",
    "TrialRecord",
    "TrialRegistryError",
    "ValidationInputError",
    "append_trial",
    "config_hash_for",
    "deflated_sharpe_ratio",
    "evaluate_gates",
    "initialize_holdout",
    "load_holdout",
    "load_trials",
    "non_annualized_sharpe_variance",
    "probability_of_backtest_overfitting",
    "require_data_allowed",
    "sharpe_ratio",
    "spend_holdout",
    "trial_count",
]
