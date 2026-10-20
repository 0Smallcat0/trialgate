"""Serial correlation: the Sharpe standard error and annualization against independent checks.

What these pin:

- On IID returns the default DSR is the number earlier releases returned, to
  the last bit, and the gate prints no warning.
- On AR(1) returns with phi from -0.5 to 0.8 the adjusted standard error of
  the Sharpe ratio matches both a Monte Carlo of the estimator and the
  closed-form asymptotic variance, where the IID one is off by up to 3x.
- Lo's eta(q) matches its own AR(1) closed form and a direct simulation of
  q-period sums.
- The Ljung-Box statistic, its p-value and the prewhitened long-run variance
  match a separate numpy/scipy implementation.

Every seed is fixed, so the simulations are deterministic.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from trialgate import (
    ValidationInputError,
    append_trial,
    deflated_sharpe_ratio,
    evaluate_registry,
    lo_annualized_sharpe,
    observations_required_for_dsr,
    sharpe_ratio,
)
from trialgate.autocorrelation import (
    ar1_eta,
    eta,
    ljung_box,
    sharpe_variance_ratio,
)

_PHIS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.8)
_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
_START = datetime(2019, 1, 1, tzinfo=UTC)
_END = datetime(2025, 6, 30, tzinfo=UTC)


def _ar1(rng: np.random.Generator, phi: float, count: int, sharpe: float) -> np.ndarray:
    """Stationary Gaussian AR(1) with unit variance and per-period Sharpe ``sharpe``."""

    shocks = rng.standard_normal(count) * math.sqrt(1.0 - phi * phi)
    series = np.empty(count)
    series[0] = rng.standard_normal()
    for index in range(1, count):
        series[index] = phi * series[index - 1] + shocks[index]
    return sharpe + series


# --- the default must not move on IID data --------------------------------

# deflated_sharpe_ratio() on origin/fix/statistics-audit, before this change.
_PINNED_IID_DSR = [
    (101, 0.0007, 0.44263299537265827),
    (202, 0.0007, 0.0938728667656496),
    (303, 0.0007, 0.5874748106495211),
    (404, 0.003, 0.9996341624433067),
]


@pytest.mark.parametrize(("seed", "drift", "pinned"), _PINNED_IID_DSR)
def test_default_dsr_is_bit_identical_to_the_previous_release(
    seed: int, drift: float, pinned: float
) -> None:
    rng = random.Random(seed)
    returns = [rng.gauss(drift, 0.01) for _ in range(750)]
    sharpes = [rng.gauss(0.5, 0.7) for _ in range(30)]

    default = deflated_sharpe_ratio(
        returns, trial_sharpes=sharpes, periods_per_year=252, effective_trials=40
    )
    ignored = deflated_sharpe_ratio(
        returns,
        trial_sharpes=sharpes,
        periods_per_year=252,
        effective_trials=40,
        autocorrelation="ignore",
    )

    assert default.deflated_sharpe_ratio == pinned
    assert ignored.deflated_sharpe_ratio == pinned
    assert default.iid_deflated_sharpe_ratio == pinned
    assert not default.autocorrelation_adjusted
    assert ignored.autocorrelation is None


def test_data_requirement_default_is_unchanged() -> None:
    rng = random.Random(303)
    returns = [rng.gauss(0.0007, 0.01) for _ in range(750)]
    sharpes = [rng.gauss(0.5, 0.7) for _ in range(30)]

    requirement = observations_required_for_dsr(
        returns, trial_sharpes=sharpes, periods_per_year=252, effective_trials=40
    )

    assert requirement.observations_required == 41472  # previous release


def _register_iid(path: Path, configs: int, periods: int, seed: int) -> None:
    rng = random.Random(seed)
    for config in range(configs):
        drift = 0.0012 if config == 0 else 0.0
        append_trial(
            path,
            recorded_at=_NOW,
            code_version="test",
            strategy_id="iid",
            parameters={"config": str(config)},
            universe=("SYNTH",),
            data_start=_START,
            data_end=_END,
            cost_assumptions={"round_trip_bps": "10"},
            metrics={},
            operator_note=f"config {config}",
            period_returns=[rng.gauss(drift, 0.01) for _ in range(periods)],
        )


def test_gate_report_on_iid_returns_is_unchanged(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    _register_iid(registry, configs=8, periods=640, seed=9)
    kwargs: dict[str, Any] = {
        "candidate_trial_id": 1,
        "periods_per_year": 252,
        "effective_trials": 8,
        "block_count": 8,
        "min_observations": 100,
    }

    default = evaluate_registry(registry, **kwargs)
    ignored = evaluate_registry(registry, autocorrelation="ignore", **kwargs)

    assert default.summary() == ignored.summary()
    assert "WARNING" not in default.summary()


# --- AR(1): the standard error against Monte Carlo and theory -------------


@pytest.mark.parametrize("phi", _PHIS)
def test_adjusted_sharpe_standard_error_matches_monte_carlo(phi: float) -> None:
    """Spread of SR-hat over 400 AR(1) paths vs the standard error the DSR uses.

    Gaussian AR(1): the sample mean's long-run variance is (1 + phi)/(1 - phi)
    and the sample variance's carries sum_k rho_k^2 = (1 + phi^2)/(1 - phi^2),
    so T * Var(SR-hat) -> (1 + phi)/(1 - phi) + SR^2/2 * (1 + phi^2)/(1 - phi^2).
    """

    rng = np.random.default_rng(int((phi + 1.0) * 1000))
    count, paths, sharpe = 1000, 400, 0.05
    estimates = []
    standard_errors = []
    for _ in range(paths):
        returns = _ar1(rng, phi, count, sharpe)
        estimates.append(returns.mean() / returns.std(ddof=1))
        standard_errors.append(math.sqrt(sharpe_variance_ratio(returns.tolist()) / (count - 1)))
    empirical = float(np.std(estimates, ddof=1))
    adjusted = float(np.mean(standard_errors))
    theory = math.sqrt(
        ((1 + phi) / (1 - phi) + sharpe**2 / 2 * (1 + phi**2) / (1 - phi**2)) / count
    )
    iid = 1.0 / math.sqrt(count - 1)

    # 400 paths measure a standard deviation to about 3.5% (1 sd).
    assert adjusted == pytest.approx(empirical, rel=0.10)
    assert adjusted == pytest.approx(theory, rel=0.03)
    if abs(phi) >= 0.25:
        assert abs(iid / empirical - 1.0) > 0.20  # what the IID formula would miss


def test_adjusted_dsr_is_calibrated_on_autocorrelated_nulls() -> None:
    """~5% of zero-edge AR(1) strategies should clear 0.95 once adjusted.

    The IID statistic on the same phi = 0.5 paths clears far more often: its
    standard error is sqrt(3) too small, so a z of 1.645 needs only 0.95 sd.
    """

    rng = np.random.default_rng(777)
    trials = 300
    adjusted_cleared = iid_cleared = 0
    for _ in range(trials):
        returns = _ar1(rng, 0.5, 500, 0.0).tolist()
        result = deflated_sharpe_ratio(
            returns,
            trial_sharpes=[1.0],
            periods_per_year=252,
            effective_trials=1,
            autocorrelation="adjust",
        )
        adjusted_cleared += result.deflated_sharpe_ratio >= 0.95
        iid_cleared += (result.iid_deflated_sharpe_ratio or 0.0) >= 0.95

    # Binomial(300, 0.05): mean 15, sd 3.8.
    assert 0.02 <= adjusted_cleared / trials <= 0.09
    assert iid_cleared / trials > 0.12


def test_adjust_mode_judges_the_adjusted_value_and_report_mode_warns(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    registry = tmp_path / "registry.jsonl"
    for config in range(6):
        sharpe = 0.25 if config == 0 else 0.0
        append_trial(
            registry,
            recorded_at=_NOW,
            code_version="test",
            strategy_id="smoothed",
            parameters={"config": str(config)},
            universe=("SYNTH",),
            data_start=_START,
            data_end=_END,
            cost_assumptions={"round_trip_bps": "10"},
            metrics={},
            operator_note=f"config {config}",
            period_returns=(_ar1(rng, 0.8, 640, sharpe) * 0.01).tolist(),
        )
    kwargs: dict[str, Any] = {
        "candidate_trial_id": 1,
        "periods_per_year": 252,
        "effective_trials": 6,
        "block_count": 8,
        "min_observations": 100,
    }

    reported = evaluate_registry(registry, **kwargs)
    adjusted = evaluate_registry(registry, autocorrelation="adjust", **kwargs)
    reported_dsr = next(check for check in reported.checks if check.gate.startswith("DSR"))
    adjusted_dsr = next(check for check in adjusted.checks if check.gate.startswith("DSR"))

    assert reported_dsr.passed
    assert "WARNING returns are serially correlated" in reported_dsr.detail
    assert "(would FAIL)" in reported_dsr.detail
    assert "by Lo's eta(q)" in reported_dsr.detail
    assert not adjusted_dsr.passed
    assert adjusted_dsr.observed < reported_dsr.observed
    assert "autocorrelation-adjusted on" in adjusted_dsr.detail
    assert f"IID DSR {reported_dsr.observed:g}" in adjusted_dsr.detail


def test_data_requirement_scales_by_the_variance_ratio() -> None:
    rng = np.random.default_rng(12)
    returns = (_ar1(rng, 0.5, 800, 0.08) * 0.01).tolist()
    kwargs: dict[str, Any] = {
        "trial_sharpes": [0.4, 1.2, 0.9],
        "periods_per_year": 252,
        "effective_trials": 3,
    }

    iid = observations_required_for_dsr(returns, **kwargs)
    adjusted = observations_required_for_dsr(returns, autocorrelation="adjust", **kwargs)
    ratio = sharpe_variance_ratio(returns)

    assert iid.observations_required is not None
    assert adjusted.observations_required is not None
    assert 2.0 < ratio < 4.0  # (1 + 0.5) / (1 - 0.5) = 3 in population
    assert adjusted.observations_required - 1 == pytest.approx(
        ratio * (iid.observations_required - 1), rel=1e-3
    )


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValidationInputError, match="autocorrelation must be one of"):
        deflated_sharpe_ratio(
            [0.01, -0.02, 0.03, 0.0],
            trial_sharpes=[1.0],
            periods_per_year=252,
            effective_trials=1,
            autocorrelation="on",  # type: ignore[arg-type]
        )


# --- Lo's eta(q) -----------------------------------------------------------


@pytest.mark.parametrize("rho", [-0.4, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8])
@pytest.mark.parametrize("periods", [2, 4, 12, 52])
def test_eta_matches_lo_ar1_closed_form(rho: float, periods: int) -> None:
    autocorrelations = [rho**lag for lag in range(1, periods)]
    by_numpy = periods / math.sqrt(
        periods + 2.0 * np.dot(periods - np.arange(1, periods), np.array(autocorrelations))
    )

    assert eta(periods, autocorrelations) == pytest.approx(ar1_eta(periods, rho), rel=1e-12)
    assert eta(periods, autocorrelations) == pytest.approx(by_numpy, rel=1e-12)


def test_eta_without_autocorrelation_is_square_root_of_q() -> None:
    assert eta(12, []) == pytest.approx(math.sqrt(12), rel=1e-15)
    assert eta(252, [0.0] * 251) == pytest.approx(math.sqrt(252), rel=1e-15)


@pytest.mark.parametrize("rho", [-0.3, 0.2, 0.5])
def test_eta_matches_simulated_q_period_sharpe(rho: float) -> None:
    """eta(q) is defined by the Sharpe of q-period sums; simulate those directly."""

    rng = np.random.default_rng(31)
    periods, sharpe = 12, 0.3
    monthly = _ar1(rng, rho, 12 * 200_000, sharpe)
    annual = monthly.reshape(-1, periods).sum(axis=1)
    simulated = annual.mean() / annual.std()

    assert simulated == pytest.approx(ar1_eta(periods, rho) * sharpe, rel=0.02)


def test_lo_annualized_sharpe_with_all_lags_matches_numpy() -> None:
    rng = np.random.default_rng(8)
    monthly = (_ar1(rng, 0.4, 180, 0.25) * 0.02).tolist()

    result = lo_annualized_sharpe(monthly, periods_per_year=12, max_lag=11)
    centered = np.asarray(monthly) - np.mean(monthly)
    rhos = np.array(
        [
            np.dot(centered[lag:], centered[:-lag]) / np.dot(centered, centered)
            for lag in range(1, 12)
        ]
    )
    factor = 12 / math.sqrt(12 + 2 * np.dot(12 - np.arange(1, 12), rhos))
    per_period = np.mean(monthly) / np.std(monthly, ddof=1)

    assert result.eta == pytest.approx(factor, rel=1e-12)
    assert result.annualized_sharpe == pytest.approx(per_period * factor, rel=1e-12)
    assert result.iid_annualized_sharpe == sharpe_ratio(monthly, periods_per_year=12)
    assert result.annualized_sharpe < result.iid_annualized_sharpe


def test_lo_annualized_sharpe_on_iid_returns_stays_near_square_root() -> None:
    rng = random.Random(4)
    daily = [rng.gauss(0.0005, 0.01) for _ in range(2000)]

    result = lo_annualized_sharpe(daily, periods_per_year=252)

    assert 1 <= result.max_lag <= 3
    assert result.eta == pytest.approx(math.sqrt(252), rel=0.10)


def test_lo_annualized_sharpe_refuses_lags_beyond_q() -> None:
    with pytest.raises(ValidationInputError, match="max_lag"):
        lo_annualized_sharpe([0.01, -0.02, 0.03, 0.0, 0.01], periods_per_year=4, max_lag=4)


# --- independent numpy/scipy implementations --------------------------------


def _numpy_ljung_box(values: np.ndarray, lags: int) -> tuple[float, float]:
    centered = values - values.mean()
    count = len(values)
    acf = np.correlate(centered, centered, mode="full")[count - 1 :] / np.dot(centered, centered)
    statistic = (
        count * (count + 2) * np.sum(acf[1 : lags + 1] ** 2 / (count - np.arange(1, lags + 1)))
    )
    return float(statistic), float(stats.chi2.sf(statistic, lags))


def _numpy_variance_ratio(values: np.ndarray) -> float:
    """Andrews-Monahan prewhitened Bartlett HAC of the Sharpe influence function."""

    mean, sigma = values.mean(), values.std()
    psi = (values - mean) / sigma - mean / (2 * sigma**3) * ((values - mean) ** 2 - sigma**2)
    coefficient = float(
        np.clip(np.dot(psi[1:], psi[:-1]) / np.dot(psi[:-1], psi[:-1]), -0.97, 0.97)
    )
    residuals = psi[1:] - coefficient * psi[:-1]
    centered = residuals - residuals.mean()
    rho = float(
        np.clip(
            np.dot(centered[1:], centered[:-1]) / np.dot(centered[:-1], centered[:-1]), -0.97, 0.97
        )
    )
    bandwidth = 1.1447 * (4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2) * len(residuals)) ** (
        1 / 3
    )
    count = len(residuals)
    lags = np.arange(1, count)
    lags = lags[lags < bandwidth]
    covariances = np.array([np.dot(residuals[lag:], residuals[:-lag]) / count for lag in lags])
    weights = 1 - lags / bandwidth
    long_run = (np.dot(residuals, residuals) / count + 2 * np.dot(weights, covariances)) / (
        1 - coefficient
    ) ** 2
    return float(long_run / np.mean(psi**2))


@pytest.mark.parametrize("phi", _PHIS)
def test_ljung_box_matches_scipy(phi: float) -> None:
    rng = np.random.default_rng(int(abs(phi) * 100) + 3)
    values = _ar1(rng, phi, 300, 0.0)

    statistic, p_value = ljung_box(values.tolist(), 10)
    expected_statistic, expected_p = _numpy_ljung_box(values, 10)

    assert statistic == pytest.approx(expected_statistic, rel=1e-10)
    assert p_value == pytest.approx(expected_p, rel=1e-8, abs=1e-300)


@pytest.mark.parametrize("phi", _PHIS)
def test_variance_ratio_matches_numpy(phi: float) -> None:
    rng = np.random.default_rng(int(abs(phi) * 100) + 17)
    values = _ar1(rng, phi, 900, 0.07) * 0.01

    assert sharpe_variance_ratio(values.tolist()) == pytest.approx(
        _numpy_variance_ratio(values), rel=1e-9
    )


def test_iid_variance_ratio_is_near_one_and_lag_zero_is_the_mertens_term() -> None:
    """Without serial correlation the long-run variance is the Mertens term itself."""

    rng = np.random.default_rng(2)
    values = rng.standard_t(5, size=4000) * 0.01 + 0.001

    mean, sigma = values.mean(), values.std()
    sharpe = mean / sigma
    skew = stats.skew(values)
    kurtosis = stats.kurtosis(values, fisher=False)
    psi = (values - mean) / sigma - mean / (2 * sigma**3) * ((values - mean) ** 2 - sigma**2)

    assert np.mean(psi**2) == pytest.approx(
        1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2, rel=1e-10
    )
    assert sharpe_variance_ratio(values.tolist()) == pytest.approx(1.0, abs=0.08)


def test_warning_survives_autocorrelations_with_no_valid_annualization(tmp_path: Path) -> None:
    """A short, strongly negative series can make Lo's truncated sum impossible.

    lo_annualized_sharpe() refuses such a series; the gate report must still
    come back, with the warning and without the annualized figure.
    """

    candidate = [0.28, -1.89, -0.02, -1.15, -0.18, -0.42, 0.07, -0.78]
    with pytest.raises(ValidationInputError, match="non-positive variance"):
        lo_annualized_sharpe(candidate, periods_per_year=12)
    registry = tmp_path / "registry.jsonl"
    other = [0.5, -0.3, 0.2, 0.1, -0.4, 0.3, 0.0, -0.1]
    for config, returns in enumerate((candidate, other)):
        append_trial(
            registry,
            recorded_at=_NOW,
            code_version="test",
            strategy_id="short",
            parameters={"config": str(config)},
            universe=("SYNTH",),
            data_start=_START,
            data_end=_END,
            cost_assumptions={"round_trip_bps": "10"},
            metrics={},
            operator_note=f"config {config}",
            period_returns=returns,
        )

    report = evaluate_registry(
        registry,
        candidate_trial_id=1,
        periods_per_year=12,
        effective_trials=2,
        block_count=2,
        min_observations=1,
    )
    dsr = next(check for check in report.checks if check.gate.startswith("DSR"))

    assert "WARNING returns are serially correlated" in dsr.detail
    assert "eta(q)" not in dsr.detail
