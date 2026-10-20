"""Serial-correlation corrections for the Sharpe ratio and its standard error.

The DSR/PSR denominator and the sqrt(p) annualization both assume returns are
independent from one period to the next. Positively autocorrelated returns
(smoothed marks, illiquid holdings, overlapping signals) break both in the
flattering direction: the annualized Sharpe is overstated and the standard
error understated, so the DSR rises for reasons that have nothing to do with
skill.

Sources (page numbers are the journal's for Lo, the authors' preprints for
Bailey & Lopez de Prado):
- Lo, "The Statistics of Sharpe Ratios", Financial Analysts Journal 58(4),
  2002, pp. 36-52: the time-aggregation factor eta(q), Eq. (20) p. 40, and
  its AR(1) closed form, Eq. (22) p. 41; the GMM variance of the Sharpe ratio
  under non-IID returns, Eq. (14) p. 39 with Appendix A Eqs. (A3) p. 46 and
  (A7), (A8), (A11) p. 47. Lo estimates the long-run covariance with
  Newey-West at a fixed truncation lag m, Eqs. (A12)-(A14) p. 47; this module
  uses the prewhitened Andrews-Monahan estimator below instead.
- Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (Journal of
  Risk 2012; preprint of July 2012): the IID non-normal variance of the
  Sharpe ratio, Eq. (8) p. 8, and the PSR, Eq. (11) p. 9. "The Deflated
  Sharpe Ratio" (JPM 2014; preprint of July 31, 2014): the DSR, Eq. (2) p. 8.
  Neither paper rescales that variance for serial correlation; doing so is
  this package's extension (see sharpe_variance_ratio).
- Andrews, "Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix Estimation" (Econometrica 1991): the Bartlett kernel's AR(1)
  plug-in bandwidth.
- Andrews & Monahan, "An Improved Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix Estimator" (Econometrica 1992): AR(1)
  prewhitening, without which a fixed-lag Newey-West estimate recovers about
  half the long-run variance of an AR(1) with phi = 0.8.
- Ljung & Box, "On a Measure of Lack of Fit in Time Series Models"
  (Biometrika 1978): the portmanteau test that decides when to warn.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

LJUNG_BOX_LAGS = 10
SIGNIFICANCE_LEVEL = 0.05
# Andrews & Monahan (1992) bound the prewhitening coefficient away from one so
# that the recoloring factor 1 / (1 - a)^2 cannot explode on a near unit root.
_MAX_PREWHITEN = 0.97


@dataclass(frozen=True, slots=True)
class AutocorrelationDiagnostics:
    """How far the returns are from the IID assumption, and what it costs.

    ``variance_ratio`` is the long-run variance of the Sharpe ratio estimator
    over its IID variance: above 1 the IID standard error is too small (the
    flattering case), below 1 it is too large. ``effective_observations`` is
    the IID sample size with the same standard error, ``1 + (T - 1) /
    variance_ratio``. ``significant`` is the Ljung-Box test at
    ``SIGNIFICANCE_LEVEL``; it decides whether a gate warns, not whether the
    adjustment is computed.
    """

    first_order_autocorrelation: float
    ljung_box_lags: int
    ljung_box_statistic: float
    ljung_box_p_value: float
    significant: bool
    variance_ratio: float
    effective_observations: float


def sample_autocorrelations(values: Sequence[float], max_lag: int) -> list[float]:
    """rho_1 .. rho_max_lag, each over the full-sample variance (Box-Jenkins)."""

    count = len(values)
    mean = statistics.fmean(values)
    centered = [value - mean for value in values]
    denominator = sum(value * value for value in centered)
    return [
        sum(centered[index] * centered[index - lag] for index in range(lag, count)) / denominator
        for lag in range(1, max_lag + 1)
    ]


def ljung_box(values: Sequence[float], lags: int) -> tuple[float, float]:
    """Ljung-Box Q over ``lags`` autocorrelations and its chi-square p-value.

    ``lags`` must be even: the chi-square survival function with an even
    number of degrees of freedom is a finite Poisson sum, so the p-value needs
    no incomplete-gamma routine and no dependency.
    """

    if lags < 2 or lags % 2:
        msg = f"lags must be an even integer of at least 2, got {lags}"
        raise ValueError(msg)
    count = len(values)
    rhos = sample_autocorrelations(values, lags)
    statistic = (
        count * (count + 2.0) * sum(rho * rho / (count - lag) for lag, rho in enumerate(rhos, 1))
    )
    half = statistic / 2.0
    term = math.exp(-half)
    total = term
    for index in range(1, lags // 2):
        term *= half / index
        total += term
    return statistic, min(1.0, total)


def ljung_box_lags_for(observations: int) -> int:
    """10 lags, or fewer (but at least 2, always even) on a short series."""

    return min(LJUNG_BOX_LAGS, max(2, 2 * (observations // 10)))


def andrews_bandwidth(values: Sequence[float]) -> float:
    """Andrews (1991) Bartlett bandwidth with an AR(1) plug-in.

    S_T = 1.1447 * (alpha(1) * T)^(1/3), alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2).
    """

    mean = statistics.fmean(values)
    rho = _ar1_coefficient([value - mean for value in values])
    rho = max(-_MAX_PREWHITEN, min(_MAX_PREWHITEN, rho))
    alpha = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    return 1.1447 * math.cbrt(alpha * len(values))


def long_run_variance(values: Sequence[float]) -> float:
    """Prewhitened Bartlett estimate of sum_k Cov(x_t, x_{t-k}) for a mean-zero series.

    Andrews & Monahan (1992): fit x_t = a x_{t-1} + e_t, estimate the long-run
    variance of e_t with the Bartlett kernel at the Andrews (1991) bandwidth,
    and recolor by 1 / (1 - a)^2. On an AR(1) the residual is white and the
    estimate is exact up to sampling error; without the prewhitening step the
    kernel's O(1/S) bias swallows most of a strongly persistent series.
    """

    coefficient = max(-_MAX_PREWHITEN, min(_MAX_PREWHITEN, _ar1_coefficient(values)))
    residuals = [values[index] - coefficient * values[index - 1] for index in range(1, len(values))]
    bandwidth = andrews_bandwidth(residuals)
    count = len(residuals)
    estimate = sum(value * value for value in residuals) / count
    lag = 1
    while lag < bandwidth and lag < count:
        weight = 1.0 - lag / bandwidth
        covariance = (
            sum(residuals[index] * residuals[index - lag] for index in range(lag, count)) / count
        )
        estimate += 2.0 * weight * covariance
        lag += 1
    return estimate / (1.0 - coefficient) ** 2


def sharpe_variance_ratio(values: Sequence[float]) -> float:
    """Long-run over IID variance of the Sharpe ratio estimator (Lo 2002, GMM case).

    The estimator SR = mu / sigma is a function of the moment vector
    (r_t - mu, (r_t - mu)^2 - sigma^2), Lo's Eq. (A8) p. 47. Its asymptotic
    variance is g' Omega g, Lo's Eq. (14) p. 39 and (A11) p. 47, with
    g = (1/sigma, -mu / (2 sigma^3)), the gradient of mu / sigma in
    (mu, sigma^2), Eq. (A3) p. 46 with Rf = 0, and Omega the moments'
    long-run covariance, Eq. (A7) p. 47. That is the long-run variance of the
    scalar influence series

        psi_t = (r_t - mu) / sigma - mu / (2 sigma^3) * ((r_t - mu)^2 - sigma^2).

    With IID returns Omega is the lag-0 covariance alone and g' Omega g is
    exactly the Mertens term 1 - skew * SR + (kurt - 1) / 4 * SR^2 used by the
    PSR/DSR (Bailey & Lopez de Prado 2012, Eq. (8) p. 8, written there as
    1 + SR^2 / 2 - skew * SR + (kurt - 3) / 4 * SR^2); serial correlation only
    adds the lag terms. The ratio of the two therefore rescales that term
    without touching its skew/kurtosis part.

    Using the ratio inside the DSR is this package's extension, not a method
    of either Bailey & Lopez de Prado paper: the DSR paper does not discuss
    serial correlation, and the PSR paper says it does not "explicitly
    address the case of serially-conditioned processes" (p. 4) and takes the
    Mertens variance to hold for stationary ergodic returns on the strength
    of Christie (2005) and Opdyke (2007) (pp. 4, 8). Lo's Eq. (14) carries
    the lag covariances that the Mertens term drops.
    """

    count = len(values)
    mean = statistics.fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / count
    sigma = math.sqrt(variance)
    slope = mean / (2.0 * sigma**3)
    influence = [
        (value - mean) / sigma - slope * ((value - mean) ** 2 - variance) for value in values
    ]
    iid = sum(value * value for value in influence) / count
    return long_run_variance(influence) / iid


def diagnose(values: Sequence[float]) -> AutocorrelationDiagnostics:
    """Autocorrelation diagnostics for a non-constant series of at least 3 values."""

    count = len(values)
    lags = ljung_box_lags_for(count)
    statistic, p_value = ljung_box(values, lags)
    ratio = sharpe_variance_ratio(values)
    return AutocorrelationDiagnostics(
        first_order_autocorrelation=sample_autocorrelations(values, 1)[0],
        ljung_box_lags=lags,
        ljung_box_statistic=statistic,
        ljung_box_p_value=p_value,
        significant=p_value < SIGNIFICANCE_LEVEL,
        variance_ratio=ratio,
        effective_observations=1.0 + (count - 1.0) / ratio,
    )


def eta(periods: int, autocorrelations: Sequence[float]) -> float:
    """Lo (2002) time-aggregation factor: SR(q) = eta(q) * SR, Eq. (20) p. 40.

    eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) rho_k), with rho_k taken
    as zero beyond ``len(autocorrelations)``. With every rho_k zero this is
    sqrt(q), the usual annualization.
    """

    lags = min(len(autocorrelations), periods - 1)
    radicand = periods + 2.0 * sum(
        (periods - lag) * autocorrelations[lag - 1] for lag in range(1, lags + 1)
    )
    if radicand <= 0.0:
        msg = (
            f"autocorrelations imply a non-positive variance for the {periods}-period sum "
            f"({radicand:.6f}); they are not a valid autocorrelation function"
        )
        raise ValueError(msg)
    return periods / math.sqrt(radicand)


def ar1_eta(periods: int, rho: float) -> float:
    """Lo (2002) closed form of eta(q) when rho_k = rho^k (an AR(1)), Eq. (22) p. 41.

    Lo writes it sqrt(q) * [1 + 2 rho / (1 - rho) * (1 - (1 - rho^q) / (q (1 - rho)))]^(-1/2);
    this is the same expression through
    sum_{k=1}^{q-1} (q - k) rho^k = rho / (1 - rho) * (q - (1 - rho^q) / (1 - rho)).
    Table 2 (p. 41) tabulates it; tests/test_lo_2002_crosscheck.py reproduces
    every cell.
    """

    return periods / math.sqrt(
        periods + 2.0 * rho / (1.0 - rho) * (periods - (1.0 - rho**periods) / (1.0 - rho))
    )


def _ar1_coefficient(values: Sequence[float]) -> float:
    count = len(values)
    denominator = sum(values[index - 1] ** 2 for index in range(1, count))
    if denominator == 0.0:
        return 0.0
    numerator = sum(values[index] * values[index - 1] for index in range(1, count))
    return numerator / denominator
