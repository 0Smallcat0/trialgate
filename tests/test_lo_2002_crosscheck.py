"""eta(q) and the Sharpe columns against the numbers printed in Lo (2002).

Source: Lo, "The Statistics of Sharpe Ratios", Financial Analysts Journal
58(4), July/August 2002, pp. 36-52. Page numbers are the journal's.

What these pin:

- Table 2 (p. 41): all 200 AR(1) scale factors eta(q), rho from -90% to 90%
  and q from 2 to 250, to the two decimals printed, from both the closed form
  Eq. (22) and the general Eq. (20) fed rho_k = rho^k.
- The AR(1) figures in the text (p. 41): 3.46 for IID, 2.88 at rho = 20%,
  4.17 at rho = -20%, all at q = 12.
- Table 4 (p. 44), all 22 funds: the monthly SR column follows from mu-hat,
  sigma-hat and Rf = 5/12% per month by Eqs. (3b)-(4), and the sqrt(12) SR
  column from that by Eq. (17), within the rounding of the printed figures.
  That holds only if the printed sigma-hat divides by T - 1 while SR uses the
  1/T estimator of Eq. (3b): taking the printed sigma-hat as the Eq. (3b)
  value leaves Janus Worldwide and Fund of funds outside rounding.
- For the 12 hedge funds the text's 45% and 65% overstatements and the
  robust rank order (pp. 44-45) follow from the transcribed columns.

What they cannot pin: the SR(12) column. Lo computes it by Eq. (20) from the
first q - 1 = 11 sample autocorrelations of each fund ("The General Case",
p. 42), and Table 4 prints only rho_1..rho_3; the returns are not public
(the funds are anonymized, note 11, p. 51). Fed only the printed three lags,
Eq. (20) misses SR(12) by up to 0.63, so the last test records that the
column is out of reach rather than fitting it.
"""

from __future__ import annotations

import math

import pytest

from trialgate.autocorrelation import ar1_eta, eta

_TABLE_2_PERIODS = (2, 3, 4, 6, 12, 24, 36, 48, 125, 250)

# rho (%) -> eta(q) for each q in _TABLE_2_PERIODS, transcribed from Table 2, p. 41.
_TABLE_2 = {
    90: (1.03, 1.05, 1.07, 1.10, 1.21, 1.41, 1.60, 1.77, 2.67, 3.70),
    80: (1.05, 1.10, 1.14, 1.21, 1.43, 1.81, 2.14, 2.42, 3.79, 5.32),
    70: (1.08, 1.15, 1.21, 1.33, 1.65, 2.19, 2.62, 3.00, 4.75, 6.68),
    60: (1.12, 1.21, 1.30, 1.46, 1.89, 2.55, 3.08, 3.53, 5.63, 7.94),
    50: (1.15, 1.28, 1.39, 1.60, 2.12, 2.91, 3.53, 4.06, 6.49, 9.15),
    40: (1.20, 1.35, 1.49, 1.75, 2.36, 3.27, 3.98, 4.58, 7.35, 10.37),
    30: (1.24, 1.43, 1.60, 1.91, 2.61, 3.65, 4.44, 5.12, 8.23, 11.62),
    20: (1.29, 1.52, 1.73, 2.07, 2.88, 4.04, 4.93, 5.68, 9.14, 12.92),
    10: (1.35, 1.62, 1.86, 2.25, 3.16, 4.45, 5.44, 6.28, 10.12, 14.31),
    0: (1.41, 1.73, 2.00, 2.45, 3.46, 4.90, 6.00, 6.93, 11.18, 15.81),
    -10: (1.49, 1.85, 2.16, 2.66, 3.80, 5.39, 6.61, 7.64, 12.35, 17.47),
    -20: (1.58, 1.99, 2.33, 2.90, 4.17, 5.95, 7.31, 8.45, 13.67, 19.35),
    -30: (1.69, 2.13, 2.53, 3.17, 4.60, 6.59, 8.10, 9.38, 15.20, 21.52),
    -40: (1.83, 2.29, 2.75, 3.48, 5.09, 7.34, 9.05, 10.48, 17.01, 24.11),
    -50: (2.00, 2.45, 3.02, 3.84, 5.69, 8.26, 10.21, 11.84, 19.26, 27.31),
    -60: (2.24, 2.61, 3.37, 4.30, 6.44, 9.44, 11.70, 13.59, 22.19, 31.50),
    -70: (2.58, 2.76, 3.86, 4.92, 7.45, 11.05, 13.77, 16.04, 26.33, 37.43),
    -80: (3.16, 2.89, 4.66, 5.91, 8.96, 13.50, 16.98, 19.88, 32.96, 47.02),
    -90: (4.47, 2.97, 6.47, 8.09, 12.06, 18.29, 23.32, 27.61, 46.99, 67.65),
}

# Table 4, p. 44, in the order printed: name, T, mu-hat %, sigma-hat %,
# rho_1..rho_3 %, SR, sqrt(12) SR, SR(12).
_Row = tuple[str, int, float, float, tuple[float, float, float], float, float, float]

_TABLE_4_MUTUAL_FUNDS: tuple[_Row, ...] = (
    ("Vanguard 500 Index", 286, 1.30, 4.27, (-4.0, -6.6, -4.9), 0.21, 0.72, 0.85),
    ("Fidelity Magellan", 402, 1.73, 6.23, (12.4, -2.3, -0.4), 0.21, 0.73, 0.66),
    ("Investment Company of America", 450, 1.17, 4.01, (1.8, -3.2, -4.5), 0.19, 0.65, 0.71),
    ("Janus", 364, 1.52, 4.75, (10.5, -0.0, -3.7), 0.23, 0.81, 0.80),
    ("Fidelity Contrafund", 397, 1.29, 4.97, (7.4, -2.5, -6.8), 0.18, 0.61, 0.67),
    ("Washington Mutual Investors", 450, 1.13, 4.09, (-0.1, -7.2, -2.6), 0.17, 0.60, 0.65),
    ("Janus Worldwide", 102, 1.81, 4.36, (11.4, 3.4, -3.8), 0.32, 1.12, 1.29),
    ("Fidelity Growth and Income", 174, 1.54, 4.13, (5.1, -1.6, -8.2), 0.27, 0.95, 1.18),
    ("American Century Ultra", 223, 1.72, 7.11, (2.3, 3.4, 1.4), 0.18, 0.64, 0.71),
    ("Growth Fund of America", 431, 1.18, 5.35, (8.5, -2.7, -4.1), 0.14, 0.50, 0.49),
)

# Hedge funds, printed by sqrt(12) SR descending.
_TABLE_4_HEDGE_FUNDS: tuple[_Row, ...] = (
    ("Convertible/option arbitrage", 104, 1.63, 0.97, (42.6, 29.0, 21.4), 1.26, 4.35, 2.99),
    ("Relative value", 97, 0.66, 0.21, (25.9, 19.2, -2.1), 1.17, 4.06, 3.38),
    ("Mortgage-backed securities", 96, 1.33, 0.79, (42.0, 22.1, 16.7), 1.16, 4.03, 2.44),
    ("High-yield debt", 79, 1.30, 0.87, (33.7, 21.8, 13.1), 1.02, 3.54, 2.25),
    ("Risk arbitrage A", 90, 1.06, 0.69, (-4.9, -10.8, 6.9), 0.94, 3.25, 3.83),
    ("Long-short equities", 138, 1.18, 0.83, (-20.2, 24.6, 8.7), 0.92, 3.19, 2.32),
    ("Multistrategy A", 72, 1.08, 0.75, (48.9, 23.4, 3.3), 0.89, 3.09, 2.18),
    ("Risk arbitrage B", 74, 0.90, 0.77, (-4.9, 2.5, -8.3), 0.63, 2.17, 2.47),
    ("Convertible arbitrage A", 100, 1.38, 1.60, (33.8, 30.8, 7.9), 0.60, 2.08, 1.43),
    ("Convertible arbitrage B", 78, 0.78, 0.62, (32.4, 9.7, -4.5), 0.60, 2.06, 1.67),
    ("Multistrategy B", 139, 1.34, 1.63, (49.0, 24.6, 10.6), 0.57, 1.96, 1.17),
    ("Fund of funds", 75, 1.68, 2.29, (29.7, 21.1, 0.9), 0.56, 1.93, 1.39),
)

_MONTHLY_RISK_FREE = 5.0 / 12.0  # percent per month, Table 4 note
_HALF_CENT = 0.005  # every Table 4 figure is printed to two decimals
_SLACK = 1e-12


def _round_half_up(value: float) -> float:
    return math.floor(value * 100.0 + 0.5) / 100.0


@pytest.mark.parametrize("rho_percent", sorted(_TABLE_2))
def test_eta_reproduces_lo_table_2(rho_percent: int) -> None:
    rho = rho_percent / 100.0
    for periods, printed in zip(_TABLE_2_PERIODS, _TABLE_2[rho_percent], strict=True):
        general = eta(periods, [rho**lag for lag in range(1, periods)])
        assert _round_half_up(general) == pytest.approx(printed, abs=1e-9), (periods, general)
        if rho_percent:
            assert ar1_eta(periods, rho) == pytest.approx(general, rel=1e-10)


@pytest.mark.parametrize(("rho", "printed"), [(0.0, 3.46), (0.2, 2.88), (-0.2, 4.17)])
def test_eta_reproduces_the_ar1_figures_in_the_text(rho: float, printed: float) -> None:
    assert _round_half_up(eta(12, [rho**lag for lag in range(1, 12)])) == printed


def _monthly_sharpe_bounds(row: _Row, *, sample_stdev: bool = True) -> tuple[float, float]:
    """Every monthly SR consistent with the printed mu, sigma, SR and sqrt(12) SR.

    With ``sample_stdev`` the printed sigma-hat is read as the T - 1 estimate
    and rescaled to the 1/T estimate of Eq. (3b) before Eq. (4).
    """

    _, count, mean, stdev, _, sharpe, iid_annual, _ = row
    rescale = math.sqrt(count / (count - 1.0)) if sample_stdev else 1.0
    low = max(
        (mean - _HALF_CENT - _MONTHLY_RISK_FREE) / (stdev + _HALF_CENT) * rescale,
        sharpe - _HALF_CENT,
        (iid_annual - _HALF_CENT) / math.sqrt(12.0),
    )
    high = min(
        (mean + _HALF_CENT - _MONTHLY_RISK_FREE) / (stdev - _HALF_CENT) * rescale,
        sharpe + _HALF_CENT,
        (iid_annual + _HALF_CENT) / math.sqrt(12.0),
    )
    return low, high


@pytest.mark.parametrize(
    "row", _TABLE_4_MUTUAL_FUNDS + _TABLE_4_HEDGE_FUNDS, ids=lambda row: row[0]
)
def test_table_4_sharpe_columns_follow_from_mean_and_volatility(row: _Row) -> None:
    low, high = _monthly_sharpe_bounds(row)

    assert low <= high + _SLACK


def test_table_4_sigma_column_is_not_the_eq_3b_estimate() -> None:
    inconsistent = []
    for row in _TABLE_4_MUTUAL_FUNDS + _TABLE_4_HEDGE_FUNDS:
        low, high = _monthly_sharpe_bounds(row, sample_stdev=False)
        if low > high + _SLACK:
            inconsistent.append(row[0])

    assert inconsistent == ["Janus Worldwide", "Fund of funds"]


def test_table_4_transcription_matches_the_text() -> None:
    annual = {name: robust for name, *_, robust in _TABLE_4_HEDGE_FUNDS}
    iid = {name: naive for name, *_, naive, _ in _TABLE_4_HEDGE_FUNDS}

    # p. 44: "overstates the annual Sharpe ratio by 45 percent"; p. 45: "65 percent".
    assert (
        round(iid["Convertible/option arbitrage"] / annual["Convertible/option arbitrage"] - 1, 2)
        == 0.45
    )
    assert (
        round(iid["Mortgage-backed securities"] / annual["Mortgage-backed securities"] - 1, 2)
        == 0.65
    )
    # p. 45: "the rank ordering changes to 3, 2, 5, 7, 1, 6, 8, 4, 10, 9, 12, and 11".
    by_robust = sorted(annual, key=annual.__getitem__, reverse=True)
    assert [by_robust.index(name) + 1 for name in annual] == [3, 2, 5, 7, 1, 6, 8, 4, 10, 9, 12, 11]


def test_table_4_annual_sharpe_is_out_of_reach_from_the_printed_lags() -> None:
    """SR(12) needs rho_4..rho_11, which Table 4 does not print.

    Inverting Eq. (20) on the printed SR(12) / SR gives the full weighted sum
    sum_{k=1}^{11} (12 - k) rho_k. Whatever lags 4..11 contribute must be a
    possible sum of autocorrelations, |rest| <= sum_{k=4}^{11} (12 - k) = 36,
    and for most funds that rest is far from zero: eta(12) on the three
    printed lags alone does not land on the printed column.
    """

    missed = []
    for row in _TABLE_4_HEDGE_FUNDS:
        name, _, _, _, rhos, _, _, robust = row
        low, high = _monthly_sharpe_bounds(row)
        eta_low = (robust - _HALF_CENT) / high
        eta_high = (robust + _HALF_CENT) / low
        printed_sum = sum((12 - lag) * rho / 100.0 for lag, rho in enumerate(rhos, 1))
        for implied_eta in (eta_low, eta_high):
            rest = ((12.0 / implied_eta) ** 2 - 12.0) / 2.0 - printed_sum
            assert abs(rest) <= 36.0, name
        three_lags = eta(12, [rho / 100.0 for rho in rhos])
        if not eta_low <= three_lags <= eta_high:
            missed.append(name)

    assert len(missed) >= 10
    assert "Convertible/option arbitrage" in missed
    assert "Long-short equities" in missed
