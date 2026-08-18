"""Whether a strategy's own volatility can survive its stop-loss line.

Gate 7. Gates 2-4 ask whether an edge is real. This one asks something
independent of skill, and usually never asked at all: running at the
volatility this strategy actually ran at, how often does an account following
it hit its own hard stop before the edge has time to pay?

The number is not intuitive in either direction, which is the point of
computing it rather than arguing about it. A *genuine* annualized Sharpe of
1.0 run at 15% annualized volatility breaches a "stop at -15%" line inside one
year 8.1% of the time; the same edge run at Sharpe 0.5 breaches it 16.7% of
the time. Nothing is wrong with the signal in the second case; the position
size and the stop line were specified by different people on different days
and never checked against each other. The shutdown then looks exactly like the
strategy being wrong, so the post-mortem blames the signal and the real defect
survives into the next strategy.

This gate is deliberately not a drawdown-from-peak statistic, and the
difference is not cosmetic: measured from a running peak instead of from
starting equity, that same Sharpe-1.0 strategy breaches 15% about 23% of the
time -- roughly three times the from-start answer. Quoting one number for the
other is the easiest way to be confidently wrong here. What this module
computes is the thing operators actually write into a risk policy: a loss
measured from starting equity, which is a first-passage problem with an exact
solution --

    P(min_{t<=T} X_t <= -L)
        = Phi((-L - mu T) / (sigma sqrt T))
          + exp(-2 mu L / sigma^2) * Phi((-L + mu T) / (sigma sqrt T))

-- rather than a simulation whose seed becomes another number to curate. Work
happens in log equity, where a multiplicative account is that arithmetic
process: "down 15%" is the level L = -ln(1 - 0.15), and the log drift is
mu - sigma^2/2.

The drawdown-from-peak counterpart is deliberately absent rather than
approximated. The obvious cheap route -- Lehoczky's expected first-passage
time for the drawdown process fed through a Poisson exceedance approximation
-- was measured against simulation before being rejected: it overstates by 3x
in the low-probability region a gate spends all its time in (0.037 predicted
against 0.012 simulated at Sharpe 1.0, 8% volatility). A gate is the wrong
place for a number that is wrong exactly where it matters.

Source: the first-passage identity for Brownian motion with drift (Karatzas &
Shreve, "Brownian Motion and Stochastic Calculus", 2nd ed., section 3.5).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

_MAX_BISECTION_VOLATILITY = 10.0
_BISECTION_STEPS = 200


class RiskInputError(ValueError):
    """Raised when a stop-line calculation receives unusable inputs."""


@dataclass(frozen=True, slots=True)
class StopLineResult:
    """Verdict on one strategy against one stop-loss policy.

    ``survivable_volatility`` is the actionable half: the annualized volatility
    at which this strategy's breach probability would equal the target. It is
    a position-sizing instruction, not a diagnosis.
    """

    breach_probability: float
    annual_return: float
    annual_volatility: float
    stop_loss_fraction: float
    horizon_years: float
    max_breach_probability: float
    survivable_volatility: float

    @property
    def passed(self) -> bool:
        return self.breach_probability <= self.max_breach_probability

    @property
    def volatility_scale(self) -> float:
        """Factor to multiply position size by to reach the survivable volatility."""

        if self.annual_volatility <= 0.0:
            return 1.0
        return self.survivable_volatility / self.annual_volatility


def stop_breach_probability(
    *,
    annual_return: float,
    annual_volatility: float,
    stop_loss_fraction: float,
    horizon_years: float,
) -> float:
    """Probability of ever being down ``stop_loss_fraction`` within the horizon.

    ``annual_return`` is the arithmetic annualized mean, ``annual_volatility``
    the annualized standard deviation, both of the same return series.
    ``stop_loss_fraction`` is a positive fraction of starting equity (0.15 for
    a -15% line), measured from the start rather than from a running peak,
    because that is how hard stops are written down.
    """

    _require_positive("annual_volatility", annual_volatility)
    _require_positive("horizon_years", horizon_years)
    if not 0.0 < stop_loss_fraction < 1.0:
        msg = f"stop_loss_fraction must be in (0, 1), got {stop_loss_fraction}"
        raise RiskInputError(msg)

    level = -math.log(1.0 - stop_loss_fraction)
    drift = annual_return - annual_volatility**2 / 2.0
    spread = annual_volatility * math.sqrt(horizon_years)
    normal = statistics.NormalDist()

    below = normal.cdf((-level - drift * horizon_years) / spread)
    tail = normal.cdf((-level + drift * horizon_years) / spread)
    if tail <= 0.0:
        reflected = 0.0
    else:
        # exp(-2 mu L / sigma^2) alone overflows for a losing strategy, where
        # the factor it multiplies is correspondingly tiny. Multiplying in log
        # space keeps the well-behaved product instead of inf * 0.
        reflected = math.exp(-2.0 * drift * level / annual_volatility**2 + math.log(tail))
    # The identity guarantees the sum is a probability; min() removes float
    # noise on that proof, and is not standing in for an unchecked assumption.
    return min(1.0, below + reflected)


def max_volatility_for_stop(
    *,
    sharpe_ratio: float,
    stop_loss_fraction: float,
    horizon_years: float,
    max_breach_probability: float,
) -> float:
    """The largest annualized volatility whose breach probability clears the target.

    Holding the Sharpe ratio fixed (annual return moves with volatility, which
    is what changing position size does), breach probability increases
    monotonically in volatility, so this bisects. The return value is the
    volatility budget the stop line implies -- size to it and the stop stops
    being a coin flip.
    """

    _require_positive("horizon_years", horizon_years)
    if not 0.0 < max_breach_probability < 1.0:
        msg = f"max_breach_probability must be in (0, 1), got {max_breach_probability}"
        raise RiskInputError(msg)

    def breach_at(volatility: float) -> float:
        return stop_breach_probability(
            annual_return=sharpe_ratio * volatility,
            annual_volatility=volatility,
            stop_loss_fraction=stop_loss_fraction,
            horizon_years=horizon_years,
        )

    high = _MAX_BISECTION_VOLATILITY
    if breach_at(high) <= max_breach_probability:
        # Even an absurd volatility clears the target; the stop line is not the
        # binding constraint here and reporting a cap would invent one.
        return high

    low = 0.0
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        if middle <= 0.0:
            break
        if breach_at(middle) > max_breach_probability:
            high = middle
        else:
            low = middle
    return low


def assess_stop_line(
    returns: list[float],
    *,
    periods_per_year: int,
    stop_loss_fraction: float,
    horizon_years: float = 1.0,
    max_breach_probability: float = 0.10,
) -> StopLineResult:
    """Judge a registered return series against a stop-loss policy.

    Both moments come from ``returns`` -- the series already in the registry --
    so the only operator inputs are the risk policy itself, which no library
    can derive without inventing someone's risk appetite.
    """

    if len(returns) < 2:
        msg = "returns must contain at least 2 observations"
        raise RiskInputError(msg)
    if periods_per_year < 1:
        msg = "periods_per_year must be at least 1"
        raise RiskInputError(msg)

    annual_volatility = statistics.stdev(returns) * math.sqrt(periods_per_year)
    if annual_volatility <= 0.0:
        msg = "returns must not be constant"
        raise RiskInputError(msg)
    annual_return = statistics.fmean(returns) * periods_per_year

    return StopLineResult(
        breach_probability=stop_breach_probability(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            stop_loss_fraction=stop_loss_fraction,
            horizon_years=horizon_years,
        ),
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        stop_loss_fraction=stop_loss_fraction,
        horizon_years=horizon_years,
        max_breach_probability=max_breach_probability,
        survivable_volatility=max_volatility_for_stop(
            sharpe_ratio=annual_return / annual_volatility,
            stop_loss_fraction=stop_loss_fraction,
            horizon_years=horizon_years,
            max_breach_probability=max_breach_probability,
        ),
    )


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        msg = f"{name} must be positive, got {value}"
        raise RiskInputError(msg)
