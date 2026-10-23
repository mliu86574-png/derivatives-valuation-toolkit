"""Plain-vanilla interest rate swap valuation on a single curve.

Simplifications, stated up front:
* one curve is used for both projection and discounting (pre-2008 style);
* the input par swap rates are for annual-pay swaps at whole-year tenors;
* accrual fractions are exactly 1 / frequency (no day-count or calendars).
These keep the mechanics transparent; a production system would add
OIS discounting, day counts, holiday calendars and interpolation choices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bootstrap_discount_factors(par_rates: dict[int, float]) -> dict[int, float]:
    """Bootstrap annual discount factors from par swap rates.

    par_rates maps tenor in years (1, 2, 3, ...) to the annual par swap rate.
    Tenors must be consecutive from 1.
    """
    tenors = sorted(par_rates)
    if tenors != list(range(1, len(tenors) + 1)):
        raise ValueError("tenors must be consecutive whole years starting at 1")
    dfs: dict[int, float] = {}
    annuity = 0.0
    for n in tenors:
        c = par_rates[n]
        dfs[n] = (1 - c * annuity) / (1 + c)
        annuity += dfs[n]
    return dfs


def discount_factor(dfs: dict[int, float], t: float) -> float:
    """Log-linear interpolation of discount factors; DF(0) = 1."""
    times = np.array([0.0] + sorted(dfs))
    logs = np.log(np.array([1.0] + [dfs[k] for k in sorted(dfs)]))
    if t < 0 or t > times[-1]:
        raise ValueError("t outside the bootstrapped curve")
    return float(np.exp(np.interp(t, times, logs)))


def zero_rate(dfs: dict[int, float], t: float, compounding: str = "continuous") -> float:
    """Zero rate at time t, continuously or annually compounded."""
    if t <= 0:
        raise ValueError("t must be positive")
    df = discount_factor(dfs, t)
    if compounding == "continuous":
        return -np.log(df) / t
    if compounding == "annual":
        return df ** (-1 / t) - 1
    raise ValueError("compounding must be 'continuous' or 'annual'")


@dataclass(frozen=True)
class SwapValuation:
    pv_fixed: float
    pv_float: float
    value: float      # to the holder: receive-fixed = pv_fixed - pv_float
    par_rate: float
    dv01: float       # change in value for a 1bp parallel rise in par rates


def value_swap(notional: float, fixed_rate: float, maturity: int,
               par_rates: dict[int, float], receive_fixed: bool = True,
               freq: int = 1) -> SwapValuation:
    """Value a vanilla swap that starts today, on the curve bootstrapped from par_rates."""
    def _value(curve_rates: dict[int, float]) -> tuple[float, float, float]:
        dfs = bootstrap_discount_factors(curve_rates)
        times = np.arange(1, maturity * freq + 1) / freq
        df = np.array([discount_factor(dfs, t) for t in times])
        annuity = df.sum() / freq
        pv_fixed = notional * fixed_rate * annuity
        pv_float = notional * (1 - df[-1])        # floating leg worth par at reset
        par = (1 - df[-1]) / annuity
        return pv_fixed, pv_float, par

    pv_fixed, pv_float, par = _value(par_rates)
    sign = 1 if receive_fixed else -1
    value = sign * (pv_fixed - pv_float)
    bumped = {k: v + 0.0001 for k, v in par_rates.items()}
    b_fixed, b_float, _ = _value(bumped)
    dv01 = sign * (b_fixed - b_float) - value
    return SwapValuation(pv_fixed, pv_float, value, par, dv01)
