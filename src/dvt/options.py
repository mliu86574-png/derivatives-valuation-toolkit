"""European and American option pricing.

Conventions
-----------
S      spot price of the underlying
K      strike price
T      time to expiry in years
r      continuously compounded risk-free rate
q      continuously compounded dividend / convenience yield (0 for none)
sigma  annualised volatility
kind   "call" or "put"
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _check_kind(kind: str) -> str:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return kind


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return d1, d1 - sigma * sqrt(T)


def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  kind: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option."""
    kind = _check_kind(kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float    # per 1.00 change in volatility (divide by 100 for per vol point)
    theta: float   # per year (divide by 365 for per calendar day)
    rho: float     # per 1.00 change in rate (divide by 100 for per 1%)


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           kind: str = "call", q: float = 0.0) -> Greeks:
    """Analytical Black-Scholes-Merton Greeks."""
    kind = _check_kind(kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf = norm.pdf(d1)
    disc_q, disc_r = exp(-q * T), exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * sqrt(T))
    vega = S * disc_q * pdf * sqrt(T)
    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrt(T))
                 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       kind: str = "call", q: float = 0.0,
                       low: float = 1e-6, high: float = 5.0) -> float:
    """Solve for the Black-Scholes volatility that reproduces a market price."""
    kind = _check_kind(kind)
    intrinsic = (max(S * exp(-q * T) - K * exp(-r * T), 0.0) if kind == "call"
                 else max(K * exp(-r * T) - S * exp(-q * T), 0.0))
    if price <= intrinsic:
        raise ValueError("price is at or below the no-arbitrage lower bound")
    return brentq(lambda v: black_scholes(S, K, T, r, v, kind, q) - price, low, high, xtol=1e-10)


def binomial_crr(S: float, K: float, T: float, r: float, sigma: float,
                 kind: str = "call", q: float = 0.0, steps: int = 500,
                 american: bool = False) -> float:
    """Cox-Ross-Rubinstein binomial tree; supports early exercise."""
    kind = _check_kind(kind)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1); increase steps")
    disc = exp(-r * dt)
    j = np.arange(steps + 1)
    prices = S * u ** (steps - j) * d ** j
    values = np.maximum(prices - K, 0) if kind == "call" else np.maximum(K - prices, 0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            prices = prices[:-1] / u
            exercise = prices - K if kind == "call" else K - prices
            values = np.maximum(values, exercise)
    return float(values[0])


def monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                kind: str = "call", q: float = 0.0, paths: int = 200_000,
                seed: int | None = 42) -> tuple[float, float]:
    """Monte Carlo price of a European option with antithetic variates.

    Returns (price, standard error).
    """
    kind = _check_kind(kind)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = exp(-r * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / sqrt(len(disc)))
