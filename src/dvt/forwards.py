"""Forwards and futures: cost-of-carry pricing, basis and FX forwards.

All rates are continuously compounded unless the function says otherwise.
"""
from __future__ import annotations

from math import exp


def forward_price(spot: float, T: float, r: float, carry_yield: float = 0.0,
                  storage_cost: float = 0.0) -> float:
    """Cost-of-carry forward price  F = S * exp((r + u - y) * T).

    carry_yield   dividend yield for an equity index, or lease rate /
                  convenience yield for a commodity such as gold
    storage_cost  proportional storage and insurance cost per year
    """
    if spot <= 0 or T < 0:
        raise ValueError("spot must be positive and T non-negative")
    return spot * exp((r + storage_cost - carry_yield) * T)


def basis(spot: float, futures: float) -> float:
    """Basis defined as futures minus spot (positive = contango)."""
    return futures - spot


def implied_carry(spot: float, futures: float, T: float) -> float:
    """Annualised net carry implied by a futures price: ln(F / S) / T."""
    if spot <= 0 or futures <= 0 or T <= 0:
        raise ValueError("prices and T must be positive")
    from math import log
    return log(futures / spot) / T


def forward_value(forward_contract_price: float, current_forward: float,
                  T: float, r: float, long: bool = True) -> float:
    """Mark-to-market of an existing forward: (F_now - K) * exp(-rT) for a long."""
    value = (current_forward - forward_contract_price) * exp(-r * T)
    return value if long else -value


def fx_forward(spot: float, T: float, r_quote: float, r_base: float,
               simple: bool = False) -> float:
    """FX forward by covered interest parity.

    spot is quoted as units of quote currency per one unit of base currency,
    e.g. USD/CNH = 7.10 means base USD, quote CNH.

    simple=False  continuously compounded rates: F = S * exp((r_quote - r_base) T)
    simple=True   money-market (simple) rates:   F = S * (1 + r_quote T) / (1 + r_base T)
    """
    if spot <= 0 or T < 0:
        raise ValueError("spot must be positive and T non-negative")
    if simple:
        return spot * (1 + r_quote * T) / (1 + r_base * T)
    return spot * exp((r_quote - r_base) * T)


def forward_points(spot: float, forward: float, pip: float = 1e-4) -> float:
    """Forward points in pips: (F - S) / pip."""
    return (forward - spot) / pip


def hedge_outcome(exposure_base: float, spot_at_maturity: float,
                  forward_rate: float, hedge_ratio: float = 1.0) -> dict:
    """Compare a hedged and an unhedged conversion of a foreign-currency exposure.

    exposure_base     amount of base currency to be converted at maturity
                      (positive = receivable to sell, e.g. USD dividends to sell for CNH)
    spot_at_maturity  spot rate realised at maturity
    forward_rate      rate locked with the forward
    hedge_ratio       share of the exposure covered by the forward (0 to 1)
    """
    if not 0 <= hedge_ratio <= 1:
        raise ValueError("hedge_ratio must be between 0 and 1")
    unhedged = exposure_base * spot_at_maturity
    hedged = (exposure_base * hedge_ratio * forward_rate
              + exposure_base * (1 - hedge_ratio) * spot_at_maturity)
    return {"unhedged": unhedged, "hedged": hedged, "hedge_pnl": hedged - unhedged}
