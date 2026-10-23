from math import exp

import pytest

import dvt


# Textbook reference: Hull, "Options, Futures, and Other Derivatives", Example 15.6
# S=42, K=40, r=10%, sigma=20%, T=0.5  ->  call ~ 4.76, put ~ 0.81
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


def test_black_scholes_textbook_values():
    assert dvt.black_scholes(**HULL, kind="call") == pytest.approx(4.76, abs=0.01)
    assert dvt.black_scholes(**HULL, kind="put") == pytest.approx(0.81, abs=0.01)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    S, K, T, r, sigma = 100, 95, 0.75, 0.04, 0.25
    c = dvt.black_scholes(S, K, T, r, sigma, "call", q)
    p = dvt.black_scholes(S, K, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * exp(-q * T) - K * exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, sigma, q = 100, 105, 0.5, 0.03, 0.22, 0.01
    g = dvt.greeks(S, K, T, r, sigma, kind, q)
    h = 1e-4
    px = lambda **kw: dvt.black_scholes(**{**dict(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind, q=q), **kw})
    assert g.delta == pytest.approx((px(S=S + h) - px(S=S - h)) / (2 * h), rel=1e-5)
    assert g.gamma == pytest.approx((px(S=S + h) - 2 * px() + px(S=S - h)) / h**2, rel=1e-3)
    assert g.vega == pytest.approx((px(sigma=sigma + h) - px(sigma=sigma - h)) / (2 * h), rel=1e-5)
    assert g.rho == pytest.approx((px(r=r + h) - px(r=r - h)) / (2 * h), rel=1e-5)
    # theta is the derivative with respect to calendar time, i.e. minus dV/dT
    assert g.theta == pytest.approx(-(px(T=T + h) - px(T=T - h)) / (2 * h), rel=1e-5)


def test_implied_volatility_round_trip():
    price = dvt.black_scholes(100, 110, 1.0, 0.02, 0.31, "call")
    assert dvt.implied_volatility(price, 100, 110, 1.0, 0.02, "call") == pytest.approx(0.31, abs=1e-8)


def test_implied_volatility_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        dvt.implied_volatility(0.5, 100, 50, 1.0, 0.02, "call")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_converges_to_black_scholes(kind):
    bs = dvt.black_scholes(**HULL, kind=kind)
    assert dvt.binomial_crr(**HULL, kind=kind, steps=2000) == pytest.approx(bs, abs=2e-3)


def test_american_put_worth_at_least_european():
    eu = dvt.binomial_crr(**HULL, kind="put", steps=500)
    am = dvt.binomial_crr(**HULL, kind="put", steps=500, american=True)
    assert am >= eu
    # American call on a non-dividend stock should equal the European call
    eu_c = dvt.binomial_crr(**HULL, kind="call", steps=500)
    am_c = dvt.binomial_crr(**HULL, kind="call", steps=500, american=True)
    assert am_c == pytest.approx(eu_c, abs=1e-10)


def test_monte_carlo_within_three_standard_errors():
    bs = dvt.black_scholes(**HULL, kind="call")
    mc, se = dvt.monte_carlo(**HULL, kind="call", paths=400_000, seed=7)
    assert abs(mc - bs) < 3 * se


def test_forward_price_and_implied_carry():
    F = dvt.forward_price(2000, 0.5, r=0.05, carry_yield=0.01, storage_cost=0.002)
    assert F == pytest.approx(2000 * exp((0.05 + 0.002 - 0.01) * 0.5))
    assert dvt.implied_carry(2000, F, 0.5) == pytest.approx(0.042)
    assert dvt.basis(2000, F) > 0


def test_forward_value_long_short_symmetry():
    long_v = dvt.forward_value(100, 104, 0.5, 0.03, long=True)
    short_v = dvt.forward_value(100, 104, 0.5, 0.03, long=False)
    assert long_v == pytest.approx(-short_v)
    assert long_v == pytest.approx(4 * exp(-0.015))


def test_fx_forward_interest_parity():
    # Higher quote-currency rate -> forward above spot (base currency at a premium)
    F = dvt.fx_forward(7.10, 1.0, r_quote=0.03, r_base=0.045, simple=True)
    assert F == pytest.approx(7.10 * 1.03 / 1.045)
    assert F < 7.10
    assert dvt.forward_points(7.10, F) == pytest.approx((F - 7.10) * 1e4)


def test_hedge_outcome():
    out = dvt.hedge_outcome(1_000_000, spot_at_maturity=7.00, forward_rate=7.15, hedge_ratio=0.8)
    assert out["unhedged"] == pytest.approx(7_000_000)
    assert out["hedged"] == pytest.approx(800_000 * 7.15 + 200_000 * 7.00)
    assert out["hedge_pnl"] == pytest.approx(120_000)


PAR = {1: 0.030, 2: 0.032, 3: 0.034, 4: 0.035, 5: 0.036}


def test_bootstrap_reprices_par_swaps():
    dfs = dvt.bootstrap_discount_factors(PAR)
    for n, c in PAR.items():
        annuity = sum(dfs[i] for i in range(1, n + 1))
        assert c * annuity + dfs[n] == pytest.approx(1.0, abs=1e-12)


def test_swap_at_par_rate_has_zero_value():
    v = dvt.value_swap(10_000_000, PAR[5], 5, PAR)
    assert v.value == pytest.approx(0.0, abs=1e-6)
    assert v.par_rate == pytest.approx(PAR[5], abs=1e-12)


def test_receive_fixed_dv01_is_negative():
    v = dvt.value_swap(10_000_000, 0.036, 5, PAR, receive_fixed=True)
    assert v.dv01 < 0
    payer = dvt.value_swap(10_000_000, 0.036, 5, PAR, receive_fixed=False)
    assert payer.dv01 == pytest.approx(-v.dv01)
    # Rough sanity: DV01 ~ notional * annuity * 1bp (~4.5 years of annuity)
    assert 3_500 < abs(v.dv01) < 5_000


def test_bootstrap_requires_consecutive_tenors():
    with pytest.raises(ValueError):
        dvt.bootstrap_discount_factors({1: 0.03, 3: 0.034})


def test_zero_rates_above_par_on_upward_curve():
    dfs = dvt.bootstrap_discount_factors(PAR)
    for n in range(2, 6):
        assert dvt.zero_rate(dfs, n, "annual") > PAR[n]
    assert dvt.zero_rate(dfs, 1, "annual") == pytest.approx(PAR[1])
