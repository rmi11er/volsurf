"""Tests for the volatility surface fitting module."""

import math
from datetime import date

import numpy as np
import pytest

from volsurf.fitting.arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_density_positivity,
    validate_surface,
    validate_term_structure,
)
from volsurf.fitting.implied_vol import (
    black_scholes_price,
    black_scholes_vega,
    calculate_forward_from_options,
    calculate_iv_bisection,
    calculate_iv_from_mid_prices,
    calculate_iv_newton,
)
from volsurf.fitting.svi import (
    check_svi_constraints,
    compute_vega_weights,
    estimate_forward_price,
    fit_svi_slice,
    svi_implied_vol,
    svi_total_variance,
)
from volsurf.models.schemas import SVIParams


class TestBlackScholes:
    """Tests for Black-Scholes calculations."""

    def test_call_price_atm(self):
        """ATM call should have positive price."""
        price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=True)
        assert price > 0
        # ATM 1-year call with 20% vol should be around 10
        assert 8 < price < 12

    def test_put_price_atm(self):
        """ATM put should have positive price."""
        price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=False)
        assert price > 0
        # ATM 1-year put should be slightly less than call due to r > 0
        assert 5 < price < 10

    def test_put_call_parity(self):
        """Put-call parity should hold."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = black_scholes_price(S, K, T, r, sigma, is_call=True)
        put = black_scholes_price(S, K, T, r, sigma, is_call=False)

        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_vega_positive(self):
        """Vega should be positive."""
        vega = black_scholes_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert vega > 0

    def test_vega_highest_atm(self):
        """Vega should be highest ATM."""
        vega_atm = black_scholes_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        vega_otm = black_scholes_vega(S=100, K=120, T=1.0, r=0.05, sigma=0.2)
        vega_itm = black_scholes_vega(S=100, K=80, T=1.0, r=0.05, sigma=0.2)

        assert vega_atm > vega_otm
        assert vega_atm > vega_itm

    def test_zero_tte_returns_intrinsic(self):
        """Zero TTE should return intrinsic value."""
        # ITM call
        call_price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.2, is_call=True)
        assert call_price == 10

        # OTM call
        call_price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.2, is_call=True)
        assert call_price == 0

        # ITM put
        put_price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.2, is_call=False)
        assert put_price == 10


class TestImpliedVolatility:
    """Tests for implied volatility calculation."""

    def test_iv_roundtrip_bisection(self):
        """IV calculation should round-trip correctly."""
        S, K, T, r = 100, 100, 1.0, 0.05
        true_sigma = 0.25

        price = black_scholes_price(S, K, T, r, true_sigma, is_call=True)
        iv = calculate_iv_bisection(price, S, K, T, r, is_call=True)

        assert iv is not None
        assert abs(iv - true_sigma) < 1e-5

    def test_iv_roundtrip_newton(self):
        """Newton method should also round-trip correctly."""
        S, K, T, r = 100, 100, 1.0, 0.05
        true_sigma = 0.25

        price = black_scholes_price(S, K, T, r, true_sigma, is_call=True)
        iv = calculate_iv_newton(price, S, K, T, r, is_call=True)

        assert iv is not None
        assert abs(iv - true_sigma) < 1e-5

    def test_iv_otm_options(self):
        """IV should work for OTM options."""
        S, K, T, r = 100, 120, 0.5, 0.05
        true_sigma = 0.30

        # OTM call
        price = black_scholes_price(S, K, T, r, true_sigma, is_call=True)
        iv = calculate_iv_newton(price, S, K, T, r, is_call=True)

        assert iv is not None
        assert abs(iv - true_sigma) < 1e-4

    def test_iv_from_mid_prices(self):
        """Test batch IV calculation."""
        strikes = np.array([90, 100, 110])
        S, T, r = 100, 1.0, 0.05
        true_sigma = 0.20

        # Generate prices
        is_call = np.array([False, True, True])
        mid_prices = np.array([
            black_scholes_price(S, 90, T, r, true_sigma, False),
            black_scholes_price(S, 100, T, r, true_sigma, True),
            black_scholes_price(S, 110, T, r, true_sigma, True),
        ])

        ivs, valid = calculate_iv_from_mid_prices(
            strikes, mid_prices, is_call, S, T, r
        )

        assert all(valid)
        assert all(abs(ivs - true_sigma) < 0.01)

    def test_iv_returns_none_for_invalid(self):
        """IV should return None for invalid inputs."""
        assert calculate_iv_newton(0, 100, 100, 1.0, 0.05, True) is None
        assert calculate_iv_newton(-1, 100, 100, 1.0, 0.05, True) is None
        assert calculate_iv_newton(10, 100, 100, 0, 0.05, True) is None


class TestForwardPrice:
    """Tests for forward price calculation."""

    def test_forward_from_put_call_parity(self):
        """Forward should be calculated correctly from options."""
        S, r, T = 100, 0.05, 1.0
        sigma = 0.2

        # Generate synthetic call and put prices at various strikes
        strikes = np.array([90, 95, 100, 105, 110])
        calls = np.array([black_scholes_price(S, K, T, r, sigma, True) for K in strikes])
        puts = np.array([black_scholes_price(S, K, T, r, sigma, False) for K in strikes])

        forward = calculate_forward_from_options(calls, puts, strikes, S, r, T)

        # Should be close to S * exp(rT)
        expected = S * math.exp(r * T)
        assert abs(forward - expected) < 0.5

    def test_estimate_forward_price(self):
        """Test cost-of-carry forward estimate."""
        forward = estimate_forward_price(
            underlying_price=100,
            tte_years=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
        )

        # F = S * exp((r - q) * T)
        expected = 100 * math.exp((0.05 - 0.02) * 1.0)
        assert abs(forward - expected) < 1e-10


class TestSVI:
    """Tests for SVI model."""

    def test_svi_total_variance_atm(self):
        """SVI at ATM (k=0, m=0) should equal a + b*sigma."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        k = np.array([0.0])

        w = svi_total_variance(k, a, b, rho, m, sigma)
        expected = a + b * sigma

        assert abs(w[0] - expected) < 1e-10

    def test_svi_wings_increase(self):
        """SVI total variance should increase in the wings."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        k_grid = np.linspace(-0.5, 0.5, 21)

        w = svi_total_variance(k_grid, a, b, rho, m, sigma)

        # Find ATM (k=0) variance
        atm_idx = 10
        w_atm = w[atm_idx]

        # Wings should be higher
        assert w[0] > w_atm  # Left wing
        assert w[-1] > w_atm  # Right wing

    def test_svi_implied_vol(self):
        """SVI implied vol should be sqrt(w/T)."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        k = np.array([0.0])
        T = 1.0

        iv = svi_implied_vol(k, T, a, b, rho, m, sigma)
        w = svi_total_variance(k, a, b, rho, m, sigma)

        assert abs(iv[0] - math.sqrt(w[0] / T)) < 1e-10

    def test_svi_constraints_valid(self):
        """Valid SVI parameters should pass constraints."""
        is_valid, msg = check_svi_constraints(
            a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1
        )
        assert is_valid
        assert msg == "OK"

    def test_svi_constraints_negative_b(self):
        """Negative b should fail."""
        is_valid, msg = check_svi_constraints(
            a=0.04, b=-0.1, rho=-0.3, m=0.0, sigma=0.1
        )
        assert not is_valid
        assert "b" in msg

    def test_svi_constraints_rho_out_of_range(self):
        """Rho outside (-1, 1) should fail."""
        is_valid, msg = check_svi_constraints(
            a=0.04, b=0.1, rho=1.5, m=0.0, sigma=0.1
        )
        assert not is_valid
        assert "rho" in msg

    def test_svi_constraints_negative_sigma(self):
        """Negative sigma should fail."""
        is_valid, msg = check_svi_constraints(
            a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=-0.1
        )
        assert not is_valid
        assert "sigma" in msg


class TestSVIFitting:
    """Tests for SVI fitting."""

    def test_fit_synthetic_smile(self):
        """Fit SVI to synthetic data generated from known parameters."""
        # True parameters
        true_a, true_b, true_rho, true_m, true_sigma = 0.04, 0.12, -0.25, 0.0, 0.1
        T = 0.25  # 3 months
        F = 100.0

        # Generate synthetic smile
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        k = np.log(strikes / F)

        w = svi_total_variance(k, true_a, true_b, true_rho, true_m, true_sigma)
        ivs = np.sqrt(w / T)

        # Add small noise
        np.random.seed(42)
        ivs_noisy = ivs + np.random.normal(0, 0.001, len(ivs))

        # Fit
        result = fit_svi_slice(strikes, ivs_noisy, F, T)

        assert result.success
        assert result.rmse < 0.005  # Should be very close

        # Parameters should be close
        assert abs(result.params.a - true_a) < 0.02
        assert abs(result.params.b - true_b) < 0.05
        assert abs(result.params.rho - true_rho) < 0.1

    def test_fit_respects_constraints(self):
        """Fitted parameters should satisfy no-arbitrage constraints."""
        # Generate some realistic-looking data
        F = 100.0
        T = 0.5
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])

        # Typical equity smile: higher IV for lower strikes (skew)
        ivs = np.array([0.28, 0.25, 0.22, 0.20, 0.19, 0.185, 0.18, 0.18, 0.19])

        result = fit_svi_slice(strikes, ivs, F, T)

        # Should satisfy constraints
        is_valid, _ = check_svi_constraints(
            result.params.a,
            result.params.b,
            result.params.rho,
            result.params.m,
            result.params.sigma,
        )
        assert is_valid

    def test_vega_weights(self):
        """Vega weights should be highest ATM."""
        F = 100.0
        T = 0.5
        strikes = np.array([80, 90, 100, 110, 120])
        ivs = np.array([0.25, 0.22, 0.20, 0.19, 0.20])

        weights = compute_vega_weights(strikes, F, ivs, T)

        # ATM weight (index 2) should be highest
        assert weights[2] == max(weights)


class TestArbitrage:
    """Tests for arbitrage validation."""

    def test_butterfly_no_arbitrage(self):
        """Valid SVI should pass butterfly check."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        k_grid = np.linspace(-0.3, 0.3, 50)

        passes, violations, messages = check_butterfly_arbitrage(params, k_grid, 1.0)

        assert passes
        assert violations == 0

    def test_density_positive(self):
        """Valid SVI should have positive density."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        k_grid = np.linspace(-0.3, 0.3, 50)

        passes, violations, messages = check_density_positivity(params, k_grid, 1.0)

        assert passes
        assert violations == 0

    def test_calendar_no_arbitrage(self):
        """Variance should increase with time."""
        # Two surfaces with increasing variance
        params1 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        params2 = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        surfaces = [(0.25, params1), (0.5, params2)]
        k_grid = np.linspace(-0.3, 0.3, 50)

        passes, violations, messages = check_calendar_arbitrage(surfaces, k_grid)

        assert passes
        assert violations == 0

    def test_calendar_arbitrage_detected(self):
        """Calendar arbitrage should be detected."""
        # Second surface has lower variance (arbitrage!)
        params1 = SVIParams(a=0.06, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        params2 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        surfaces = [(0.25, params1), (0.5, params2)]
        k_grid = np.linspace(-0.3, 0.3, 50)

        passes, violations, messages = check_calendar_arbitrage(surfaces, k_grid)

        assert not passes
        assert violations > 0

    def test_validate_surface(self):
        """Full surface validation should work."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        result = validate_surface(params, tte_years=0.5)

        assert result.passes
        assert result.butterfly_violations == 0

    def test_validate_term_structure(self):
        """Term structure validation should work."""
        params1 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        params2 = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        params3 = SVIParams(a=0.06, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        surfaces = [(0.25, params1), (0.5, params2), (1.0, params3)]

        result = validate_term_structure(surfaces)

        assert result.passes
        assert result.calendar_violations == 0


class TestSVIParamsModel:
    """Tests for SVIParams Pydantic model."""

    def test_params_total_variance(self):
        """Test SVIParams.total_variance method."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        w = params.total_variance(0.0)
        expected = 0.04 + 0.1 * 0.1  # a + b*sigma at ATM

        assert abs(w - expected) < 1e-10

    def test_params_implied_vol(self):
        """Test SVIParams.implied_vol method."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        iv = params.implied_vol(0.0, 1.0)
        w = params.total_variance(0.0)

        assert abs(iv - math.sqrt(w)) < 1e-10

    def test_params_validation(self):
        """Test Pydantic validation."""
        # Valid params
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert params.b >= 0
        assert -1 <= params.rho <= 1
        assert params.sigma > 0

        # Invalid: negative b
        with pytest.raises(ValueError):
            SVIParams(a=0.04, b=-0.1, rho=-0.3, m=0.0, sigma=0.1)

        # Invalid: rho out of range
        with pytest.raises(ValueError):
            SVIParams(a=0.04, b=0.1, rho=1.5, m=0.0, sigma=0.1)

        # Invalid: non-positive sigma
        with pytest.raises(ValueError):
            SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0)
