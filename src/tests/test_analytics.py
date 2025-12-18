"""Tests for analytics module."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from volsurf.analytics.realized_vol import (
    calculate_close_to_close_vol,
    calculate_parkinson_vol,
    calculate_garman_klass_vol,
    RealizedVolCalculator,
)
from volsurf.analytics.term_structure import (
    power_law,
    fit_power_law,
    TermStructureAnalyzer,
)
from volsurf.analytics.surface_metrics import (
    get_atm_vol_from_svi,
    get_vol_at_moneyness,
)


# =============================================================================
# Realized Volatility Tests
# =============================================================================


class TestCloseToCloseVol:
    """Tests for close-to-close realized volatility."""

    def test_zero_volatility(self):
        """Zero returns should give zero volatility."""
        returns = np.zeros(20)
        vol = calculate_close_to_close_vol(returns)
        assert vol == 0.0

    def test_known_volatility(self):
        """Test with known constant returns."""
        # If daily returns are constant, variance = return^2
        daily_return = 0.01  # 1%
        returns = np.full(252, daily_return)
        vol = calculate_close_to_close_vol(returns, annualization_factor=252)
        # Expected: sqrt(252 * 0.01^2) = sqrt(0.0252) ≈ 0.1587
        expected = np.sqrt(252 * 0.01**2)
        assert np.isclose(vol, expected, rtol=1e-6)

    def test_empty_returns(self):
        """Empty array should return NaN."""
        returns = np.array([])
        vol = calculate_close_to_close_vol(returns)
        assert np.isnan(vol)

    def test_annualization_factor(self):
        """Different annualization factors should scale correctly."""
        returns = np.random.randn(100) * 0.01
        vol_252 = calculate_close_to_close_vol(returns, annualization_factor=252)
        vol_365 = calculate_close_to_close_vol(returns, annualization_factor=365)
        # vol_365 should be sqrt(365/252) times vol_252
        expected_ratio = np.sqrt(365 / 252)
        assert np.isclose(vol_365 / vol_252, expected_ratio, rtol=1e-6)


class TestParkinsonVol:
    """Tests for Parkinson (high-low) volatility."""

    def test_zero_range(self):
        """When high = low, volatility should be zero."""
        prices = np.full(20, 100.0)
        vol = calculate_parkinson_vol(prices, prices)
        assert vol == 0.0

    def test_positive_range(self):
        """Positive high-low range should give positive vol."""
        high = np.array([101, 102, 101.5, 103])
        low = np.array([99, 98, 99.5, 97])
        vol = calculate_parkinson_vol(high, low)
        assert vol > 0

    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        vol = calculate_parkinson_vol(np.array([]), np.array([]))
        assert np.isnan(vol)

    def test_invalid_prices(self):
        """Zero prices should be filtered out."""
        high = np.array([100, 0, 100])
        low = np.array([99, 0, 99])
        vol = calculate_parkinson_vol(high, low)
        # Should only use valid points
        assert not np.isnan(vol)


class TestGarmanKlassVol:
    """Tests for Garman-Klass volatility."""

    def test_no_movement(self):
        """When all prices are equal, volatility should be zero."""
        prices = np.full(20, 100.0)
        vol = calculate_garman_klass_vol(prices, prices, prices, prices)
        assert vol == 0.0

    def test_positive_volatility(self):
        """Normal OHLC data should give positive vol."""
        n = 50
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.2
        high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)

        vol = calculate_garman_klass_vol(open_, high, low, close)
        assert vol > 0 and not np.isnan(vol)


# =============================================================================
# Term Structure Tests
# =============================================================================


class TestPowerLaw:
    """Tests for power law term structure model."""

    def test_power_law_formula(self):
        """Test power law formula: σ(T) = a * T^b."""
        t = np.array([0.1, 0.25, 0.5, 1.0])
        a, b = 0.2, -0.1
        result = power_law(t, a, b)
        expected = a * np.power(t, b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_law_decay(self):
        """Negative b should give decreasing vol with time."""
        t = np.array([0.1, 0.5, 1.0])
        result = power_law(t, a=0.2, b=-0.1)
        # Should be monotonically decreasing
        assert all(result[i] > result[i + 1] for i in range(len(result) - 1))

    def test_power_law_growth(self):
        """Positive b should give increasing vol with time."""
        t = np.array([0.1, 0.5, 1.0])
        result = power_law(t, a=0.2, b=0.1)
        # Should be monotonically increasing
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))


class TestFitPowerLaw:
    """Tests for power law fitting."""

    def test_fit_exact_data(self):
        """Fitting exact power law data should recover parameters."""
        t = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
        true_a, true_b = 0.15, -0.2
        y = power_law(t, true_a, true_b)

        a, b, rmse = fit_power_law(t, y)

        assert np.isclose(a, true_a, rtol=0.01)
        assert np.isclose(b, true_b, rtol=0.01)
        assert rmse < 1e-6

    def test_fit_noisy_data(self):
        """Fitting noisy data should give reasonable results."""
        np.random.seed(42)
        t = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
        y = power_law(t, 0.15, -0.2) + np.random.randn(5) * 0.001

        a, b, rmse = fit_power_law(t, y)

        # Should be close to true values
        assert 0.10 < a < 0.20
        assert -0.3 < b < -0.1
        assert rmse < 0.01

    def test_fit_insufficient_data(self):
        """Fitting with < 2 points should raise error."""
        t = np.array([0.5])
        y = np.array([0.15])

        with pytest.raises(ValueError):
            fit_power_law(t, y)


# =============================================================================
# SVI Surface Metrics Tests
# =============================================================================


class TestSVIMetrics:
    """Tests for SVI-derived metrics."""

    def test_atm_vol_calculation(self):
        """Test ATM vol calculation from SVI params."""
        # Typical SVI parameters
        a = 0.04  # 4% total variance at ATM
        b = 0.1
        rho = -0.3
        m = 0.0
        sigma = 0.1
        tte = 1.0  # 1 year

        atm_vol = get_atm_vol_from_svi(a, b, rho, m, sigma, tte)

        # With these params at ATM (k=0):
        # w(0) = a + b * (rho * (-m) + sqrt(m^2 + sigma^2))
        # w(0) = 0.04 + 0.1 * (0 + 0.1) = 0.04 + 0.01 = 0.05
        # IV = sqrt(0.05 / 1) = 0.2236
        expected = np.sqrt(0.05)
        assert np.isclose(atm_vol, expected, rtol=0.01)

    def test_atm_vol_negative_variance(self):
        """Negative variance should return NaN."""
        # Force negative total variance
        atm_vol = get_atm_vol_from_svi(a=-1, b=0.01, rho=0, m=0, sigma=0.01, tte_years=1.0)
        assert np.isnan(atm_vol)

    def test_vol_at_moneyness(self):
        """Test vol calculation at different moneyness levels."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        tte = 1.0

        # ATM vol
        vol_atm = get_vol_at_moneyness(a, b, rho, m, sigma, tte, 0.0)

        # OTM put (negative moneyness)
        vol_otm_put = get_vol_at_moneyness(a, b, rho, m, sigma, tte, -0.2)

        # OTM call (positive moneyness)
        vol_otm_call = get_vol_at_moneyness(a, b, rho, m, sigma, tte, 0.2)

        # With negative rho, OTM puts should have higher vol (skew)
        assert vol_otm_put > vol_atm

        # Both should be positive
        assert vol_atm > 0
        assert vol_otm_put > 0
        assert vol_otm_call > 0

    def test_vol_symmetry_with_zero_rho(self):
        """With rho=0, smile should be symmetric."""
        a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        tte = 1.0

        vol_neg = get_vol_at_moneyness(a, b, rho, m, sigma, tte, -0.1)
        vol_pos = get_vol_at_moneyness(a, b, rho, m, sigma, tte, 0.1)

        assert np.isclose(vol_neg, vol_pos, rtol=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRealizedVolCalculator:
    """Integration tests for RealizedVolCalculator."""

    def test_calculate_for_date_no_data(self):
        """Should return result with None values when no data."""
        calc = RealizedVolCalculator()
        result = calc.calculate_for_date("INVALID_SYMBOL", date(2024, 1, 1))

        assert result.symbol == "INVALID_SYMBOL"
        assert result.rv_10d is None


class TestTermStructureAnalyzer:
    """Integration tests for TermStructureAnalyzer."""

    def test_fit_empty_dataframe(self):
        """Should return None for empty DataFrame."""
        analyzer = TermStructureAnalyzer()
        df = pd.DataFrame()

        result = analyzer.fit_atm_term_structure(df)
        assert result is None

    def test_fit_insufficient_data(self):
        """Should return None when < 2 expirations."""
        analyzer = TermStructureAnalyzer()
        df = pd.DataFrame({"tte_years": [0.1], "atm_vol": [0.2]})

        result = analyzer.fit_atm_term_structure(df)
        assert result is None
