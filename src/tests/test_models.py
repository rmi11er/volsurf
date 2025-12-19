"""Tests for SABR, polynomial, and comparison models."""

import math

import numpy as np
import pytest

from volsurf.fitting.sabr import (
    SABRFitResult,
    check_sabr_constraints,
    fit_sabr_slice,
    sabr_atm_vol,
    sabr_implied_vol,
    sabr_implied_vol_vectorized,
    sabr_skew,
)
from volsurf.fitting.polynomial import (
    PolynomialFitResult,
    fit_polynomial_slice,
    polynomial_atm_vol,
    polynomial_curvature,
    polynomial_implied_vol,
    polynomial_skew,
)
from volsurf.fitting.comparison import (
    FullComparisonResult,
    ModelComparator,
    ModelComparisonResult,
    ModelType,
)
from volsurf.models.schemas import SABRParams, PolynomialParams


class TestSABR:
    """Tests for SABR model."""

    def test_sabr_implied_vol_atm(self):
        """ATM SABR vol should be close to alpha for beta=1."""
        forward = 100.0
        strike = 100.0
        tte = 1.0
        alpha = 0.20
        beta = 1.0
        rho = 0.0
        nu = 0.3

        iv = sabr_implied_vol(forward, strike, tte, alpha, beta, rho, nu)

        # For beta=1, ATM vol should be close to alpha
        assert abs(iv - alpha) < 0.02

    def test_sabr_implied_vol_symmetry(self):
        """SABR with rho=0 should be symmetric around ATM."""
        forward = 100.0
        tte = 0.5
        alpha = 0.20
        beta = 0.7
        rho = 0.0
        nu = 0.4

        iv_put = sabr_implied_vol(forward, 90, tte, alpha, beta, rho, nu)
        iv_call = sabr_implied_vol(forward, 110, tte, alpha, beta, rho, nu)

        # Approximately symmetric (not exact due to Hagan approximation)
        assert abs(iv_put - iv_call) < 0.01

    def test_sabr_implied_vol_skew(self):
        """Negative rho should give higher OTM put vol."""
        forward = 100.0
        tte = 0.5
        alpha = 0.20
        beta = 0.7
        rho = -0.5
        nu = 0.4

        iv_put = sabr_implied_vol(forward, 90, tte, alpha, beta, rho, nu)
        iv_call = sabr_implied_vol(forward, 110, tte, alpha, beta, rho, nu)

        # Negative rho means puts have higher vol (negative skew)
        assert iv_put > iv_call

    def test_sabr_implied_vol_vectorized(self):
        """Vectorized SABR should match scalar version."""
        forward = 100.0
        strikes = np.array([90, 95, 100, 105, 110])
        tte = 0.5
        alpha = 0.20
        beta = 0.7
        rho = -0.3
        nu = 0.4

        ivs_vec = sabr_implied_vol_vectorized(forward, strikes, tte, alpha, beta, rho, nu)

        for i, K in enumerate(strikes):
            iv_scalar = sabr_implied_vol(forward, K, tte, alpha, beta, rho, nu)
            assert abs(ivs_vec[i] - iv_scalar) < 1e-10

    def test_sabr_atm_vol(self):
        """ATM vol function should match implied vol at K=F."""
        forward = 100.0
        tte = 0.5
        alpha = 0.20
        beta = 0.7
        rho = -0.3
        nu = 0.4

        atm = sabr_atm_vol(forward, tte, alpha, beta, rho, nu)
        iv_atm = sabr_implied_vol(forward, forward, tte, alpha, beta, rho, nu)

        assert abs(atm - iv_atm) < 1e-10

    def test_sabr_skew_with_strong_rho(self):
        """SABR skew should reflect negative rho in put-call vol diff."""
        forward = 100.0
        tte = 0.5
        alpha = 0.20
        beta = 0.7
        rho = -0.8  # Strong negative correlation
        nu = 0.5

        # Check that OTM puts have higher vol than OTM calls
        iv_put = sabr_implied_vol(forward, 90, tte, alpha, beta, rho, nu)
        iv_call = sabr_implied_vol(forward, 110, tte, alpha, beta, rho, nu)

        # With strong negative rho, OTM puts should have higher vol
        assert iv_put > iv_call

    def test_sabr_constraints_valid(self):
        """Valid SABR parameters should pass constraints."""
        is_valid, msg = check_sabr_constraints(
            alpha=0.20, beta=0.7, rho=-0.3, nu=0.4
        )
        assert is_valid
        assert msg == "OK"

    def test_sabr_constraints_invalid_alpha(self):
        """Negative alpha should fail."""
        is_valid, msg = check_sabr_constraints(
            alpha=-0.20, beta=0.7, rho=-0.3, nu=0.4
        )
        assert not is_valid
        assert "alpha" in msg.lower()

    def test_sabr_constraints_invalid_beta(self):
        """Beta outside [0,1] should fail."""
        is_valid, msg = check_sabr_constraints(
            alpha=0.20, beta=1.5, rho=-0.3, nu=0.4
        )
        assert not is_valid
        assert "beta" in msg.lower()

    def test_sabr_constraints_invalid_rho(self):
        """Rho outside [-1,1] should fail."""
        is_valid, msg = check_sabr_constraints(
            alpha=0.20, beta=0.7, rho=-1.5, nu=0.4
        )
        assert not is_valid
        assert "rho" in msg.lower()

    def test_sabr_constraints_invalid_nu(self):
        """Negative nu should fail."""
        is_valid, msg = check_sabr_constraints(
            alpha=0.20, beta=0.7, rho=-0.3, nu=-0.4
        )
        assert not is_valid
        assert "nu" in msg.lower()

    def test_fit_sabr_synthetic(self):
        """Fit SABR to synthetic data."""
        # True parameters
        forward = 100.0
        tte = 0.5
        true_alpha = 0.20
        true_beta = 0.7
        true_rho = -0.3
        true_nu = 0.4

        # Generate synthetic smile
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        ivs = sabr_implied_vol_vectorized(
            forward, strikes, tte, true_alpha, true_beta, true_rho, true_nu
        )

        # Fit with fixed beta
        result = fit_sabr_slice(strikes, ivs, forward, tte, beta=true_beta)

        assert result is not None
        assert result.success
        assert result.rmse < 0.001  # Should be very accurate on clean data

    def test_fit_sabr_with_noise(self):
        """SABR fitting should handle noisy data."""
        forward = 100.0
        tte = 0.5
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])

        # Generate realistic noisy smile
        np.random.seed(42)
        ivs = np.array([0.25, 0.23, 0.21, 0.20, 0.195, 0.19, 0.20])
        ivs += np.random.normal(0, 0.002, len(ivs))

        result = fit_sabr_slice(strikes, ivs, forward, tte, beta=0.7)

        assert result is not None
        assert result.rmse < 0.01


class TestPolynomial:
    """Tests for polynomial smile model."""

    def test_polynomial_implied_vol_quadratic(self):
        """Test quadratic polynomial evaluation."""
        k = np.array([-0.1, 0.0, 0.1])
        coeffs = [0.20, -0.05, 0.10]  # a0 + a1*k + a2*k^2

        ivs = polynomial_implied_vol(k, coeffs)

        # Manual calculation
        expected = 0.20 + (-0.05) * k + 0.10 * k**2
        np.testing.assert_allclose(ivs, expected)

    def test_polynomial_atm_vol(self):
        """ATM vol should equal a0 (intercept)."""
        coeffs = [0.20, -0.05, 0.10]

        atm = polynomial_atm_vol(coeffs)

        assert abs(atm - 0.20) < 1e-10

    def test_polynomial_skew(self):
        """Skew is IV(put) - IV(call) at +/- delta_k."""
        # With negative linear term, lower strikes have higher vol
        coeffs = [0.20, -0.10, 0.05]  # a0 + a1*k + a2*k^2

        # Using default delta_k=0.10
        # IV(-0.10) = 0.20 + (-0.10)*(-0.10) + 0.05*0.01 = 0.20 + 0.01 + 0.0005 = 0.2105
        # IV(+0.10) = 0.20 + (-0.10)*(0.10) + 0.05*0.01 = 0.20 - 0.01 + 0.0005 = 0.1905
        # Skew = 0.2105 - 0.1905 = 0.02
        skew = polynomial_skew(coeffs)

        # With negative a1, lower strikes have higher vol, so skew > 0
        assert skew > 0

    def test_polynomial_curvature(self):
        """Curvature should equal 2*a2."""
        coeffs = [0.20, -0.05, 0.10]

        curv = polynomial_curvature(coeffs)

        assert abs(curv - 0.20) < 1e-10  # 2 * 0.10

    def test_fit_polynomial_quadratic(self):
        """Fit quadratic polynomial to synthetic data."""
        forward = 100.0
        tte = 0.5

        # True polynomial: 0.20 - 0.05*k + 0.10*k^2
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        k = np.log(strikes / forward)
        true_ivs = 0.20 - 0.05 * k + 0.10 * k**2

        result = fit_polynomial_slice(strikes, true_ivs, forward, tte, degree=2)

        assert result is not None
        assert result.success
        assert result.rmse < 0.001

        # Check recovered coefficients (access via params)
        coeffs = result.params.coefficients
        assert abs(coeffs[0] - 0.20) < 0.01
        assert abs(coeffs[1] - (-0.05)) < 0.01
        assert abs(coeffs[2] - 0.10) < 0.02

    def test_fit_polynomial_higher_degree(self):
        """Higher degree polynomial should fit better."""
        forward = 100.0
        tte = 0.5
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])

        # Realistic asymmetric smile
        ivs = np.array([0.30, 0.26, 0.23, 0.21, 0.20, 0.195, 0.19, 0.19, 0.195])

        result_2 = fit_polynomial_slice(strikes, ivs, forward, tte, degree=2)
        result_4 = fit_polynomial_slice(strikes, ivs, forward, tte, degree=4)

        assert result_4.rmse <= result_2.rmse


class TestSABRParams:
    """Tests for SABRParams Pydantic model."""

    def test_params_creation(self):
        """SABRParams should be created correctly."""
        params = SABRParams(alpha=0.20, beta=0.7, rho=-0.3, nu=0.4)

        assert params.alpha == 0.20
        assert params.beta == 0.7
        assert params.rho == -0.3
        assert params.nu == 0.4

    def test_params_validation_alpha(self):
        """Negative alpha should fail validation."""
        with pytest.raises(ValueError):
            SABRParams(alpha=-0.20, beta=0.7, rho=-0.3, nu=0.4)

    def test_params_validation_beta(self):
        """Beta outside [0,1] should fail validation."""
        with pytest.raises(ValueError):
            SABRParams(alpha=0.20, beta=1.5, rho=-0.3, nu=0.4)

        with pytest.raises(ValueError):
            SABRParams(alpha=0.20, beta=-0.1, rho=-0.3, nu=0.4)

    def test_params_validation_rho(self):
        """Rho outside [-1,1] should fail validation."""
        with pytest.raises(ValueError):
            SABRParams(alpha=0.20, beta=0.7, rho=-1.5, nu=0.4)

    def test_params_validation_nu(self):
        """Negative nu should fail validation."""
        with pytest.raises(ValueError):
            SABRParams(alpha=0.20, beta=0.7, rho=-0.3, nu=-0.4)


class TestPolynomialParams:
    """Tests for PolynomialParams Pydantic model."""

    def test_params_creation(self):
        """PolynomialParams should be created correctly."""
        params = PolynomialParams(coefficients=[0.20, -0.05, 0.10], degree=2)

        assert params.coefficients == [0.20, -0.05, 0.10]
        assert params.degree == 2

    def test_params_validation_degree(self):
        """Degree should be positive and not too high."""
        with pytest.raises(ValueError):
            PolynomialParams(coefficients=[0.20], degree=0)

        with pytest.raises(ValueError):
            PolynomialParams(coefficients=[0.20], degree=10)


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_types(self):
        """All expected model types should exist."""
        assert ModelType.SVI.value == "SVI"
        assert ModelType.SABR.value == "SABR"
        assert ModelType.POLYNOMIAL.value == "POLYNOMIAL"


class TestModelComparisonResult:
    """Tests for ModelComparisonResult dataclass."""

    def test_result_creation(self):
        """ModelComparisonResult should be created correctly."""
        from datetime import date
        from volsurf.fitting.svi import SVIFitResult
        from volsurf.models.schemas import SVIParams

        # Create a mock SVI result
        svi_result = SVIFitResult(
            params=SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1),
            rmse=0.002,
            mae=0.001,
            max_error=0.005,
            num_points=10,
            success=True,
            message="OK",
        )

        result = ModelComparisonResult(
            expiration_date=date(2025, 12, 20),
            tte_years=0.5,
            num_points=10,
            forward_price=100.0,
            svi_result=svi_result,
            best_model=ModelType.SVI,
            best_rmse=0.002,
        )

        assert result.best_model == ModelType.SVI
        assert result.svi_result.rmse == 0.002

    def test_get_all_rmse(self):
        """get_all_rmse should return RMSE for all successful fits."""
        from datetime import date
        from volsurf.fitting.svi import SVIFitResult
        from volsurf.fitting.sabr import SABRFitResult
        from volsurf.models.schemas import SVIParams, SABRParams

        svi_result = SVIFitResult(
            params=SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1),
            rmse=0.002,
            mae=0.001,
            max_error=0.005,
            num_points=10,
            success=True,
            message="OK",
        )

        sabr_result = SABRFitResult(
            params=SABRParams(alpha=0.2, beta=0.7, rho=-0.3, nu=0.4),
            rmse=0.003,
            mae=0.002,
            max_error=0.006,
            num_points=10,
            success=True,
            message="OK",
        )

        result = ModelComparisonResult(
            expiration_date=date(2025, 12, 20),
            tte_years=0.5,
            num_points=10,
            forward_price=100.0,
            svi_result=svi_result,
            sabr_result=sabr_result,
        )

        rmse_dict = result.get_all_rmse()

        assert ModelType.SVI in rmse_dict
        assert ModelType.SABR in rmse_dict
        assert rmse_dict[ModelType.SVI] == 0.002
        assert rmse_dict[ModelType.SABR] == 0.003


class TestFullComparisonResult:
    """Tests for FullComparisonResult dataclass."""

    def test_full_result_creation(self):
        """FullComparisonResult should be created correctly."""
        from datetime import date

        result = FullComparisonResult(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            expiration_results=[],
            avg_svi_rmse=0.002,
            avg_sabr_rmse=0.003,
            avg_poly_rmse=0.004,
            overall_best_model=ModelType.SVI,
        )

        assert result.symbol == "SPY"
        assert result.overall_best_model == ModelType.SVI
        assert result.avg_svi_rmse == 0.002
