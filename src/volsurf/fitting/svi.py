"""SVI (Stochastic Volatility Inspired) model fitting."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from scipy import optimize

from volsurf.models.schemas import SVIParams


@dataclass
class SVIFitResult:
    """Result of SVI fitting."""

    params: SVIParams
    rmse: float
    mae: float
    max_error: float
    num_points: int
    success: bool
    message: str


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """
    Calculate SVI total variance.

    w(k) = a + b * (ρ * (k - m) + √((k - m)² + σ²))

    Args:
        k: Log-moneyness array (log(K/F))
        a: Vertical shift (ATM variance level)
        b: Wing slope (non-negative)
        rho: Skew parameter (-1 to 1)
        m: Horizontal shift
        sigma: Smoothness parameter (positive)

    Returns:
        Total variance w(k)
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_implied_vol(
    k: np.ndarray,
    tte: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """
    Calculate SVI implied volatility.

    Args:
        k: Log-moneyness array
        tte: Time to expiration in years
        a, b, rho, m, sigma: SVI parameters

    Returns:
        Implied volatility array
    """
    w = svi_total_variance(k, a, b, rho, m, sigma)
    # Ensure non-negative variance before sqrt
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / tte)


def check_svi_constraints(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Tuple[bool, str]:
    """
    Check SVI no-arbitrage constraints.

    Constraints:
    1. b >= 0 (wing slope non-negative)
    2. |rho| < 1 (valid correlation)
    3. a + b * sigma * sqrt(1 - rho^2) >= 0 (non-negative min variance)

    Returns:
        (is_valid, message)
    """
    if b < 0:
        return False, f"b must be >= 0, got {b}"

    if abs(rho) >= 1:
        return False, f"|rho| must be < 1, got {rho}"

    if sigma <= 0:
        return False, f"sigma must be > 0, got {sigma}"

    # Minimum variance constraint
    min_variance = a + b * sigma * math.sqrt(1 - rho**2)
    if min_variance < 0:
        return False, f"Minimum variance is negative: {min_variance}"

    return True, "OK"


def fit_svi_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward_price: float,
    tte_years: float,
    weights: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> SVIFitResult:
    """
    Fit SVI model to a single expiration slice.

    Args:
        strikes: Strike prices
        implied_vols: Implied volatilities (annualized, as decimals e.g. 0.20 for 20%)
        forward_price: Forward price of underlying
        tte_years: Time to expiration in years
        weights: Optional weights for each data point (e.g., vega weights)
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance

    Returns:
        SVIFitResult with fitted parameters and quality metrics
    """
    # Convert to log-moneyness
    k = np.log(strikes / forward_price)

    # Convert to total variance
    w_market = implied_vols**2 * tte_years

    # Default uniform weights if not provided
    if weights is None:
        weights = np.ones_like(k)

    # Normalize weights
    weights = weights / np.sum(weights)

    # Initial parameter guess
    atm_idx = np.argmin(np.abs(k))
    atm_variance = w_market[atm_idx] if len(w_market) > 0 else 0.04

    x0 = np.array([
        max(atm_variance, 0.01),  # a: ATM variance level
        0.1,                       # b: moderate wing slope
        -0.3,                      # rho: typical equity skew
        0.0,                       # m: centered at ATM
        0.1,                       # sigma: moderate smoothness
    ])

    def objective(params):
        """Weighted sum of squared errors."""
        a, b, rho, m, sigma = params
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        residuals = w_model - w_market
        return np.sum(weights * residuals**2)

    def constraint_b_positive(params):
        """b >= 0"""
        return params[1]

    def constraint_rho_upper(params):
        """-1 < rho < 1 (upper bound: 1 - rho > 0)"""
        return 0.999 - params[2]

    def constraint_rho_lower(params):
        """-1 < rho < 1 (lower bound: rho + 1 > 0)"""
        return params[2] + 0.999

    def constraint_sigma_positive(params):
        """sigma > 0"""
        return params[4] - 1e-6

    def constraint_min_variance(params):
        """a + b * sigma * sqrt(1 - rho^2) >= 0"""
        a, b, rho, m, sigma = params
        return a + b * sigma * math.sqrt(max(1 - rho**2, 1e-10))

    constraints = [
        {"type": "ineq", "fun": constraint_b_positive},
        {"type": "ineq", "fun": constraint_rho_upper},
        {"type": "ineq", "fun": constraint_rho_lower},
        {"type": "ineq", "fun": constraint_sigma_positive},
        {"type": "ineq", "fun": constraint_min_variance},
    ]

    # Bounds for parameters
    bounds = [
        (1e-6, 1.0),      # a: small positive to reasonable variance
        (0.0, 2.0),       # b: non-negative, reasonable upper
        (-0.999, 0.999),  # rho: strictly between -1 and 1
        (-0.5, 0.5),      # m: around ATM
        (1e-4, 1.0),      # sigma: positive, reasonable range
    ]

    try:
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        a, b, rho, m, sigma = result.x

        # Validate constraints
        is_valid, msg = check_svi_constraints(a, b, rho, m, sigma)

        # Calculate fit metrics
        w_fitted = svi_total_variance(k, a, b, rho, m, sigma)
        iv_fitted = np.sqrt(np.maximum(w_fitted / tte_years, 1e-10))
        iv_errors = iv_fitted - implied_vols

        rmse = float(np.sqrt(np.mean(iv_errors**2)))
        mae = float(np.mean(np.abs(iv_errors)))
        max_error = float(np.max(np.abs(iv_errors)))

        params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

        return SVIFitResult(
            params=params,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            num_points=len(strikes),
            success=result.success and is_valid,
            message=result.message if result.success else msg,
        )

    except Exception as e:
        logger.error(f"SVI fitting failed: {e}")
        # Return a default failed result
        return SVIFitResult(
            params=SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1),
            rmse=float("inf"),
            mae=float("inf"),
            max_error=float("inf"),
            num_points=len(strikes),
            success=False,
            message=str(e),
        )


def compute_vega_weights(
    strikes: np.ndarray,
    forward_price: float,
    implied_vols: np.ndarray,
    tte_years: float,
    reference_tte: float = 100 / 365,  # ~100 days
) -> np.ndarray:
    """
    Compute vega-based weights for SVI fitting.

    ATM options have highest vega, so this emphasizes ATM points.

    Args:
        strikes: Strike prices
        forward_price: Forward price
        implied_vols: Implied volatilities
        tte_years: Time to expiration in years
        reference_tte: Reference TTE for scaling

    Returns:
        Weight array (not normalized)
    """
    # Approximate vega: vega ~ S * sqrt(T) * N'(d1)
    # For simplicity, use moneyness-based proxy
    moneyness = strikes / forward_price

    # Vega is highest ATM, decreases as we move away
    # Use a Gaussian-like weighting centered at ATM
    log_m = np.log(moneyness)
    atm_vol = np.mean(implied_vols)  # Rough ATM vol estimate

    # Standard deviation for weighting (wider for longer-dated)
    weight_std = atm_vol * np.sqrt(tte_years) * 0.5

    weights = np.exp(-0.5 * (log_m / weight_std) ** 2)

    # Scale by sqrt(TTE/reference) to normalize across expirations
    weights *= np.sqrt(tte_years / reference_tte)

    return weights


def estimate_forward_price(
    underlying_price: float,
    tte_years: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.015,
) -> float:
    """
    Estimate forward price using cost-of-carry model.

    F = S * exp((r - q) * T)

    Args:
        underlying_price: Current spot price
        tte_years: Time to expiration in years
        risk_free_rate: Risk-free rate (annualized)
        dividend_yield: Dividend yield (annualized)

    Returns:
        Estimated forward price
    """
    return underlying_price * math.exp((risk_free_rate - dividend_yield) * tte_years)
