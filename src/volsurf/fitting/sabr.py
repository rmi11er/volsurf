"""SABR (Stochastic Alpha Beta Rho) model fitting.

The SABR model is a stochastic volatility model widely used for
interest rate derivatives and equity options. It captures the
volatility smile through a closed-form approximation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from scipy import optimize

from volsurf.models.schemas import SABRParams


@dataclass
class SABRFitResult:
    """Result of SABR fitting."""

    params: SABRParams
    rmse: float
    mae: float
    max_error: float
    num_points: int
    success: bool
    message: str


def sabr_implied_vol(
    forward: float,
    strike: float,
    tte: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Calculate SABR implied volatility using Hagan's approximation.

    This is the 2002 Hagan et al. approximation formula.

    Args:
        forward: Forward price
        strike: Strike price
        tte: Time to expiration in years
        alpha: Initial volatility level
        beta: CEV exponent (0 = normal, 1 = lognormal)
        rho: Correlation between forward and vol
        nu: Vol-of-vol

    Returns:
        Implied volatility (annualized)
    """
    if tte <= 0:
        return 0.0

    # Handle ATM case specially for numerical stability
    if abs(forward - strike) < 1e-10:
        # ATM approximation
        fk = forward
        fk_beta = fk ** (1 - beta)

        term1 = alpha / fk_beta
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / fk ** (2 - 2 * beta))
            + (rho * beta * nu * alpha) / (4 * fk_beta)
            + ((2 - 3 * rho ** 2) * nu ** 2) / 24
        ) * tte

        return term1 * term2

    # General case
    fk = forward * strike
    fk_beta = fk ** ((1 - beta) / 2)

    log_fk = math.log(forward / strike)

    # z parameter
    z = (nu / alpha) * fk_beta * log_fk

    # x(z) function
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        sqrt_term = math.sqrt(1 - 2 * rho * z + z ** 2)
        x_z = z / math.log((sqrt_term + z - rho) / (1 - rho))

    # Expansion terms
    one_minus_beta = 1 - beta
    fk_pow = fk ** (one_minus_beta / 2)

    # Denominator
    denom = fk_pow * (
        1
        + (one_minus_beta ** 2 / 24) * log_fk ** 2
        + (one_minus_beta ** 4 / 1920) * log_fk ** 4
    )

    # Numerator correction
    numer_corr = 1 + (
        (one_minus_beta ** 2 / 24) * (alpha ** 2 / fk ** one_minus_beta)
        + (rho * beta * nu * alpha) / (4 * fk_pow)
        + ((2 - 3 * rho ** 2) * nu ** 2) / 24
    ) * tte

    vol = (alpha / denom) * x_z * numer_corr

    return max(vol, 1e-10)


def sabr_implied_vol_vectorized(
    forward: float,
    strikes: np.ndarray,
    tte: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Vectorized SABR implied vol calculation."""
    return np.array([
        sabr_implied_vol(forward, k, tte, alpha, beta, rho, nu)
        for k in strikes
    ])


def check_sabr_constraints(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> Tuple[bool, str]:
    """
    Check SABR parameter constraints.

    Constraints:
    1. alpha > 0 (positive initial vol)
    2. 0 <= beta <= 1 (CEV exponent)
    3. -1 < rho < 1 (valid correlation)
    4. nu >= 0 (non-negative vol-of-vol)

    Returns:
        (is_valid, message)
    """
    if alpha <= 0:
        return False, f"alpha must be > 0, got {alpha}"

    if beta < 0 or beta > 1:
        return False, f"beta must be in [0, 1], got {beta}"

    if abs(rho) >= 1:
        return False, f"|rho| must be < 1, got {rho}"

    if nu < 0:
        return False, f"nu must be >= 0, got {nu}"

    return True, "OK"


def fit_sabr_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward_price: float,
    tte_years: float,
    beta: float = 0.5,
    weights: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> SABRFitResult:
    """
    Fit SABR model to a single expiration slice.

    Args:
        strikes: Strike prices
        implied_vols: Implied volatilities (annualized, as decimals)
        forward_price: Forward price of underlying
        tte_years: Time to expiration in years
        beta: CEV exponent (fixed, typically 0.5 for equities)
        weights: Optional weights for each data point
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance

    Returns:
        SABRFitResult with fitted parameters and quality metrics
    """
    if len(strikes) < 3:
        return SABRFitResult(
            params=SABRParams(alpha=0.2, beta=beta, rho=-0.3, nu=0.3),
            rmse=float("inf"),
            mae=float("inf"),
            max_error=float("inf"),
            num_points=len(strikes),
            success=False,
            message="Need at least 3 strikes for SABR fitting",
        )

    # Default uniform weights
    if weights is None:
        weights = np.ones_like(strikes)
    weights = weights / np.sum(weights)

    # Initial guess: estimate alpha from ATM vol
    atm_idx = np.argmin(np.abs(strikes - forward_price))
    atm_vol = implied_vols[atm_idx]

    # For beta = 0.5, alpha ≈ ATM_vol * F^(1-beta)
    alpha_init = atm_vol * forward_price ** (1 - beta)

    x0 = np.array([
        alpha_init,  # alpha
        -0.3,        # rho (typical equity skew)
        0.3,         # nu (moderate vol-of-vol)
    ])

    def objective(params):
        """Weighted sum of squared errors."""
        alpha, rho, nu = params

        try:
            iv_model = sabr_implied_vol_vectorized(
                forward_price, strikes, tte_years, alpha, beta, rho, nu
            )
            residuals = iv_model - implied_vols
            return np.sum(weights * residuals ** 2)
        except (ValueError, RuntimeWarning):
            return 1e10

    # Bounds
    bounds = [
        (1e-4, 2.0),      # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 2.0),      # nu
    ]

    try:
        result = optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        alpha, rho, nu = result.x

        # Validate constraints
        is_valid, msg = check_sabr_constraints(alpha, beta, rho, nu)

        # Calculate fit metrics
        iv_fitted = sabr_implied_vol_vectorized(
            forward_price, strikes, tte_years, alpha, beta, rho, nu
        )
        iv_errors = iv_fitted - implied_vols

        rmse = float(np.sqrt(np.mean(iv_errors ** 2)))
        mae = float(np.mean(np.abs(iv_errors)))
        max_error = float(np.max(np.abs(iv_errors)))

        params = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)

        return SABRFitResult(
            params=params,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            num_points=len(strikes),
            success=result.success and is_valid,
            message=result.message if result.success else msg,
        )

    except Exception as e:
        logger.error(f"SABR fitting failed: {e}")
        return SABRFitResult(
            params=SABRParams(alpha=0.2, beta=beta, rho=-0.3, nu=0.3),
            rmse=float("inf"),
            mae=float("inf"),
            max_error=float("inf"),
            num_points=len(strikes),
            success=False,
            message=str(e),
        )


def sabr_atm_vol(
    forward: float,
    tte: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Calculate ATM implied vol from SABR parameters."""
    return sabr_implied_vol(forward, forward, tte, alpha, beta, rho, nu)


def sabr_skew(
    forward: float,
    tte: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    delta_strike_pct: float = 0.10,
) -> float:
    """
    Calculate vol skew (put vol - call vol) from SABR parameters.

    Args:
        forward: Forward price
        tte: Time to expiration
        alpha, beta, rho, nu: SABR parameters
        delta_strike_pct: Moneyness offset for skew calculation

    Returns:
        Skew (OTM put vol - OTM call vol)
    """
    put_strike = forward * (1 - delta_strike_pct)
    call_strike = forward * (1 + delta_strike_pct)

    put_vol = sabr_implied_vol(forward, put_strike, tte, alpha, beta, rho, nu)
    call_vol = sabr_implied_vol(forward, call_strike, tte, alpha, beta, rho, nu)

    return put_vol - call_vol
