"""Polynomial smile model fitting.

Simple polynomial fit to the volatility smile. Useful as a baseline
for comparison with parametric models like SVI and SABR.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from volsurf.models.schemas import PolynomialParams


@dataclass
class PolynomialFitResult:
    """Result of polynomial fitting."""

    params: PolynomialParams
    rmse: float
    mae: float
    max_error: float
    num_points: int
    success: bool
    message: str


def polynomial_implied_vol(
    k: np.ndarray,
    coefficients: list[float],
) -> np.ndarray:
    """
    Calculate implied volatility from polynomial coefficients.

    IV(k) = a0 + a1*k + a2*k^2 + a3*k^3 + ...

    Args:
        k: Log-moneyness array
        coefficients: Polynomial coefficients [a0, a1, a2, ...]

    Returns:
        Implied volatility array
    """
    result = np.zeros_like(k, dtype=float)
    for i, coef in enumerate(coefficients):
        result += coef * (k ** i)
    return np.maximum(result, 1e-10)


def fit_polynomial_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward_price: float,
    tte_years: float,
    degree: int = 4,
    weights: Optional[np.ndarray] = None,
) -> PolynomialFitResult:
    """
    Fit polynomial model to a single expiration slice.

    Args:
        strikes: Strike prices
        implied_vols: Implied volatilities (annualized, as decimals)
        forward_price: Forward price of underlying
        tte_years: Time to expiration in years (not used but kept for API consistency)
        degree: Polynomial degree (1-6)
        weights: Optional weights for each data point

    Returns:
        PolynomialFitResult with fitted coefficients and quality metrics
    """
    if len(strikes) < degree + 1:
        return PolynomialFitResult(
            params=PolynomialParams(coefficients=[0.2], degree=1),
            rmse=float("inf"),
            mae=float("inf"),
            max_error=float("inf"),
            num_points=len(strikes),
            success=False,
            message=f"Need at least {degree + 1} strikes for degree {degree} polynomial",
        )

    # Convert to log-moneyness
    k = np.log(strikes / forward_price)

    # Default uniform weights
    if weights is None:
        weights = np.ones_like(k)

    try:
        # Weighted polynomial fit using numpy
        coefficients = np.polyfit(k, implied_vols, degree, w=weights)

        # Reverse to get [a0, a1, a2, ...] order (polyfit returns highest first)
        coefficients = coefficients[::-1].tolist()

        # Calculate fit metrics
        iv_fitted = polynomial_implied_vol(k, coefficients)
        iv_errors = iv_fitted - implied_vols

        rmse = float(np.sqrt(np.mean(iv_errors ** 2)))
        mae = float(np.mean(np.abs(iv_errors)))
        max_error = float(np.max(np.abs(iv_errors)))

        params = PolynomialParams(coefficients=coefficients, degree=degree)

        return PolynomialFitResult(
            params=params,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            num_points=len(strikes),
            success=True,
            message="OK",
        )

    except Exception as e:
        logger.error(f"Polynomial fitting failed: {e}")
        return PolynomialFitResult(
            params=PolynomialParams(coefficients=[0.2], degree=1),
            rmse=float("inf"),
            mae=float("inf"),
            max_error=float("inf"),
            num_points=len(strikes),
            success=False,
            message=str(e),
        )


def polynomial_atm_vol(coefficients: list[float]) -> float:
    """
    Get ATM implied vol from polynomial (k=0).

    At ATM (k=0), IV = a0 (constant term).
    """
    return coefficients[0] if coefficients else 0.0


def polynomial_skew(
    coefficients: list[float],
    delta_k: float = 0.10,
) -> float:
    """
    Calculate vol skew from polynomial coefficients.

    Skew = IV(-delta_k) - IV(+delta_k)

    Args:
        coefficients: Polynomial coefficients
        delta_k: Log-moneyness offset for skew calculation

    Returns:
        Skew (OTM put vol - OTM call vol)
    """
    put_vol = polynomial_implied_vol(np.array([-delta_k]), coefficients)[0]
    call_vol = polynomial_implied_vol(np.array([delta_k]), coefficients)[0]
    return put_vol - call_vol


def polynomial_curvature(coefficients: list[float]) -> float:
    """
    Get smile curvature (second derivative at ATM).

    Curvature = 2 * a2 (coefficient of k^2 term).
    """
    if len(coefficients) >= 3:
        return 2 * coefficients[2]
    return 0.0
