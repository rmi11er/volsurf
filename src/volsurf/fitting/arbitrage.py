"""Arbitrage validation for fitted volatility surfaces."""

from dataclasses import dataclass
from typing import List

import numpy as np

from volsurf.models.schemas import SVIParams


@dataclass
class ArbitrageCheckResult:
    """Result of arbitrage validation."""

    passes: bool
    butterfly_violations: int
    calendar_violations: int
    messages: List[str]


def check_butterfly_arbitrage(
    params: SVIParams,
    k_grid: np.ndarray,
    tte_years: float,
    tolerance: float = 1e-6,
) -> tuple[bool, int, List[str]]:
    """
    Check for butterfly arbitrage violations.

    Butterfly arbitrage is violated if the total variance function w(k)
    is not convex, i.e., d²w/dk² < 0 at any point.

    For SVI: d²w/dk² = b * σ² / ((k-m)² + σ²)^(3/2)

    Since b >= 0 and σ > 0, this is always >= 0 for valid SVI parameters.
    However, we still check numerically to catch edge cases.

    Args:
        params: SVI parameters
        k_grid: Grid of log-moneyness points to check
        tte_years: Time to expiration
        tolerance: Tolerance for violation detection

    Returns:
        (passes, violation_count, messages)
    """
    messages = []
    violations = 0

    # Calculate second derivative analytically for SVI
    # d²w/dk² = b * σ² / ((k-m)² + σ²)^(3/2)
    k_minus_m = k_grid - params.m
    denominator = (k_minus_m**2 + params.sigma**2) ** 1.5
    d2w_dk2 = params.b * params.sigma**2 / denominator

    # Check for negative values (violations)
    violation_mask = d2w_dk2 < -tolerance
    violations = int(np.sum(violation_mask))

    if violations > 0:
        messages.append(f"Butterfly arbitrage: {violations} violations at k={k_grid[violation_mask][:5]}")

    passes = violations == 0
    return passes, violations, messages


def check_calendar_arbitrage(
    surfaces: List[tuple[float, SVIParams]],
    k_grid: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[bool, int, List[str]]:
    """
    Check for calendar arbitrage violations.

    Calendar arbitrage is violated if total variance decreases with time
    for any strike. That is, for T1 < T2, we need w(k, T1) <= w(k, T2).

    Args:
        surfaces: List of (tte_years, SVIParams) sorted by tte ascending
        k_grid: Grid of log-moneyness points to check
        tolerance: Tolerance for violation detection

    Returns:
        (passes, violation_count, messages)
    """
    messages = []
    violations = 0

    if len(surfaces) < 2:
        return True, 0, []

    # Sort by TTE
    surfaces_sorted = sorted(surfaces, key=lambda x: x[0])

    for i in range(len(surfaces_sorted) - 1):
        tte1, params1 = surfaces_sorted[i]
        tte2, params2 = surfaces_sorted[i + 1]

        # Calculate total variance at each TTE
        w1 = _calculate_total_variance(params1, k_grid)
        w2 = _calculate_total_variance(params2, k_grid)

        # Check if variance decreases
        violation_mask = w2 < w1 - tolerance
        count = int(np.sum(violation_mask))

        if count > 0:
            violations += count
            messages.append(
                f"Calendar arbitrage: {count} violations between T={tte1:.4f} and T={tte2:.4f}"
            )

    passes = violations == 0
    return passes, violations, messages


def _calculate_total_variance(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """Calculate total variance for an array of log-moneyness values."""
    return params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )


def check_density_positivity(
    params: SVIParams,
    k_grid: np.ndarray,
    tte_years: float,
    tolerance: float = 1e-6,
) -> tuple[bool, int, List[str]]:
    """
    Check that the implied density is non-negative.

    The risk-neutral density must be non-negative, which requires:
    g(k) = (1 - k*w'/(2w))² - w'^2/4 * (1/w + 1/4) + w''/2 >= 0

    This is a necessary condition for no static arbitrage.

    Args:
        params: SVI parameters
        k_grid: Grid of log-moneyness points
        tte_years: Time to expiration
        tolerance: Tolerance for violation detection

    Returns:
        (passes, violation_count, messages)
    """
    messages = []

    # Calculate w, w', w''
    k = k_grid
    k_minus_m = k - params.m
    sqrt_term = np.sqrt(k_minus_m**2 + params.sigma**2)

    w = params.a + params.b * (params.rho * k_minus_m + sqrt_term)

    # First derivative: dw/dk = b * (rho + (k-m)/sqrt((k-m)² + σ²))
    dw = params.b * (params.rho + k_minus_m / sqrt_term)

    # Second derivative: d²w/dk² = b * σ² / ((k-m)² + σ²)^(3/2)
    d2w = params.b * params.sigma**2 / (sqrt_term**3)

    # Check density condition
    # g(k) = (1 - k*w'/(2w))² - w'^2/4 * (1/w + 1/4) + w''/2
    term1 = (1 - k * dw / (2 * w)) ** 2
    term2 = dw**2 / 4 * (1 / w + 0.25)
    term3 = d2w / 2

    g = term1 - term2 + term3

    violation_mask = g < -tolerance
    violations = int(np.sum(violation_mask))

    if violations > 0:
        messages.append(f"Density positivity: {violations} violations")

    passes = violations == 0
    return passes, violations, messages


def validate_surface(
    params: SVIParams,
    tte_years: float,
    k_min: float = -0.5,
    k_max: float = 0.5,
    n_points: int = 100,
) -> ArbitrageCheckResult:
    """
    Run all arbitrage checks on a fitted surface.

    Args:
        params: SVI parameters
        tte_years: Time to expiration in years
        k_min: Minimum log-moneyness to check
        k_max: Maximum log-moneyness to check
        n_points: Number of points in the grid

    Returns:
        ArbitrageCheckResult with all validation results
    """
    k_grid = np.linspace(k_min, k_max, n_points)
    messages = []

    # Check butterfly arbitrage
    butterfly_passes, butterfly_violations, butterfly_msgs = check_butterfly_arbitrage(
        params, k_grid, tte_years
    )
    messages.extend(butterfly_msgs)

    # Check density positivity
    density_passes, density_violations, density_msgs = check_density_positivity(
        params, k_grid, tte_years
    )
    messages.extend(density_msgs)

    # Overall pass requires all checks to pass
    passes = butterfly_passes and density_passes

    return ArbitrageCheckResult(
        passes=passes,
        butterfly_violations=butterfly_violations + density_violations,
        calendar_violations=0,  # Calendar checked separately across expirations
        messages=messages,
    )


def validate_term_structure(
    surfaces: List[tuple[float, SVIParams]],
    k_min: float = -0.5,
    k_max: float = 0.5,
    n_points: int = 100,
) -> ArbitrageCheckResult:
    """
    Validate a term structure of surfaces for calendar arbitrage.

    Args:
        surfaces: List of (tte_years, SVIParams) tuples
        k_min: Minimum log-moneyness
        k_max: Maximum log-moneyness
        n_points: Number of grid points

    Returns:
        ArbitrageCheckResult for calendar arbitrage
    """
    k_grid = np.linspace(k_min, k_max, n_points)

    passes, violations, messages = check_calendar_arbitrage(surfaces, k_grid)

    return ArbitrageCheckResult(
        passes=passes,
        butterfly_violations=0,
        calendar_violations=violations,
        messages=messages,
    )
