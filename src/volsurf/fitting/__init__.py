"""Volatility surface fitting module."""

from volsurf.fitting.arbitrage import (
    ArbitrageCheckResult,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    validate_surface,
    validate_term_structure,
)
from volsurf.fitting.comparison import (
    FullComparisonResult,
    ModelComparator,
    ModelComparisonResult,
    ModelType,
    print_comparison_summary,
)
from volsurf.fitting.implied_vol import (
    black_scholes_price,
    black_scholes_vega,
    calculate_forward_from_options,
    calculate_iv_bisection,
    calculate_iv_from_mid_prices,
    calculate_iv_newton,
)
from volsurf.fitting.pipeline import (
    FitStats,
    PipelineConfig,
    SurfaceFittingPipeline,
    batch_fit_surfaces,
    fit_surfaces_for_date,
)
from volsurf.fitting.polynomial import (
    PolynomialFitResult,
    fit_polynomial_slice,
    polynomial_atm_vol,
    polynomial_curvature,
    polynomial_implied_vol,
    polynomial_skew,
)
from volsurf.fitting.sabr import (
    SABRFitResult,
    check_sabr_constraints,
    fit_sabr_slice,
    sabr_atm_vol,
    sabr_implied_vol,
    sabr_implied_vol_vectorized,
    sabr_skew,
)
from volsurf.fitting.svi import (
    SVIFitResult,
    check_svi_constraints,
    compute_vega_weights,
    estimate_forward_price,
    fit_svi_slice,
    svi_implied_vol,
    svi_total_variance,
)

__all__ = [
    # Arbitrage
    "ArbitrageCheckResult",
    "check_butterfly_arbitrage",
    "check_calendar_arbitrage",
    "validate_surface",
    "validate_term_structure",
    # Comparison
    "FullComparisonResult",
    "ModelComparator",
    "ModelComparisonResult",
    "ModelType",
    "print_comparison_summary",
    # Implied vol
    "black_scholes_price",
    "black_scholes_vega",
    "calculate_forward_from_options",
    "calculate_iv_bisection",
    "calculate_iv_from_mid_prices",
    "calculate_iv_newton",
    # Pipeline
    "FitStats",
    "PipelineConfig",
    "SurfaceFittingPipeline",
    "batch_fit_surfaces",
    "fit_surfaces_for_date",
    # Polynomial
    "PolynomialFitResult",
    "fit_polynomial_slice",
    "polynomial_atm_vol",
    "polynomial_curvature",
    "polynomial_implied_vol",
    "polynomial_skew",
    # SABR
    "SABRFitResult",
    "check_sabr_constraints",
    "fit_sabr_slice",
    "sabr_atm_vol",
    "sabr_implied_vol",
    "sabr_implied_vol_vectorized",
    "sabr_skew",
    # SVI
    "SVIFitResult",
    "check_svi_constraints",
    "compute_vega_weights",
    "estimate_forward_price",
    "fit_svi_slice",
    "svi_implied_vol",
    "svi_total_variance",
]
