"""Volatility surface fitting module."""

from volsurf.fitting.arbitrage import (
    ArbitrageCheckResult,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
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
from volsurf.fitting.pipeline import (
    FitStats,
    PipelineConfig,
    SurfaceFittingPipeline,
    batch_fit_surfaces,
    fit_surfaces_for_date,
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
    # SVI
    "SVIFitResult",
    "check_svi_constraints",
    "compute_vega_weights",
    "estimate_forward_price",
    "fit_svi_slice",
    "svi_implied_vol",
    "svi_total_variance",
]
