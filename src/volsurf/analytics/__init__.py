"""Analytics module for volatility surface analysis."""

from volsurf.analytics.realized_vol import (
    RealizedVolCalculator,
    RealizedVolResult,
    calculate_close_to_close_vol,
    calculate_garman_klass_vol,
    calculate_parkinson_vol,
)
from volsurf.analytics.surface_metrics import (
    SurfaceMetrics,
    SurfaceSummary,
    get_atm_vol_from_svi,
    get_vol_at_moneyness,
)
from volsurf.analytics.term_structure import (
    TermStructureAnalyzer,
    TermStructureFit,
    TermStructureResult,
    fit_power_law,
    power_law,
)
from volsurf.analytics.variance_risk_premium import (
    VRPCalculator,
    VRPResult,
)
from volsurf.analytics.visualization import (
    plot_atm_vol_timeseries,
    plot_iv_vs_rv,
    plot_term_structure,
    plot_vol_smile,
    plot_vrp_timeseries,
    print_ascii_timeseries,
    print_realized_vol_table,
    print_surface_summary,
    print_term_structure_table,
    print_vrp_summary,
)

__all__ = [
    # Realized vol
    "RealizedVolCalculator",
    "RealizedVolResult",
    "calculate_close_to_close_vol",
    "calculate_parkinson_vol",
    "calculate_garman_klass_vol",
    # VRP
    "VRPCalculator",
    "VRPResult",
    # Term structure
    "TermStructureAnalyzer",
    "TermStructureFit",
    "TermStructureResult",
    "power_law",
    "fit_power_law",
    # Surface metrics
    "SurfaceMetrics",
    "SurfaceSummary",
    "get_atm_vol_from_svi",
    "get_vol_at_moneyness",
    # Visualization
    "print_ascii_timeseries",
    "print_term_structure_table",
    "print_surface_summary",
    "print_vrp_summary",
    "print_realized_vol_table",
    "plot_atm_vol_timeseries",
    "plot_term_structure",
    "plot_vol_smile",
    "plot_iv_vs_rv",
    "plot_vrp_timeseries",
]
