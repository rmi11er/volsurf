"""Web dashboard components."""

from volsurf.web.components.diagnostics import render_diagnostics
from volsurf.web.components.macro_context import render_macro_context
from volsurf.web.components.surface_viewer import render_surface_viewer
from volsurf.web.components.term_structure import render_term_structure
from volsurf.web.components.vrp_analysis import render_vrp_analysis
from volsurf.web.components.trade_journal import render_trade_journal
from volsurf.web.components.watchlist import render_watchlist

__all__ = [
    "render_surface_viewer",
    "render_term_structure",
    "render_vrp_analysis",
    "render_diagnostics",
    "render_macro_context",
    "render_watchlist",
    "render_trade_journal",
]
