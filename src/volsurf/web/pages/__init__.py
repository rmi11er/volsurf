"""Web dashboard pages."""

from volsurf.web.pages.diagnostics import render_diagnostics
from volsurf.web.pages.surface_viewer import render_surface_viewer
from volsurf.web.pages.term_structure import render_term_structure
from volsurf.web.pages.vrp_analysis import render_vrp_analysis

__all__ = [
    "render_surface_viewer",
    "render_term_structure",
    "render_vrp_analysis",
    "render_diagnostics",
]
