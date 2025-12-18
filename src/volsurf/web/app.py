"""Main Streamlit application for volatility surface analysis."""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="VolSurf - Volatility Surface Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from datetime import date, timedelta

from volsurf.analytics import SurfaceMetrics
from volsurf.database.schema import init_schema


def main() -> None:
    """Main entry point for the Streamlit app."""
    # Initialize database schema
    init_schema()

    # Sidebar navigation
    st.sidebar.title("VolSurf Dashboard")
    st.sidebar.markdown("---")

    # Symbol selection
    symbol = st.sidebar.text_input("Symbol", value="SPY")

    # Get available dates
    metrics = SurfaceMetrics()
    available_dates = metrics.get_available_dates(symbol)

    if not available_dates:
        st.warning(f"No data found for {symbol}. Please run data ingestion first.")
        st.code("volsurf ingest backfill SPY --start 2025-12-01 --end 2025-12-16")
        return

    # Date selection
    st.sidebar.markdown("### Date Range")
    min_date = available_dates[0]
    max_date = available_dates[-1]

    selected_date = st.sidebar.date_input(
        "Analysis Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Surface Viewer", "Term Structure", "VRP Analysis", "Diagnostics"],
    )

    # Main content based on page selection
    if page == "Overview":
        show_overview(symbol, selected_date, available_dates)
    elif page == "Surface Viewer":
        show_surface_viewer(symbol, selected_date)
    elif page == "Term Structure":
        show_term_structure(symbol, selected_date, available_dates)
    elif page == "VRP Analysis":
        show_vrp_analysis(symbol, available_dates)
    elif page == "Diagnostics":
        show_diagnostics(symbol, selected_date)


def show_overview(symbol: str, selected_date: date, available_dates: list) -> None:
    """Show overview dashboard."""
    st.title(f"{symbol} Volatility Overview")
    st.markdown(f"**Analysis Date:** {selected_date}")

    metrics = SurfaceMetrics()
    summary = metrics.get_surface_summary(symbol, selected_date)

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ATM Vol (30d)",
            f"{summary.atm_vol_mean:.2%}" if summary.atm_vol_mean else "N/A",
        )
    with col2:
        st.metric(
            "Avg Skew",
            f"{summary.skew_mean:.4f}" if summary.skew_mean else "N/A",
        )
    with col3:
        st.metric("Surfaces Fitted", summary.num_surfaces)
    with col4:
        st.metric(
            "Avg Fit RMSE",
            f"{summary.avg_rmse:.4%}" if summary.avg_rmse else "N/A",
        )

    # ATM Vol time series
    st.markdown("---")
    st.subheader("ATM Implied Volatility (30-Day)")

    start_date = available_dates[0]
    end_date = available_dates[-1]

    atm_df = metrics.get_atm_vol_timeseries(symbol, start_date, end_date, tte_target_days=30)

    if not atm_df.empty:
        import plotly.express as px

        fig = px.line(
            atm_df,
            x="date",
            y="atm_vol",
            title=f"{symbol} 30-Day ATM Implied Volatility",
            labels={"atm_vol": "ATM Vol", "date": "Date"},
        )
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ATM vol data available for the selected range.")

    # Data availability summary
    st.markdown("---")
    st.subheader("Data Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Date Range:** {available_dates[0]} to {available_dates[-1]}")
        st.markdown(f"**Trading Days:** {len(available_dates)}")
    with col2:
        if summary.num_surfaces > 0:
            st.markdown(f"**Expirations on {selected_date}:** {summary.num_surfaces}")
            if summary.atm_vol_min and summary.atm_vol_max:
                st.markdown(
                    f"**ATM Vol Range:** {summary.atm_vol_min:.2%} - {summary.atm_vol_max:.2%}"
                )


def show_surface_viewer(symbol: str, selected_date: date) -> None:
    """Show 3D surface viewer and smile plots."""
    st.title(f"{symbol} Volatility Surface")
    st.markdown(f"**Date:** {selected_date}")

    from volsurf.web.pages.surface_viewer import render_surface_viewer

    render_surface_viewer(symbol, selected_date)


def show_term_structure(symbol: str, selected_date: date, available_dates: list) -> None:
    """Show term structure analysis."""
    st.title(f"{symbol} Term Structure Analysis")

    from volsurf.web.pages.term_structure import render_term_structure

    render_term_structure(symbol, selected_date, available_dates)


def show_vrp_analysis(symbol: str, available_dates: list) -> None:
    """Show VRP analysis page."""
    st.title(f"{symbol} Variance Risk Premium Analysis")

    from volsurf.web.pages.vrp_analysis import render_vrp_analysis

    render_vrp_analysis(symbol, available_dates)


def show_diagnostics(symbol: str, selected_date: date) -> None:
    """Show model diagnostics page."""
    st.title(f"{symbol} Model Diagnostics")

    from volsurf.web.pages.diagnostics import render_diagnostics

    render_diagnostics(symbol, selected_date)


if __name__ == "__main__":
    main()
