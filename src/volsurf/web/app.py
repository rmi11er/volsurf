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
from pathlib import Path

from volsurf.analytics import SurfaceMetrics
from volsurf.database.connection import get_connection
from volsurf.database.schema import init_schema


def get_available_symbols() -> list[dict]:
    """Get list of symbols with data in the database."""
    conn = get_connection()
    query = """
        SELECT
            symbol,
            MIN(quote_date) as start_date,
            MAX(quote_date) as end_date,
            COUNT(DISTINCT quote_date) as trading_days
        FROM fitted_surfaces
        GROUP BY symbol
        ORDER BY symbol
    """
    result = conn.execute(query).fetchall()
    return [
        {
            "symbol": r[0],
            "start_date": r[1],
            "end_date": r[2],
            "trading_days": r[3],
        }
        for r in result
    ]


def main() -> None:
    """Main entry point for the Streamlit app."""
    init_schema()

    st.sidebar.title("VolSurf Dashboard")
    st.sidebar.markdown("---")

    # Navigation (first, so Watchlist/Trade Journal can short-circuit)
    page = st.sidebar.radio(
        "Navigation",
        [
            "Watchlist",
            "Macro Context",
            "Overview",
            "Surface Viewer",
            "Term Structure",
            "VRP Analysis",
            "Intraday",
            "Diagnostics",
            "Trade Journal",
        ],
    )

    # Pages that manage their own data scope (cross-symbol)
    if page == "Watchlist":
        show_watchlist()
        return
    if page == "Macro Context":
        show_macro_context()
        return
    if page == "Trade Journal":
        show_trade_journal_placeholder()
        return

    # ------------------------------------------------------------------
    # Symbol & date controls (for single-symbol pages)
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")

    available_symbols = get_available_symbols()

    if not available_symbols:
        st.warning("No fitted surfaces found. Please run data ingestion and fitting first.")
        st.code("volsurf ingest backfill SPY --start 2024-01-01 --end 2024-12-31")
        st.code("volsurf fit batch SPY --start 2024-01-01 --end 2024-12-31")
        return

    symbol_options = [s["symbol"] for s in available_symbols]
    symbol_info = {s["symbol"]: s for s in available_symbols}

    symbol = st.sidebar.selectbox(
        "Symbol",
        options=symbol_options,
        index=0,
        format_func=lambda x: f"{x} ({symbol_info[x]['trading_days']} days)",
    )

    info = symbol_info[symbol]
    st.sidebar.caption(f"Data: {info['start_date']} → {info['end_date']}")

    metrics = SurfaceMetrics()
    available_dates = metrics.get_available_dates(symbol)

    if not available_dates:
        st.warning(f"No data found for {symbol}. Please run data ingestion first.")
        return

    # Date range
    st.sidebar.markdown("### Analysis Window")
    min_date = available_dates[0]
    max_date = available_dates[-1]

    lookback_options = {
        "7 days": 7,
        "14 days": 14,
        "30 days": 30,
        "60 days": 60,
        "90 days": 90,
        "180 days": 180,
        "YTD": "ytd",
        "All": "all",
        "Custom": "custom",
    }

    lookback = st.sidebar.selectbox(
        "Lookback Period",
        options=list(lookback_options.keys()),
        index=2,
    )

    lookback_value = lookback_options[lookback]

    if lookback_value == "all":
        range_start = min_date
        range_end = max_date
    elif lookback_value == "ytd":
        range_start = date(max_date.year, 1, 1)
        if range_start < min_date:
            range_start = min_date
        range_end = max_date
    elif lookback_value == "custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            range_start = st.date_input(
                "From",
                value=max_date - timedelta(days=30),
                min_value=min_date,
                max_value=max_date,
                key="custom_start",
            )
        with col2:
            range_end = st.date_input(
                "To",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="custom_end",
            )
    else:
        range_start = max_date - timedelta(days=lookback_value)
        if range_start < min_date:
            range_start = min_date
        range_end = max_date

    filtered_dates = [d for d in available_dates if range_start <= d <= range_end]

    if not filtered_dates:
        st.warning(f"No data in selected range ({range_start} to {range_end})")
        return

    st.sidebar.caption(f"Window: {range_start} → {range_end} ({len(filtered_dates)} days)")

    st.sidebar.markdown("---")
    selected_date = st.sidebar.date_input(
        "Focus Date",
        value=filtered_dates[-1],
        min_value=filtered_dates[0],
        max_value=filtered_dates[-1],
        help="Date for single-date views (Surface Viewer, Diagnostics)",
    )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if page == "Overview":
        show_overview(symbol, selected_date, filtered_dates)
    elif page == "Surface Viewer":
        show_surface_viewer(symbol, selected_date)
    elif page == "Term Structure":
        show_term_structure(symbol, selected_date, filtered_dates)
    elif page == "VRP Analysis":
        show_vrp_analysis(symbol, filtered_dates)
    elif page == "Intraday":
        show_intraday(symbol)
    elif page == "Diagnostics":
        show_diagnostics(symbol, selected_date)


# ======================================================================
# Page renderers
# ======================================================================

def show_watchlist() -> None:
    from volsurf.web.components.watchlist import render_watchlist

    render_watchlist()


def show_macro_context() -> None:
    from volsurf.web.components.macro_context import render_macro_context

    render_macro_context()


def show_trade_journal_placeholder() -> None:
    from volsurf.web.components.trade_journal import render_trade_journal

    render_trade_journal()


def show_overview(symbol: str, selected_date: date, available_dates: list) -> None:
    """Overview page with surface metrics snapshot and percentile context."""
    st.title(f"{symbol} Volatility Overview")

    start_date = available_dates[0]
    end_date = available_dates[-1]
    st.caption(f"Window: {start_date} → {end_date} ({len(available_dates)} trading days)")

    metrics = SurfaceMetrics()
    summary = metrics.get_surface_summary(symbol, selected_date)

    # Try to get enriched snapshot for deltas and percentiles
    from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator

    smh_calc = SurfaceMetricsHistoryCalculator()
    enriched = smh_calc.get_enriched_snapshot(symbol, selected_date)

    if enriched:
        snap = enriched["snapshot"]
        pctiles = enriched.get("percentiles", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            val = f"{snap.atm_vol_30d:.2%}" if snap.atm_vol_30d else "N/A"
            delta = f"{snap.atm_vol_30d_1d_chg:.2%}" if snap.atm_vol_30d_1d_chg else None
            st.metric("ATM Vol (30d)", val, delta=delta)
            p = pctiles.get("atm_vol_30d", {}).get(252)
            if p is not None:
                st.caption(f"{p:.0f}th pctile (252d)")
        with col2:
            val = f"{snap.skew_25d_30d:.4f}" if snap.skew_25d_30d else "N/A"
            delta = f"{snap.skew_25d_30d_1d_chg:.4f}" if snap.skew_25d_30d_1d_chg else None
            st.metric("Skew (30d)", val, delta=delta)
            p = pctiles.get("skew_25d_30d", {}).get(252)
            if p is not None:
                st.caption(f"{p:.0f}th pctile (252d)")
        with col3:
            st.metric("Surfaces Fitted", summary.num_surfaces)
            st.metric(
                "Avg Fit RMSE",
                f"{summary.avg_rmse:.4%}" if summary.avg_rmse else "N/A",
            )
        with col4:
            val = f"{snap.term_slope:.4f}" if snap.term_slope else "N/A"
            delta = f"{snap.term_slope_1d_chg:.4f}" if snap.term_slope_1d_chg else None
            st.metric("Term Slope", val, delta=delta)
            p = pctiles.get("term_slope", {}).get(252)
            if p is not None:
                st.caption(f"{p:.0f}th pctile (252d)")
    else:
        # Fallback: original overview without percentile context
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

    # 25-Delta Skew time series
    st.markdown("---")
    st.subheader("25-Delta Skew (30-Day)")

    skew_df = metrics.get_skew_timeseries(symbol, start_date, end_date, tte_target_days=30)

    if not skew_df.empty:
        import plotly.express as px

        fig = px.line(
            skew_df,
            x="date",
            y="skew",
            title=f"{symbol} 30-Day 25-Delta Skew",
            labels={"skew": "Skew", "date": "Date"},
        )
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skew data available for the selected range.")

    # Window summary
    st.markdown("---")
    st.subheader("Window Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Analysis Window:** {start_date} to {end_date}")
        st.markdown(f"**Trading Days:** {len(available_dates)}")
    with col2:
        if summary.num_surfaces > 0:
            st.markdown(f"**Expirations on {selected_date}:** {summary.num_surfaces}")
            if summary.atm_vol_min and summary.atm_vol_max:
                st.markdown(
                    f"**ATM Vol Range:** {summary.atm_vol_min:.2%} - {summary.atm_vol_max:.2%}"
                )


def show_surface_viewer(symbol: str, selected_date: date) -> None:
    st.title(f"{symbol} Volatility Surface")
    st.markdown(f"**Date:** {selected_date}")

    from volsurf.web.components.surface_viewer import render_surface_viewer

    render_surface_viewer(symbol, selected_date)


def show_term_structure(symbol: str, selected_date: date, available_dates: list) -> None:
    st.title(f"{symbol} Term Structure Analysis")

    from volsurf.web.components.term_structure import render_term_structure

    render_term_structure(symbol, selected_date, available_dates)


def show_vrp_analysis(symbol: str, available_dates: list) -> None:
    st.title(f"{symbol} Variance Risk Premium Analysis")

    from volsurf.web.components.vrp_analysis import render_vrp_analysis

    render_vrp_analysis(symbol, available_dates)


def show_diagnostics(symbol: str, selected_date: date) -> None:
    st.title(f"{symbol} Model Diagnostics")

    from volsurf.web.components.diagnostics import render_diagnostics

    render_diagnostics(symbol, selected_date)


def show_intraday(symbol: str) -> None:
    from volsurf.web.components.intraday_viewer import render_intraday_viewer

    data_dir = Path("data/intraday")
    render_intraday_viewer(symbol, data_dir)


if __name__ == "__main__":
    main()
