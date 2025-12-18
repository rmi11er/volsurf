"""Term Structure page for Streamlit dashboard."""

from datetime import date
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics import SurfaceMetrics, TermStructureAnalyzer, power_law
from volsurf.database.connection import get_connection


def render_term_structure(
    symbol: str, selected_date: date, available_dates: List[date]
) -> None:
    """Render the term structure analysis page."""
    tabs = st.tabs(["Current Term Structure", "ATM Vol Evolution", "Skew Evolution"])

    with tabs[0]:
        render_current_term_structure(symbol, selected_date)

    with tabs[1]:
        render_atm_evolution(symbol, available_dates)

    with tabs[2]:
        render_skew_evolution(symbol, available_dates)


def render_current_term_structure(symbol: str, selected_date: date) -> None:
    """Render current term structure with power law fit."""
    st.subheader(f"Term Structure on {selected_date}")

    analyzer = TermStructureAnalyzer()
    data = analyzer.get_term_structure_data(symbol, selected_date)

    if not data:
        st.warning("No term structure data available for this date.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Fit power law
    result = analyzer.analyze_date(symbol, selected_date)

    # Create figure with both ATM vol and fitted curve
    fig = go.Figure()

    # Actual ATM vols
    fig.add_trace(
        go.Scatter(
            x=df["tte_days"],
            y=df["atm_vol"] * 100,
            mode="markers+lines",
            name="ATM Vol",
            marker=dict(size=10),
            hovertemplate="DTE: %{x:.0f}<br>ATM Vol: %{y:.2f}%<extra></extra>",
        )
    )

    # Fitted power law
    if result and result.atm_fit:
        fit = result.atm_fit
        tte_smooth = np.linspace(df["tte_years"].min(), df["tte_years"].max(), 100)
        vol_fitted = power_law(tte_smooth, fit.a, fit.b)

        fig.add_trace(
            go.Scatter(
                x=tte_smooth * 365,
                y=vol_fitted * 100,
                mode="lines",
                name=f"Fit: {fit.a:.4f} * T^{fit.b:.4f}",
                line=dict(dash="dash", color="red"),
            )
        )

    fig.update_layout(
        title=f"{symbol} ATM Volatility Term Structure",
        xaxis_title="Days to Expiry",
        yaxis_title="ATM Implied Vol (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show fit parameters
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Term Structure Data:**")
        display_df = df.copy()
        display_df["atm_vol_pct"] = (display_df["atm_vol"] * 100).round(2)
        if "skew" in display_df.columns:
            display_df["skew_pct"] = (display_df["skew"] * 100).round(2)
            show_cols = ["expiration_date", "tte_days", "atm_vol_pct", "skew_pct"]
        else:
            show_cols = ["expiration_date", "tte_days", "atm_vol_pct"]

        st.dataframe(
            display_df[show_cols].rename(
                columns={
                    "expiration_date": "Expiration",
                    "tte_days": "DTE",
                    "atm_vol_pct": "ATM Vol (%)",
                    "skew_pct": "Skew (%)",
                }
            ),
            hide_index=True,
        )

    with col2:
        if result and result.atm_fit:
            st.markdown("**Power Law Fit:**")
            st.markdown(f"**Model:** `sigma(T) = a * T^b`")
            st.metric("a (level)", f"{result.atm_fit.a:.6f}")
            st.metric("b (slope)", f"{result.atm_fit.b:.6f}")
            st.metric("RMSE", f"{result.atm_fit.rmse:.4%}")

            # Interpretation
            if result.atm_fit.b < 0:
                st.info("Negative b: Inverted term structure (short-term vol > long-term)")
            elif result.atm_fit.b > 0:
                st.info("Positive b: Normal term structure (long-term vol > short-term)")
            else:
                st.info("Flat term structure")


def render_atm_evolution(symbol: str, available_dates: List[date]) -> None:
    """Render ATM vol evolution over time."""
    st.subheader("ATM Volatility Evolution")

    col1, col2 = st.columns([1, 3])

    with col1:
        tenor = st.selectbox(
            "Select Tenor (days)",
            options=[7, 14, 21, 30, 45, 60, 90, 180],
            index=3,  # Default to 30
        )

    metrics = SurfaceMetrics()
    start_date = available_dates[0]
    end_date = available_dates[-1]

    df = metrics.get_atm_vol_timeseries(symbol, start_date, end_date, tte_target_days=tenor)

    if df.empty:
        st.warning("No ATM vol data available for the selected tenor.")
        return

    # Create time series plot
    fig = px.line(
        df,
        x="date",
        y="atm_vol",
        title=f"{symbol} {tenor}-Day ATM Implied Volatility",
        labels={"atm_vol": "ATM Vol", "date": "Date"},
    )

    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current", f"{df['atm_vol'].iloc[-1]:.2%}")
    with col2:
        st.metric("Mean", f"{df['atm_vol'].mean():.2%}")
    with col3:
        st.metric("Min", f"{df['atm_vol'].min():.2%}")
    with col4:
        st.metric("Max", f"{df['atm_vol'].max():.2%}")


def render_skew_evolution(symbol: str, available_dates: List[date]) -> None:
    """Render skew evolution over time."""
    st.subheader("Volatility Skew Evolution")

    col1, col2 = st.columns([1, 3])

    with col1:
        tenor = st.selectbox(
            "Select Tenor (days)",
            options=[7, 14, 21, 30, 45, 60, 90, 180],
            index=3,
            key="skew_tenor",
        )

    metrics = SurfaceMetrics()
    start_date = available_dates[0]
    end_date = available_dates[-1]

    df = metrics.get_skew_timeseries(symbol, start_date, end_date, tte_target_days=tenor)

    if df.empty:
        st.warning("No skew data available for the selected tenor.")
        return

    # Create time series plot
    fig = px.line(
        df,
        x="date",
        y="skew",
        title=f"{symbol} {tenor}-Day 25-Delta Skew",
        labels={"skew": "Skew", "date": "Date"},
    )

    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(hovermode="x unified")

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current", f"{df['skew'].iloc[-1]:.2%}")
    with col2:
        st.metric("Mean", f"{df['skew'].mean():.2%}")
    with col3:
        st.metric("Min", f"{df['skew'].min():.2%}")
    with col4:
        st.metric("Max", f"{df['skew'].max():.2%}")

    # Interpretation
    with st.expander("Skew Interpretation"):
        st.markdown(
            """
        **25-Delta Skew** = 25D Put IV - 25D Call IV

        - **Negative skew** (typical for equities): Put protection is expensive
        - **More negative** = Higher demand for downside protection
        - **Less negative / positive** = Unusual, may indicate bullish sentiment

        Skew typically increases (becomes more negative) during market stress.
        """
        )
