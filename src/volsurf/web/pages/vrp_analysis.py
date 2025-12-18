"""VRP Analysis page for Streamlit dashboard."""

from datetime import date
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics import RealizedVolCalculator, SurfaceMetrics, VRPCalculator
from volsurf.database.connection import get_connection


def render_vrp_analysis(symbol: str, available_dates: List[date]) -> None:
    """Render the VRP analysis page."""
    tabs = st.tabs(["VRP Time Series", "IV vs RV", "Realized Vol"])

    with tabs[0]:
        render_vrp_timeseries(symbol, available_dates)

    with tabs[1]:
        render_iv_vs_rv(symbol, available_dates)

    with tabs[2]:
        render_realized_vol(symbol, available_dates)


def render_vrp_timeseries(symbol: str, available_dates: List[date]) -> None:
    """Render VRP time series analysis."""
    st.subheader("Variance Risk Premium Time Series")

    calculator = VRPCalculator()
    start_date = available_dates[0]
    end_date = available_dates[-1]

    df = calculator.get_vrp_timeseries(symbol, start_date, end_date)

    if df.empty:
        st.warning("No VRP data available. Backfill VRP first:")
        st.code(f"volsurf analyze vrp {symbol} --backfill --start {start_date} --end {end_date}")

        # Try to calculate on the fly for the latest date
        st.info("Calculating VRP for latest date...")
        result = calculator.calculate_for_date(symbol, end_date)
        if result.vrp_30d is not None:
            st.success(f"VRP (30-day) on {end_date}: {result.vrp_30d:.2%}")
            st.markdown(f"- IV (30d): {result.implied_vol_30d:.2%}")
            st.markdown(f"- RV (21d): {result.realized_vol_30d:.2%}")
        return

    # Create VRP time series plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["vrp_30d"] * 100,
            mode="lines",
            name="VRP (30d)",
            fill="tozeroy",
            fillcolor="rgba(0, 100, 80, 0.2)",
            line=dict(color="rgb(0, 100, 80)"),
            hovertemplate="Date: %{x}<br>VRP: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"{symbol} Variance Risk Premium (IV - RV)",
        xaxis_title="Date",
        yaxis_title="VRP (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current = df["vrp_30d"].iloc[-1]
        st.metric("Current VRP", f"{current:.2%}")
    with col2:
        mean = df["vrp_30d"].mean()
        st.metric("Mean VRP", f"{mean:.2%}")
    with col3:
        pct_positive = (df["vrp_30d"] > 0).mean()
        st.metric("% Positive", f"{pct_positive:.1%}")
    with col4:
        std = df["vrp_30d"].std()
        zscore = (current - mean) / std if std > 0 else 0
        st.metric("Z-Score", f"{zscore:.2f}")

    # VRP distribution
    st.markdown("---")
    st.subheader("VRP Distribution")

    fig = px.histogram(
        df,
        x="vrp_30d",
        nbins=30,
        title="VRP Distribution",
        labels={"vrp_30d": "VRP (30-day)"},
    )

    fig.update_xaxes(tickformat=".1%")

    # Add current value marker
    fig.add_vline(
        x=current,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: {current:.2%}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    with st.expander("VRP Interpretation"):
        st.markdown(
            """
        **Variance Risk Premium (VRP)** = Implied Volatility - Realized Volatility

        - **Positive VRP**: IV > RV, options are "expensive" relative to realized moves
        - **Negative VRP**: IV < RV, options are "cheap" relative to realized moves

        VRP is typically positive, representing the risk premium option sellers collect.

        **Trading Implications:**
        - High VRP: Potential opportunity to sell options (short vol)
        - Low/Negative VRP: Caution for short vol, potential to buy protection

        **Z-Score**: How many standard deviations from mean
        - Z > 2: Unusually high VRP
        - Z < -2: Unusually low VRP
        """
        )


def render_iv_vs_rv(symbol: str, available_dates: List[date]) -> None:
    """Render IV vs RV comparison."""
    st.subheader("Implied vs Realized Volatility")

    metrics = SurfaceMetrics()
    rv_calc = RealizedVolCalculator()

    start_date = available_dates[0]
    end_date = available_dates[-1]

    # Get IV (30-day ATM)
    iv_df = metrics.get_atm_vol_timeseries(symbol, start_date, end_date, tte_target_days=30)

    # Get RV (21-day)
    rv_df = rv_calc.get_vol_timeseries(symbol, start_date, end_date, metric="rv_21d")

    if iv_df.empty:
        st.warning("No IV data available.")
        return

    # Create comparison plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iv_df["date"],
            y=iv_df["atm_vol"] * 100,
            mode="lines",
            name="30-Day IV",
            line=dict(color="blue"),
        )
    )

    if not rv_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rv_df["date"],
                y=rv_df["realized_vol"] * 100,
                mode="lines",
                name="21-Day RV",
                line=dict(color="orange"),
            )
        )

    fig.update_layout(
        title=f"{symbol} Implied vs Realized Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Current values
    col1, col2, col3 = st.columns(3)
    with col1:
        if not iv_df.empty:
            st.metric("Current IV (30d)", f"{iv_df['atm_vol'].iloc[-1]:.2%}")
    with col2:
        if not rv_df.empty:
            st.metric("Current RV (21d)", f"{rv_df['realized_vol'].iloc[-1]:.2%}")
    with col3:
        if not iv_df.empty and not rv_df.empty:
            vrp = iv_df["atm_vol"].iloc[-1] - rv_df["realized_vol"].iloc[-1]
            st.metric("Current VRP", f"{vrp:.2%}")


def render_realized_vol(symbol: str, available_dates: List[date]) -> None:
    """Render realized volatility analysis."""
    st.subheader("Realized Volatility Analysis")

    calculator = RealizedVolCalculator()

    # Get latest RV data
    latest_date = available_dates[-1]
    result = calculator.calculate_for_date(symbol, latest_date)

    # Display current RV metrics
    st.markdown(f"**Realized Volatility as of {latest_date}:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = f"{result.rv_10d:.2%}" if result.rv_10d else "N/A"
        st.metric("10-Day RV", val)
    with col2:
        val = f"{result.rv_21d:.2%}" if result.rv_21d else "N/A"
        st.metric("21-Day RV", val)
    with col3:
        val = f"{result.rv_63d:.2%}" if result.rv_63d else "N/A"
        st.metric("63-Day RV", val)
    with col4:
        val = f"{result.rv_252d:.2%}" if result.rv_252d else "N/A"
        st.metric("252-Day RV", val)

    # RV estimators comparison
    st.markdown("---")
    st.markdown("**Alternative Estimators (21-Day):**")

    col1, col2, col3 = st.columns(3)
    with col1:
        val = f"{result.rv_21d:.2%}" if result.rv_21d else "N/A"
        st.metric("Close-to-Close", val)
    with col2:
        val = f"{result.parkinson_21d:.2%}" if result.parkinson_21d else "N/A"
        st.metric("Parkinson (H-L)", val)
    with col3:
        val = f"{result.gk_21d:.2%}" if result.gk_21d else "N/A"
        st.metric("Garman-Klass", val)

    # RV time series
    st.markdown("---")
    st.subheader("Realized Vol Time Series")

    start_date = available_dates[0]
    end_date = available_dates[-1]

    col1, col2 = st.columns([1, 3])
    with col1:
        window = st.selectbox(
            "Window",
            options=["rv_10d", "rv_21d", "rv_63d"],
            format_func=lambda x: {"rv_10d": "10-Day", "rv_21d": "21-Day", "rv_63d": "63-Day"}[x],
            index=1,
        )

    df = calculator.get_vol_timeseries(symbol, start_date, end_date, metric=window)

    if df.empty:
        st.warning("No realized vol data available. Need more historical underlying prices.")
        st.code(f"volsurf ingest backfill {symbol} --start 2025-11-01 --end {end_date}")
        return

    fig = px.line(
        df,
        x="date",
        y="realized_vol",
        title=f"{symbol} Realized Volatility ({window.replace('rv_', '').replace('d', '-Day')})",
        labels={"realized_vol": "RV", "date": "Date"},
    )

    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # Estimator explanation
    with st.expander("Realized Vol Estimators"):
        st.markdown(
            """
        **Close-to-Close (Standard)**
        - Uses daily log returns: `r = ln(Close_t / Close_{t-1})`
        - RV = sqrt(252 * mean(r^2))
        - Most common estimator

        **Parkinson (High-Low)**
        - Uses intraday range: `RV = sqrt(252 / (4 * ln(2)) * mean(ln(H/L)^2))`
        - ~5x more efficient than close-to-close
        - Underestimates vol if there are jumps

        **Garman-Klass (OHLC)**
        - Uses all of Open, High, Low, Close
        - Most efficient estimator
        - `RV = sqrt(252 * mean(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))`
        """
        )
