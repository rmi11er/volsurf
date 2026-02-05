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
    tabs = st.tabs([
        "Current Term Structure",
        "ATM Vol Evolution",
        "Skew Evolution",
        "SVI Parameters",
    ])

    with tabs[0]:
        render_current_term_structure(symbol, selected_date)

    with tabs[1]:
        render_atm_evolution(symbol, available_dates)

    with tabs[2]:
        render_skew_evolution(symbol, available_dates)

    with tabs[3]:
        render_svi_params(symbol, selected_date, available_dates)


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

    st.plotly_chart(fig, width="stretch")

    # Show fit parameters
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Term Structure Data:**")
        display_df = df.copy()
        display_df["atm_vol_pct"] = (display_df["atm_vol"] * 100).round(2)
        if "skew_25delta" in display_df.columns:
            display_df["skew_pct"] = (display_df["skew_25delta"] * 100).round(2)
            show_cols = ["expiration", "tte_days", "atm_vol_pct", "skew_pct"]
        else:
            show_cols = ["expiration", "tte_days", "atm_vol_pct"]

        st.dataframe(
            display_df[show_cols].rename(
                columns={
                    "expiration": "Expiration",
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

    # Create time series plot with percentile bands
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["atm_vol"] * 100,
            mode="lines", name="ATM Vol",
            line=dict(color="#1976d2"),
        )
    )

    # 10th / 90th percentile bands from the displayed window
    p10 = np.percentile(df["atm_vol"].dropna(), 10) * 100
    p90 = np.percentile(df["atm_vol"].dropna(), 90) * 100

    fig.add_hline(y=p10, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="10th pctile", annotation_position="bottom left")
    fig.add_hline(y=p90, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text="90th pctile", annotation_position="top left")

    fig.update_layout(
        title=f"{symbol} {tenor}-Day ATM Implied Volatility",
        xaxis_title="Date", yaxis_title="ATM Vol (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

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

    # Create time series plot with percentile bands
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["skew"] * 100,
            mode="lines", name="Skew",
            line=dict(color="#1976d2"),
        )
    )

    # 10th / 90th percentile bands
    skew_clean = df["skew"].dropna()
    if len(skew_clean) > 10:
        p10 = np.percentile(skew_clean, 10) * 100
        p90 = np.percentile(skew_clean, 90) * 100
        fig.add_hline(y=p10, line_dash="dot", line_color="green", opacity=0.5,
                      annotation_text="10th pctile", annotation_position="bottom left")
        fig.add_hline(y=p90, line_dash="dot", line_color="red", opacity=0.5,
                      annotation_text="90th pctile", annotation_position="top left")

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"{symbol} {tenor}-Day 25-Delta Skew",
        xaxis_title="Date", yaxis_title="Skew (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

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


def render_svi_params(symbol: str, selected_date: date, available_dates: List[date]) -> None:
    """Render SVI model parameters analysis."""
    st.subheader("SVI Model Parameters")

    view_mode = st.radio(
        "View Mode",
        ["Term Structure (by expiration)", "Time Series (by tenor)"],
        horizontal=True,
    )

    metrics = SurfaceMetrics()

    if view_mode == "Term Structure (by expiration)":
        render_svi_term_structure(symbol, selected_date, metrics)
    else:
        render_svi_timeseries(symbol, available_dates, metrics)


def render_svi_term_structure(symbol: str, selected_date: date, metrics: SurfaceMetrics) -> None:
    """Render SVI parameters across expirations for a single date."""
    st.markdown(f"**SVI Parameters by Expiration on {selected_date}**")

    df = metrics.get_svi_params_term_structure(symbol, selected_date)

    if df.empty:
        st.warning("No SVI parameter data available for this date.")
        return

    # Add tte_days column from tte_years
    df["tte_days"] = df["tte_years"] * 252

    # Rename columns for convenience
    df["a"] = df["svi_a"]
    df["b"] = df["svi_b"]
    df["rho"] = df["svi_rho"]
    df["m"] = df["svi_m"]
    df["sigma"] = df["svi_sigma"]

    # Create multi-panel plot for each SVI parameter
    params = ["a", "b", "rho", "m", "sigma"]
    param_descriptions = {
        "a": "Level (vertical shift)",
        "b": "Angle (ATM slope)",
        "rho": "Rotation (skew direction)",
        "m": "Translation (smile center)",
        "sigma": "Curvature (smile width)",
    }

    # Create subplot with 5 rows
    fig = go.Figure()

    for i, param in enumerate(params):
        fig.add_trace(
            go.Scatter(
                x=df["tte_days"],
                y=df[param],
                mode="markers+lines",
                name=f"{param} - {param_descriptions[param]}",
                hovertemplate=f"DTE: %{{x:.0f}}<br>{param}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{symbol} SVI Parameters Term Structure",
        xaxis_title="Days to Expiry",
        yaxis_title="Parameter Value",
        hovermode="x unified",
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0, orientation="h"),
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    # Individual parameter plots
    st.markdown("**Individual Parameter Term Structures:**")

    col1, col2 = st.columns(2)

    with col1:
        # Rho (skew direction) - most important for equity vol
        fig_rho = px.scatter(
            df,
            x="tte_days",
            y="rho",
            title="ρ (Rho) - Skew Direction",
            labels={"tte_days": "DTE", "rho": "ρ"},
        )
        fig_rho.update_traces(mode="markers+lines")
        fig_rho.update_layout(height=300)
        st.plotly_chart(fig_rho, width="stretch")

        # b (ATM slope)
        fig_b = px.scatter(
            df,
            x="tte_days",
            y="b",
            title="b - ATM Slope",
            labels={"tte_days": "DTE", "b": "b"},
        )
        fig_b.update_traces(mode="markers+lines")
        fig_b.update_layout(height=300)
        st.plotly_chart(fig_b, width="stretch")

    with col2:
        # sigma (curvature)
        fig_sigma = px.scatter(
            df,
            x="tte_days",
            y="sigma",
            title="σ (Sigma) - Smile Curvature",
            labels={"tte_days": "DTE", "sigma": "σ"},
        )
        fig_sigma.update_traces(mode="markers+lines")
        fig_sigma.update_layout(height=300)
        st.plotly_chart(fig_sigma, width="stretch")

        # m (translation)
        fig_m = px.scatter(
            df,
            x="tte_days",
            y="m",
            title="m - Smile Center",
            labels={"tte_days": "DTE", "m": "m"},
        )
        fig_m.update_traces(mode="markers+lines")
        fig_m.update_layout(height=300)
        st.plotly_chart(fig_m, width="stretch")

    # Data table
    with st.expander("View Raw Data"):
        display_df = df.copy()
        display_df = display_df.round(6)
        st.dataframe(display_df, hide_index=True)

    # Parameter interpretation
    with st.expander("SVI Parameter Interpretation"):
        st.markdown(
            """
        **SVI (Stochastic Volatility Inspired) Model:**

        `w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))`

        Where `w` is total implied variance and `k` is log-moneyness.

        **Parameters:**
        - **a**: Overall variance level (vertical shift)
        - **b**: Tightness/angle - controls ATM slope and wing behavior
        - **ρ (rho)**: Rotation - controls skew direction (-1 to 1, negative = put skew)
        - **m**: Translation - horizontal shift of minimum variance
        - **σ (sigma)**: Smoothness - controls curvature at the bottom of the smile

        **Typical Equity Values:**
        - ρ is usually negative (-0.3 to -0.9) reflecting put skew
        - σ increases with tenor (smiles flatten)
        - b decreases with tenor (less curvature in long-dated options)
        """
        )


def render_svi_timeseries(symbol: str, available_dates: List[date], metrics: SurfaceMetrics) -> None:
    """Render SVI parameters time series for a specific tenor."""
    col1, col2 = st.columns([1, 3])

    with col1:
        tenor = st.selectbox(
            "Select Tenor (days)",
            options=[7, 14, 21, 30, 45, 60, 90, 180],
            index=3,
            key="svi_tenor",
        )

    start_date = available_dates[0]
    end_date = available_dates[-1]

    df = metrics.get_svi_params_timeseries(symbol, start_date, end_date, tte_target_days=tenor)

    if df.empty:
        st.warning("No SVI parameter data available for the selected tenor.")
        return

    # Rename columns for convenience
    df["a"] = df["svi_a"]
    df["b"] = df["svi_b"]
    df["rho"] = df["svi_rho"]
    df["m"] = df["svi_m"]
    df["sigma"] = df["svi_sigma"]

    st.markdown(f"**SVI Parameters Evolution ({tenor}-Day Tenor)**")

    # Create subplots for each parameter
    params = ["a", "b", "rho", "m", "sigma"]
    param_titles = {
        "a": "a - Level",
        "b": "b - ATM Slope",
        "rho": "ρ - Skew Direction",
        "m": "m - Smile Center",
        "sigma": "σ - Curvature",
    }

    # Combined plot
    fig = go.Figure()
    for param in params:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[param],
                mode="lines",
                name=param_titles[param],
            )
        )

    fig.update_layout(
        title=f"{symbol} {tenor}-Day SVI Parameters Over Time",
        xaxis_title="Date",
        yaxis_title="Parameter Value",
        hovermode="x unified",
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0, orientation="h"),
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    # Individual plots in 2x3 grid
    st.markdown("**Individual Parameter Time Series:**")

    # Row 1: rho and b (most important)
    col1, col2 = st.columns(2)

    with col1:
        fig_rho = px.line(
            df,
            x="date",
            y="rho",
            title="ρ (Rho) - Skew Direction",
            labels={"date": "Date", "rho": "ρ"},
        )
        fig_rho.update_layout(height=250, hovermode="x unified")
        fig_rho.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_rho, width="stretch")

        # Stats for rho
        st.caption(
            f"Current: {df['rho'].iloc[-1]:.4f} | "
            f"Mean: {df['rho'].mean():.4f} | "
            f"Range: [{df['rho'].min():.4f}, {df['rho'].max():.4f}]"
        )

    with col2:
        fig_b = px.line(
            df,
            x="date",
            y="b",
            title="b - ATM Slope",
            labels={"date": "Date", "b": "b"},
        )
        fig_b.update_layout(height=250, hovermode="x unified")
        st.plotly_chart(fig_b, width="stretch")

        st.caption(
            f"Current: {df['b'].iloc[-1]:.4f} | "
            f"Mean: {df['b'].mean():.4f} | "
            f"Range: [{df['b'].min():.4f}, {df['b'].max():.4f}]"
        )

    # Row 2: sigma and m
    col1, col2 = st.columns(2)

    with col1:
        fig_sigma = px.line(
            df,
            x="date",
            y="sigma",
            title="σ (Sigma) - Smile Curvature",
            labels={"date": "Date", "sigma": "σ"},
        )
        fig_sigma.update_layout(height=250, hovermode="x unified")
        st.plotly_chart(fig_sigma, width="stretch")

        st.caption(
            f"Current: {df['sigma'].iloc[-1]:.4f} | "
            f"Mean: {df['sigma'].mean():.4f} | "
            f"Range: [{df['sigma'].min():.4f}, {df['sigma'].max():.4f}]"
        )

    with col2:
        fig_m = px.line(
            df,
            x="date",
            y="m",
            title="m - Smile Center",
            labels={"date": "Date", "m": "m"},
        )
        fig_m.update_layout(height=250, hovermode="x unified")
        fig_m.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_m, width="stretch")

        st.caption(
            f"Current: {df['m'].iloc[-1]:.4f} | "
            f"Mean: {df['m'].mean():.4f} | "
            f"Range: [{df['m'].min():.4f}, {df['m'].max():.4f}]"
        )

    # Row 3: a (level)
    fig_a = px.line(
        df,
        x="date",
        y="a",
        title="a - Variance Level",
        labels={"date": "Date", "a": "a"},
    )
    fig_a.update_layout(height=250, hovermode="x unified")
    st.plotly_chart(fig_a, width="stretch")

    st.caption(
        f"Current: {df['a'].iloc[-1]:.4f} | "
        f"Mean: {df['a'].mean():.4f} | "
        f"Range: [{df['a'].min():.4f}, {df['a'].max():.4f}]"
    )

    # Data table
    with st.expander("View Raw Data"):
        display_df = df.copy()
        display_df = display_df.round(6)
        st.dataframe(display_df, hide_index=True)
