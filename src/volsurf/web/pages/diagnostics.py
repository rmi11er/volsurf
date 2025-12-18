"""Model Diagnostics page for Streamlit dashboard."""

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics import get_vol_at_moneyness
from volsurf.database.connection import get_connection


def render_diagnostics(symbol: str, selected_date: date) -> None:
    """Render the model diagnostics page."""
    tabs = st.tabs(["Fit Quality", "Residuals", "Arbitrage Checks", "Data Coverage"])

    with tabs[0]:
        render_fit_quality(symbol, selected_date)

    with tabs[1]:
        render_residuals(symbol, selected_date)

    with tabs[2]:
        render_arbitrage_checks(symbol, selected_date)

    with tabs[3]:
        render_data_coverage(symbol)


def render_fit_quality(symbol: str, selected_date: date) -> None:
    """Render fit quality metrics."""
    st.subheader("Surface Fit Quality")

    conn = get_connection()

    # Get fit quality metrics
    df = conn.execute(
        """
        SELECT
            expiration_date,
            tte_years * 365 as dte,
            rmse,
            mae,
            max_error,
            num_points,
            atm_vol,
            passes_no_arbitrage,
            butterfly_arbitrage_violations,
            calendar_arbitrage_violations
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ?
        ORDER BY expiration_date
        """,
        [symbol, selected_date],
    ).fetchdf()

    if df.empty:
        st.warning(f"No fitted surfaces found for {symbol} on {selected_date}")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_rmse = df["rmse"].mean()
        st.metric("Avg RMSE", f"{avg_rmse:.4%}")
    with col2:
        max_rmse = df["rmse"].max()
        st.metric("Max RMSE", f"{max_rmse:.4%}")
    with col3:
        total_points = df["num_points"].sum()
        st.metric("Total Points", f"{total_points:,}")
    with col4:
        arb_free = df["passes_no_arbitrage"].sum() if "passes_no_arbitrage" in df.columns else len(df)
        st.metric("Arb-Free Fits", f"{arb_free}/{len(df)}")

    # RMSE by expiration
    st.markdown("---")
    st.markdown("**RMSE by Expiration:**")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["expiration_date"].astype(str),
            y=df["rmse"] * 100,
            name="RMSE",
            marker_color="steelblue",
            hovertemplate="Exp: %{x}<br>RMSE: %{y:.4f}%<extra></extra>",
        )
    )

    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="0.5% Target")

    fig.update_layout(
        title="Fit RMSE by Expiration",
        xaxis_title="Expiration Date",
        yaxis_title="RMSE (%)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown("**Fit Details:**")

    display_df = df.copy()
    display_df["dte"] = display_df["dte"].round(0).astype(int)
    display_df["rmse_pct"] = (display_df["rmse"] * 100).round(4)
    display_df["mae_pct"] = (display_df["mae"] * 100).round(4) if "mae" in display_df.columns else None
    display_df["max_err_pct"] = (display_df["max_error"] * 100).round(4) if "max_error" in display_df.columns else None
    display_df["atm_vol_pct"] = (display_df["atm_vol"] * 100).round(2)

    cols = ["expiration_date", "dte", "num_points", "rmse_pct", "atm_vol_pct"]
    if display_df["mae_pct"].notna().any():
        cols.append("mae_pct")
    if display_df["max_err_pct"].notna().any():
        cols.append("max_err_pct")

    st.dataframe(
        display_df[cols].rename(
            columns={
                "expiration_date": "Expiration",
                "dte": "DTE",
                "num_points": "Points",
                "rmse_pct": "RMSE (%)",
                "mae_pct": "MAE (%)",
                "max_err_pct": "Max Err (%)",
                "atm_vol_pct": "ATM Vol (%)",
            }
        ),
        hide_index=True,
    )


def render_residuals(symbol: str, selected_date: date) -> None:
    """Render fit residuals analysis."""
    st.subheader("Fit Residuals")

    conn = get_connection()

    # Get fitted surfaces
    surfaces_df = conn.execute(
        """
        SELECT expiration_date, tte_years, svi_a, svi_b, svi_rho, svi_m, svi_sigma
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ?
        ORDER BY expiration_date
        """,
        [symbol, selected_date],
    ).fetchdf()

    if surfaces_df.empty:
        st.warning("No fitted surfaces found.")
        return

    # Get raw options data for comparison
    raw_df = conn.execute(
        """
        SELECT
            expiration_date,
            strike,
            option_type,
            implied_volatility,
            underlying_price
        FROM raw_options_chains
        WHERE symbol = ? AND quote_date = ? AND is_liquid = TRUE
        ORDER BY expiration_date, strike
        """,
        [symbol, selected_date],
    ).fetchdf()

    if raw_df.empty:
        st.warning("No raw options data found for residual analysis.")
        return

    # Select expiration for detailed view
    expirations = surfaces_df["expiration_date"].tolist()
    selected_exp = st.selectbox("Select Expiration", options=expirations)

    # Filter data
    exp_surface = surfaces_df[surfaces_df["expiration_date"] == selected_exp].iloc[0]
    exp_raw = raw_df[raw_df["expiration_date"] == selected_exp]

    if exp_raw.empty:
        st.warning("No raw data for selected expiration.")
        return

    # Calculate residuals
    underlying = exp_raw["underlying_price"].iloc[0]
    residuals = []

    for _, row in exp_raw.iterrows():
        if pd.isna(row["implied_volatility"]) or row["implied_volatility"] <= 0:
            continue

        k = np.log(row["strike"] / underlying)  # log-moneyness
        fitted_vol = get_vol_at_moneyness(
            exp_surface["svi_a"],
            exp_surface["svi_b"],
            exp_surface["svi_rho"],
            exp_surface["svi_m"],
            exp_surface["svi_sigma"],
            exp_surface["tte_years"],
            k,
        )

        if not np.isnan(fitted_vol):
            residuals.append(
                {
                    "strike": row["strike"],
                    "moneyness": k,
                    "market_iv": row["implied_volatility"],
                    "fitted_iv": fitted_vol,
                    "residual": row["implied_volatility"] - fitted_vol,
                    "option_type": row["option_type"],
                }
            )

    if not residuals:
        st.warning("Could not calculate residuals.")
        return

    residuals_df = pd.DataFrame(residuals)

    # Residuals plot
    fig = go.Figure()

    for opt_type in ["CALL", "PUT"]:
        type_df = residuals_df[residuals_df["option_type"] == opt_type]
        if not type_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=type_df["moneyness"],
                    y=type_df["residual"] * 100,
                    mode="markers",
                    name=opt_type,
                    hovertemplate="K=%{customdata:.0f}<br>Residual=%{y:.3f}%<extra></extra>",
                    customdata=type_df["strike"],
                )
            )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Fit Residuals (Market IV - Fitted IV) for {selected_exp}",
        xaxis_title="Log-Moneyness",
        yaxis_title="Residual (%)",
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Residual statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Residual", f"{residuals_df['residual'].mean():.4%}")
    with col2:
        st.metric("Std Residual", f"{residuals_df['residual'].std():.4%}")
    with col3:
        st.metric("Max Residual", f"{residuals_df['residual'].abs().max():.4%}")
    with col4:
        st.metric("Points", len(residuals_df))

    # Market vs Fitted comparison
    st.markdown("---")
    st.markdown("**Market vs Fitted IV:**")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=residuals_df["strike"],
            y=residuals_df["market_iv"] * 100,
            mode="markers",
            name="Market IV",
            marker=dict(symbol="circle", size=8),
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=residuals_df["strike"],
            y=residuals_df["fitted_iv"] * 100,
            mode="lines",
            name="Fitted IV (SVI)",
            line=dict(color="red"),
        )
    )

    fig2.update_layout(
        title="Market IV vs Fitted SVI",
        xaxis_title="Strike",
        yaxis_title="Implied Vol (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig2, use_container_width=True)


def render_arbitrage_checks(symbol: str, selected_date: date) -> None:
    """Render arbitrage violation analysis."""
    st.subheader("Arbitrage Analysis")

    conn = get_connection()

    df = conn.execute(
        """
        SELECT
            expiration_date,
            tte_years * 365 as dte,
            passes_no_arbitrage,
            butterfly_arbitrage_violations,
            calendar_arbitrage_violations,
            svi_a, svi_b, svi_rho, svi_sigma
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ?
        ORDER BY expiration_date
        """,
        [symbol, selected_date],
    ).fetchdf()

    if df.empty:
        st.warning("No fitted surfaces found.")
        return

    # Summary
    total = len(df)
    arb_free = df["passes_no_arbitrage"].sum() if df["passes_no_arbitrage"].notna().any() else total
    butterfly_violations = df["butterfly_arbitrage_violations"].sum() if df["butterfly_arbitrage_violations"].notna().any() else 0
    calendar_violations = df["calendar_arbitrage_violations"].sum() if df["calendar_arbitrage_violations"].notna().any() else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "green" if arb_free == total else "red"
        st.markdown(f"**Arbitrage-Free Surfaces:** :{color}[{arb_free}/{total}]")
    with col2:
        st.metric("Butterfly Violations", int(butterfly_violations))
    with col3:
        st.metric("Calendar Violations", int(calendar_violations))

    # Arbitrage conditions explanation
    st.markdown("---")
    st.markdown("**SVI No-Arbitrage Conditions:**")

    conditions_df = df.copy()
    conditions_df["dte"] = conditions_df["dte"].round(0).astype(int)

    # Calculate conditions
    conditions_df["b_positive"] = conditions_df["svi_b"] >= 0
    conditions_df["variance_positive"] = (
        conditions_df["svi_a"]
        + conditions_df["svi_b"] * conditions_df["svi_sigma"] * np.sqrt(1 - conditions_df["svi_rho"] ** 2)
    ) >= 0
    conditions_df["rho_valid"] = conditions_df["svi_rho"].abs() < 1

    st.dataframe(
        conditions_df[
            ["expiration_date", "dte", "b_positive", "variance_positive", "rho_valid", "passes_no_arbitrage"]
        ].rename(
            columns={
                "expiration_date": "Expiration",
                "dte": "DTE",
                "b_positive": "b >= 0",
                "variance_positive": "Min Var >= 0",
                "rho_valid": "|rho| < 1",
                "passes_no_arbitrage": "All Pass",
            }
        ),
        hide_index=True,
    )

    # Explanation
    with st.expander("Arbitrage Conditions Explained"):
        st.markdown(
            """
        **SVI No-Arbitrage Conditions:**

        1. **b >= 0**: Ensures no calendar arbitrage (variance increases with time)

        2. **a + b * sigma * sqrt(1 - rho^2) >= 0**: Ensures non-negative total variance
           at all strikes (minimum variance condition)

        3. **|rho| < 1**: Valid correlation parameter

        **Butterfly Arbitrage:**
        - Occurs when the vol smile is not convex
        - Implies negative probability density
        - Detected by checking second derivative of total variance

        **Calendar Arbitrage:**
        - Occurs when forward variance is negative
        - Longer-dated options should have >= total variance
        - Ensured by b >= 0 in SVI
        """
        )


def render_data_coverage(symbol: str) -> None:
    """Render data coverage analysis."""
    st.subheader("Data Coverage Analysis")

    conn = get_connection()

    # Get date coverage
    coverage_df = conn.execute(
        """
        SELECT
            quote_date,
            COUNT(*) as total_options,
            SUM(CASE WHEN is_liquid THEN 1 ELSE 0 END) as liquid_options,
            COUNT(DISTINCT expiration_date) as num_expirations,
            AVG(underlying_price) as underlying_price
        FROM raw_options_chains
        WHERE symbol = ?
        GROUP BY quote_date
        ORDER BY quote_date
        """,
        [symbol],
    ).fetchdf()

    if coverage_df.empty:
        st.warning(f"No data found for {symbol}")
        return

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trading Days", len(coverage_df))
    with col2:
        date_range = f"{coverage_df['quote_date'].min()} to {coverage_df['quote_date'].max()}"
        st.markdown(f"**Date Range:**\n{date_range}")
    with col3:
        st.metric("Avg Options/Day", f"{coverage_df['total_options'].mean():,.0f}")
    with col4:
        st.metric("Avg Liquid/Day", f"{coverage_df['liquid_options'].mean():,.0f}")

    # Options count over time
    st.markdown("---")
    st.markdown("**Options Count Over Time:**")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=coverage_df["quote_date"],
            y=coverage_df["total_options"],
            name="Total Options",
            marker_color="lightblue",
        )
    )

    fig.add_trace(
        go.Bar(
            x=coverage_df["quote_date"],
            y=coverage_df["liquid_options"],
            name="Liquid Options",
            marker_color="steelblue",
        )
    )

    fig.update_layout(
        title=f"{symbol} Options Data Coverage",
        xaxis_title="Date",
        yaxis_title="Number of Options",
        barmode="overlay",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Expirations coverage
    st.markdown("**Expirations per Day:**")

    fig2 = px.line(
        coverage_df,
        x="quote_date",
        y="num_expirations",
        title="Number of Expirations Over Time",
        labels={"num_expirations": "Expirations", "quote_date": "Date"},
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Fitted surfaces coverage
    st.markdown("---")
    st.markdown("**Fitted Surfaces Coverage:**")

    fitted_df = conn.execute(
        """
        SELECT
            quote_date,
            COUNT(*) as num_surfaces,
            AVG(rmse) as avg_rmse,
            AVG(num_points) as avg_points
        FROM fitted_surfaces
        WHERE symbol = ?
        GROUP BY quote_date
        ORDER BY quote_date
        """,
        [symbol],
    ).fetchdf()

    if not fitted_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Days with Fits", len(fitted_df))
        with col2:
            st.metric("Avg Surfaces/Day", f"{fitted_df['num_surfaces'].mean():.1f}")
        with col3:
            st.metric("Avg RMSE", f"{fitted_df['avg_rmse'].mean():.4%}")
    else:
        st.info("No fitted surfaces yet. Run: `volsurf fit batch SPY --start <date> --end <date>`")
