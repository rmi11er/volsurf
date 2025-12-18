"""Surface Viewer page for Streamlit dashboard."""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics import SurfaceMetrics, get_vol_at_moneyness
from volsurf.database.connection import get_connection


def render_surface_viewer(symbol: str, selected_date: date) -> None:
    """Render the surface viewer page."""
    # Get fitted surfaces for this date
    conn = get_connection()
    surfaces_df = conn.execute(
        """
        SELECT
            expiration_date,
            tte_years,
            svi_a, svi_b, svi_rho, svi_m, svi_sigma,
            atm_vol, skew_25delta,
            rmse, num_points
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ?
        ORDER BY expiration_date
        """,
        [symbol, selected_date],
    ).fetchdf()

    if surfaces_df.empty:
        st.warning(f"No fitted surfaces found for {symbol} on {selected_date}")
        st.info("Run surface fitting first: `volsurf fit surfaces SPY --date <date>`")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["3D Surface", "Vol Smile", "Surface Parameters"])

    with tab1:
        render_3d_surface(symbol, selected_date, surfaces_df)

    with tab2:
        render_vol_smile(symbol, surfaces_df)

    with tab3:
        render_surface_params(surfaces_df)


def render_3d_surface(symbol: str, selected_date: date, surfaces_df: pd.DataFrame) -> None:
    """Render 3D volatility surface plot."""
    st.subheader("3D Volatility Surface")

    # Create grid for surface
    moneyness_range = np.linspace(-0.3, 0.3, 50)  # log-moneyness
    tte_values = surfaces_df["tte_years"].values

    # Build IV surface
    iv_surface = np.zeros((len(tte_values), len(moneyness_range)))

    for i, row in surfaces_df.iterrows():
        for j, k in enumerate(moneyness_range):
            iv = get_vol_at_moneyness(
                row["svi_a"],
                row["svi_b"],
                row["svi_rho"],
                row["svi_m"],
                row["svi_sigma"],
                row["tte_years"],
                k,
            )
            iv_surface[i, j] = iv if not np.isnan(iv) else 0

    # Create 3D surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=moneyness_range,
                y=tte_values * 365,  # Convert to days
                z=iv_surface * 100,  # Convert to percentage
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
            )
        ]
    )

    fig.update_layout(
        title=f"{symbol} Implied Volatility Surface ({selected_date})",
        scene=dict(
            xaxis_title="Log-Moneyness",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Vol (%)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        width=800,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show ATM term structure on the side
    st.markdown("**ATM Volatility by Expiration:**")
    atm_data = surfaces_df[["expiration_date", "tte_years", "atm_vol"]].copy()
    atm_data["dte"] = (atm_data["tte_years"] * 365).round(0).astype(int)
    atm_data["atm_vol_pct"] = (atm_data["atm_vol"] * 100).round(2)
    st.dataframe(
        atm_data[["expiration_date", "dte", "atm_vol_pct"]].rename(
            columns={
                "expiration_date": "Expiration",
                "dte": "DTE",
                "atm_vol_pct": "ATM Vol (%)",
            }
        ),
        hide_index=True,
    )


def render_vol_smile(symbol: str, surfaces_df: pd.DataFrame) -> None:
    """Render volatility smile plots for selected expirations."""
    st.subheader("Volatility Smile by Expiration")

    # Let user select expirations to display
    expirations = surfaces_df["expiration_date"].tolist()
    selected_exps = st.multiselect(
        "Select Expirations",
        options=expirations,
        default=expirations[:3] if len(expirations) >= 3 else expirations,
        format_func=lambda x: f"{x} ({int(surfaces_df[surfaces_df['expiration_date'] == x]['tte_years'].values[0] * 365)}d)",
    )

    if not selected_exps:
        st.info("Select at least one expiration to view the smile.")
        return

    # Generate smile plots
    moneyness_range = np.linspace(-0.25, 0.25, 100)
    fig = go.Figure()

    for exp in selected_exps:
        row = surfaces_df[surfaces_df["expiration_date"] == exp].iloc[0]

        ivs = [
            get_vol_at_moneyness(
                row["svi_a"],
                row["svi_b"],
                row["svi_rho"],
                row["svi_m"],
                row["svi_sigma"],
                row["tte_years"],
                k,
            )
            for k in moneyness_range
        ]

        dte = int(row["tte_years"] * 365)
        fig.add_trace(
            go.Scatter(
                x=moneyness_range,
                y=[iv * 100 for iv in ivs],
                mode="lines",
                name=f"{exp} ({dte}d)",
                hovertemplate="Moneyness: %{x:.3f}<br>IV: %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Fitted Volatility Smiles",
        xaxis_title="Log-Moneyness (k = ln(K/F))",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show skew metrics
    st.markdown("**Skew Metrics:**")
    skew_data = surfaces_df[surfaces_df["expiration_date"].isin(selected_exps)][
        ["expiration_date", "tte_years", "atm_vol", "skew_25delta", "svi_rho"]
    ].copy()
    skew_data["dte"] = (skew_data["tte_years"] * 365).round(0).astype(int)
    skew_data["atm_vol_pct"] = (skew_data["atm_vol"] * 100).round(2)
    skew_data["skew_pct"] = (skew_data["skew_25delta"] * 100).round(2) if "skew_25delta" in skew_data.columns else None

    display_cols = ["expiration_date", "dte", "atm_vol_pct", "svi_rho"]
    if skew_data["skew_pct"].notna().any():
        display_cols.append("skew_pct")

    st.dataframe(
        skew_data[display_cols].rename(
            columns={
                "expiration_date": "Expiration",
                "dte": "DTE",
                "atm_vol_pct": "ATM Vol (%)",
                "svi_rho": "Rho (Skew)",
                "skew_pct": "25D Skew (%)",
            }
        ),
        hide_index=True,
    )


def render_surface_params(surfaces_df: pd.DataFrame) -> None:
    """Render table of SVI parameters."""
    st.subheader("SVI Parameters by Expiration")

    # Format the dataframe
    display_df = surfaces_df.copy()
    display_df["dte"] = (display_df["tte_years"] * 365).round(0).astype(int)
    display_df["atm_vol_pct"] = (display_df["atm_vol"] * 100).round(2)
    display_df["rmse_pct"] = (display_df["rmse"] * 100).round(4)

    # Round SVI params
    for col in ["svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma"]:
        display_df[col] = display_df[col].round(6)

    st.dataframe(
        display_df[
            [
                "expiration_date",
                "dte",
                "svi_a",
                "svi_b",
                "svi_rho",
                "svi_m",
                "svi_sigma",
                "atm_vol_pct",
                "rmse_pct",
                "num_points",
            ]
        ].rename(
            columns={
                "expiration_date": "Expiration",
                "dte": "DTE",
                "svi_a": "a",
                "svi_b": "b",
                "svi_rho": "rho",
                "svi_m": "m",
                "svi_sigma": "sigma",
                "atm_vol_pct": "ATM Vol (%)",
                "rmse_pct": "RMSE (%)",
                "num_points": "Points",
            }
        ),
        hide_index=True,
    )

    # Parameter explanation
    with st.expander("SVI Parameter Interpretation"):
        st.markdown(
            """
        **SVI (Stochastic Volatility Inspired) Model:**

        Total variance: `w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))`

        Where `k = ln(K/F)` is log-moneyness.

        **Parameters:**
        - **a**: Vertical shift (ATM variance level)
        - **b**: Slope of the wings (overall smile curvature)
        - **rho**: Skew parameter (-1 to 1). Negative = put skew
        - **m**: Horizontal shift (ATM log-moneyness offset)
        - **sigma**: Smoothness/curvature at ATM

        **Fit Quality:**
        - **RMSE**: Root mean squared error of fit (lower = better)
        - **Points**: Number of liquid strikes used in calibration
        """
        )
