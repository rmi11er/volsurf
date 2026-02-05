"""Intraday volatility surface viewer component."""

from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.models.schemas import IntradaySurfaceResult


def get_available_intraday_dates(data_dir: Path, symbol: str = "SPY") -> List[date]:
    """Get list of dates with intraday data available."""
    if not data_dir.exists():
        return []

    dates = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            # Check for daily aggregate or individual timestamp files
            has_data = (
                (d / f"{symbol}_daily.parquet").exists()
                or len(list(d.glob(f"{symbol}_*.parquet"))) > 0
            )
            if has_data:
                try:
                    dates.append(date.fromisoformat(d.name))
                except ValueError:
                    continue
    return dates


def get_available_timestamps(data_dir: Path, target_date: date, symbol: str = "SPY") -> List[time]:
    """Get available timestamps for a given date."""
    day_dir = data_dir / target_date.strftime("%Y-%m-%d")
    if not day_dir.exists():
        return []

    times = []
    for f in sorted(day_dir.glob(f"{symbol}_*.parquet")):
        # Parse time from filename like SPY_0935.parquet
        time_str = f.stem.split("_")[1]
        try:
            t = datetime.strptime(time_str, "%H%M").time()
            times.append(t)
        except ValueError:
            continue
    return times


def load_intraday_surface(
    data_dir: Path,
    target_date: date,
    target_time: time,
    symbol: str = "SPY",
) -> Optional[IntradaySurfaceResult]:
    """Load intraday surface from Parquet file."""
    filename = f"{symbol}_{target_time.strftime('%H%M')}.parquet"
    filepath = data_dir / target_date.strftime("%Y-%m-%d") / filename

    if not filepath.exists():
        return None

    try:
        return IntradaySurfaceResult.from_parquet(str(filepath))
    except Exception as e:
        st.error(f"Failed to load {filepath}: {e}")
        return None


def load_all_timestamps_for_date(
    data_dir: Path,
    target_date: date,
    symbol: str = "SPY",
) -> List[Tuple[time, IntradaySurfaceResult]]:
    """Load all intraday surfaces for a given date."""
    times = get_available_timestamps(data_dir, target_date, symbol)
    results = []

    for t in times:
        result = load_intraday_surface(data_dir, target_date, t, symbol)
        if result:
            results.append((t, result))

    return results


def render_intraday_viewer(symbol: str, data_dir: Path) -> None:
    """Render the intraday volatility viewer."""
    st.header("Intraday Volatility Surface Analysis")

    # Check for available data
    available_dates = get_available_intraday_dates(data_dir, symbol)

    if not available_dates:
        st.warning(f"No intraday data found for {symbol} in {data_dir}")
        st.code(f"volsurf intraday batch {symbol} -s 2024-12-16 -e 2024-12-20 -i 60 --mock")
        return

    # Sidebar controls
    st.sidebar.markdown("### Intraday Controls")

    # Date selection
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )

    # Granularity filter (this affects which files we show)
    granularity = st.sidebar.selectbox(
        "Time Granularity",
        options=["All Available", "Hourly", "30-min", "15-min"],
        index=0,
        help="Filter timestamps to show. Data must exist at the selected granularity.",
    )

    # Get available timestamps for this date
    all_times = get_available_timestamps(data_dir, selected_date, symbol)

    if not all_times:
        st.warning(f"No intraday data for {selected_date}")
        return

    # Filter times based on granularity
    if granularity == "Hourly":
        filtered_times = [t for t in all_times if t.minute == 35 or t.minute == 0]
    elif granularity == "30-min":
        filtered_times = [t for t in all_times if t.minute in [0, 5, 30, 35]]
    elif granularity == "15-min":
        filtered_times = [t for t in all_times if t.minute in [0, 5, 15, 20, 30, 35, 45, 50]]
    else:
        filtered_times = all_times

    if not filtered_times:
        filtered_times = all_times  # Fall back to all if filter too strict

    # Time slider
    time_options = [t.strftime("%H:%M") for t in filtered_times]
    selected_time_str = st.select_slider(
        "Time of Day (ET)",
        options=time_options,
        value=time_options[len(time_options) // 2],
    )
    selected_time = datetime.strptime(selected_time_str, "%H:%M").time()

    # Load data for selected timestamp
    result = load_intraday_surface(data_dir, selected_date, selected_time, symbol)

    if not result or not result.surfaces:
        st.warning(f"No surface data for {selected_date} at {selected_time_str}")
        return

    # Display timestamp info
    st.markdown(f"**Timestamp:** {selected_date} {selected_time_str} ET")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Underlying", f"${float(result.underlying_price):.2f}")

    with col2:
        if result.surfaces:
            # Get 30-day ATM vol (approximate)
            short_term = [s for s in result.surfaces if 20 <= int(s.tte_years * 365) <= 45]
            if short_term:
                atm_30d = short_term[0].atm_vol
                st.metric("ATM Vol (30d)", f"{atm_30d:.1%}")
            else:
                st.metric("ATM Vol (30d)", "N/A")

    with col3:
        st.metric("Expirations", len(result.surfaces))

    with col4:
        if result.avg_rmse:
            st.metric("Avg RMSE", f"{result.avg_rmse:.3%}")

    st.markdown("---")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["ATM Evolution", "Surface View", "Term Structure", "Spot-Vol Dynamics"])

    with tab1:
        render_atm_evolution(data_dir, selected_date, symbol, filtered_times)

    with tab2:
        render_surface_snapshot(result)

    with tab3:
        render_term_structure(result)

    with tab4:
        render_spot_vol_dynamics(data_dir, symbol, available_dates)


def render_atm_evolution(
    data_dir: Path,
    target_date: date,
    symbol: str,
    times: List[time],
) -> None:
    """Render ATM vol evolution throughout the day."""
    st.subheader("Intraday ATM Volatility Evolution")

    # Load all surfaces for the day
    data_rows = []

    for t in times:
        result = load_intraday_surface(data_dir, target_date, t, symbol)
        if result and result.surfaces:
            for s in result.surfaces:
                tte_days = int(s.tte_years * 365)
                data_rows.append({
                    "time": t.strftime("%H:%M"),
                    "expiration": str(s.expiration_date),
                    "tte_days": tte_days,
                    "atm_vol": s.atm_vol,
                    "skew": s.skew_25delta if s.skew_25delta else 0,
                    "underlying": float(result.underlying_price),
                })

    if not data_rows:
        st.info("No data to display")
        return

    df = pd.DataFrame(data_rows)

    # Filter to specific tenors
    tenor_selection = st.multiselect(
        "Select Tenors (DTE)",
        options=sorted(df["tte_days"].unique()),
        default=[d for d in [7, 30, 60, 90] if d in df["tte_days"].unique()][:3],
    )

    if not tenor_selection:
        tenor_selection = sorted(df["tte_days"].unique())[:3]

    df_filtered = df[df["tte_days"].isin(tenor_selection)]

    # ATM Vol chart
    fig = px.line(
        df_filtered,
        x="time",
        y="atm_vol",
        color="tte_days",
        title=f"Intraday ATM Vol - {target_date}",
        labels={"atm_vol": "ATM Vol", "time": "Time (ET)", "tte_days": "DTE"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(hovermode="x unified", legend_title="Days to Exp")
    st.plotly_chart(fig, use_container_width=True)

    # Underlying price chart
    underlying_df = df.drop_duplicates(subset=["time"])[["time", "underlying"]]
    fig2 = px.line(
        underlying_df,
        x="time",
        y="underlying",
        title=f"Underlying Price - {target_date}",
        labels={"underlying": "Price", "time": "Time (ET)"},
    )
    fig2.update_layout(hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)


def render_surface_snapshot(result: IntradaySurfaceResult) -> None:
    """Render 3D surface plot for a single timestamp."""
    st.subheader("Volatility Surface Snapshot")

    if not result.surfaces:
        st.info("No surface data")
        return

    # Build grid data for 3D surface
    import numpy as np

    # Create moneyness range
    moneyness = np.linspace(-0.3, 0.3, 50)  # log-moneyness

    # Get sorted surfaces by TTE
    surfaces = sorted(result.surfaces, key=lambda s: s.tte_years)
    ttes = [s.tte_years * 365 for s in surfaces]  # Convert to days

    # Build IV grid
    iv_grid = np.zeros((len(surfaces), len(moneyness)))

    for i, s in enumerate(surfaces):
        for j, k in enumerate(moneyness):
            iv_grid[i, j] = s.svi_params.implied_vol(k, s.tte_years)

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=moneyness * 100,  # Convert to percentage
        y=ttes,
        z=iv_grid * 100,  # Convert to percentage
        colorscale='Viridis',
        colorbar=dict(title="IV %"),
    )])

    fig.update_layout(
        title=f"Vol Surface at {result.timestamp.strftime('%H:%M')} ET",
        scene=dict(
            xaxis_title="Moneyness (%)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Vol (%)",
        ),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Also show smile curves
    st.subheader("Volatility Smiles")

    # Select a few key tenors
    tenor_indices = [0, len(surfaces) // 3, 2 * len(surfaces) // 3, -1]
    tenor_indices = [i for i in tenor_indices if 0 <= i < len(surfaces)]

    smile_data = []
    for idx in tenor_indices:
        s = surfaces[idx]
        tte_days = int(s.tte_years * 365)
        for k in moneyness:
            iv = s.svi_params.implied_vol(k, s.tte_years)
            smile_data.append({
                "moneyness": k * 100,
                "iv": iv * 100,
                "tenor": f"{tte_days}d",
            })

    smile_df = pd.DataFrame(smile_data)

    fig2 = px.line(
        smile_df,
        x="moneyness",
        y="iv",
        color="tenor",
        title="Volatility Smiles by Tenor",
        labels={"moneyness": "Log-Moneyness (%)", "iv": "Implied Vol (%)", "tenor": "Tenor"},
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_term_structure(result: IntradaySurfaceResult) -> None:
    """Render ATM term structure for a single timestamp."""
    st.subheader("ATM Term Structure")

    if not result.surfaces:
        st.info("No surface data")
        return

    # Build term structure data
    ts_data = []
    for s in sorted(result.surfaces, key=lambda x: x.tte_years):
        tte_days = int(s.tte_years * 365)
        ts_data.append({
            "tte_days": tte_days,
            "atm_vol": s.atm_vol * 100,
            "skew": (s.skew_25delta or 0) * 100,
            "expiration": str(s.expiration_date),
        })

    ts_df = pd.DataFrame(ts_data)

    # ATM vol term structure
    fig = px.line(
        ts_df,
        x="tte_days",
        y="atm_vol",
        title=f"ATM Term Structure at {result.timestamp.strftime('%H:%M')} ET",
        labels={"tte_days": "Days to Expiry", "atm_vol": "ATM Vol (%)"},
        markers=True,
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

    # Skew term structure
    fig2 = px.line(
        ts_df,
        x="tte_days",
        y="skew",
        title="25-Delta Skew Term Structure",
        labels={"tte_days": "Days to Expiry", "skew": "Skew (%)"},
        markers=True,
    )
    fig2.update_traces(mode="lines+markers")
    st.plotly_chart(fig2, use_container_width=True)

    # Data table
    with st.expander("Surface Data Table"):
        st.dataframe(ts_df, use_container_width=True)


def render_spot_vol_dynamics(
    data_dir: Path,
    symbol: str,
    available_dates: List[date],
) -> None:
    """Render spot-vol dynamics analysis panel."""
    import numpy as np
    from volsurf.analytics.spot_vol_dynamics import SpotVolDynamicsAnalyzer

    st.subheader("Spot-Vol Dynamics Analysis")

    if len(available_dates) < 2:
        st.warning("Need at least 2 days of data for spot-vol analysis")
        return

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=available_dates[0],
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="spotvol_start",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=available_dates[-1],
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="spotvol_end",
        )

    # Target tenor selection
    target_tte = st.slider(
        "Target Tenor (DTE)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Interpolate ATM vol and skew to this tenor for analysis",
    )

    # Run analysis
    analyzer = SpotVolDynamicsAnalyzer(data_dir)
    result = analyzer.analyze_period(symbol, start_date, end_date, target_tte)

    if result.sample_size < 20:
        st.warning(f"Insufficient data: only {result.sample_size} observations. Need at least 20.")
        return

    # Summary metrics row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if result.leverage_correlation is not None and not np.isnan(result.leverage_correlation):
            st.metric(
                "Leverage Corr",
                f"{result.leverage_correlation:.3f}",
                help="Correlation between spot returns and vol changes (negative = leverage effect)",
            )
        else:
            st.metric("Leverage Corr", "N/A")

    with col2:
        if result.skew_spot_correlation is not None and not np.isnan(result.skew_spot_correlation):
            st.metric(
                "Skew-Spot Corr",
                f"{result.skew_spot_correlation:.3f}",
                help="Correlation between spot returns and skew changes",
            )
        else:
            st.metric("Skew-Spot Corr", "N/A")

    with col3:
        if result.vol_of_vol is not None and not np.isnan(result.vol_of_vol):
            st.metric(
                "Vol-of-Vol",
                f"{result.vol_of_vol:.1%}",
                help="Annualized volatility of ATM implied vol",
            )
        else:
            st.metric("Vol-of-Vol", "N/A")

    with col4:
        st.metric("Observations", f"{result.sample_size:,}")

    st.markdown("---")

    # Create sub-tabs for detailed views
    subtab1, subtab2, subtab3 = st.tabs(["Leverage Effect", "Time Series", "Statistics"])

    with subtab1:
        render_leverage_scatter(analyzer, symbol, start_date, end_date, target_tte, result)

    with subtab2:
        render_spot_vol_timeseries(analyzer, symbol, start_date, end_date, target_tte)

    with subtab3:
        render_spot_vol_statistics(result)


def render_leverage_scatter(
    analyzer,
    symbol: str,
    start_date: date,
    end_date: date,
    target_tte: int,
    result,
) -> None:
    """Render leverage effect scatter plot."""
    import numpy as np

    scatter_df = analyzer.get_spot_vol_scatter_data(symbol, start_date, end_date, target_tte)

    if scatter_df.empty:
        st.info("No data available for scatter plot")
        return

    # Create scatter plot with regression line
    fig = px.scatter(
        scatter_df,
        x="spot_return",
        y="vol_change",
        opacity=0.5,
        title="Spot Returns vs Vol Changes (Leverage Effect)",
        labels={
            "spot_return": "Spot Return (%)",
            "vol_change": "Vol Change (vol points)",
        },
    )

    # Add regression line if we have enough data
    if len(scatter_df) >= 10 and result.leverage_beta is not None:
        x_range = np.array([scatter_df["spot_return"].min(), scatter_df["spot_return"].max()])
        y_pred = result.leverage_beta * (x_range / 100) * 100  # Convert back for display

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            name=f"Regression (beta={result.leverage_beta:.4f})",
            line=dict(color="red", dash="dash"),
        ))

    fig.update_layout(
        hovermode="closest",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add interpretation
    if result.leverage_correlation is not None:
        corr = result.leverage_correlation
        if corr < -0.3:
            interpretation = "Strong negative correlation (classic leverage effect)"
        elif corr < -0.1:
            interpretation = "Moderate leverage effect"
        elif corr > 0.1:
            interpretation = "Unusual: positive spot-vol correlation"
        else:
            interpretation = "Weak spot-vol relationship"

        st.caption(f"**Interpretation:** {interpretation}")


def render_spot_vol_timeseries(
    analyzer,
    symbol: str,
    start_date: date,
    end_date: date,
    target_tte: int,
) -> None:
    """Render spot and vol time series on dual axis."""
    df = analyzer.load_intraday_timeseries(symbol, start_date, end_date, target_tte)

    if df.empty:
        st.info("No data available for time series")
        return

    # Create dual-axis plot
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Spot price on primary axis
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["underlying_price"],
            name="Spot Price",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    # ATM Vol on secondary axis
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["atm_vol"] * 100,
            name=f"ATM Vol ({target_tte}d)",
            line=dict(color="orange"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Spot Price vs ATM Implied Vol ({target_tte}d)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Spot Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="ATM Vol (%)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation chart
    st.markdown("**Rolling Leverage Correlation**")
    window = st.slider("Rolling Window", min_value=20, max_value=120, value=60, step=10)

    rolling_df = analyzer.get_rolling_leverage(symbol, start_date, end_date, window, target_tte)

    if not rolling_df.empty:
        fig2 = px.line(
            rolling_df,
            x="timestamp",
            y="rolling_leverage_corr",
            title=f"Rolling {window}-Observation Leverage Correlation",
            labels={"rolling_leverage_corr": "Correlation", "timestamp": "Time"},
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Typical SPY")
        st.plotly_chart(fig2, use_container_width=True)


def render_spot_vol_statistics(result) -> None:
    """Render detailed statistics table."""
    import numpy as np

    st.markdown("**Leverage Effect**")
    lev_data = {
        "Metric": ["Correlation", "Beta (slope)", "R-squared", "P-value"],
        "Value": [
            f"{result.leverage_correlation:.4f}" if result.leverage_correlation and not np.isnan(result.leverage_correlation) else "N/A",
            f"{result.leverage_beta:.6f}" if result.leverage_beta and not np.isnan(result.leverage_beta) else "N/A",
            f"{result.leverage_r_squared:.4f}" if result.leverage_r_squared and not np.isnan(result.leverage_r_squared) else "N/A",
            f"{result.leverage_pvalue:.4e}" if result.leverage_pvalue and not np.isnan(result.leverage_pvalue) else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(lev_data), use_container_width=True, hide_index=True)

    st.markdown("**Skew Dynamics**")
    skew_data = {
        "Metric": ["Skew-Spot Correlation", "Skew-Spot Beta"],
        "Value": [
            f"{result.skew_spot_correlation:.4f}" if result.skew_spot_correlation and not np.isnan(result.skew_spot_correlation) else "N/A",
            f"{result.skew_spot_beta:.6f}" if result.skew_spot_beta and not np.isnan(result.skew_spot_beta) else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(skew_data), use_container_width=True, hide_index=True)

    st.markdown("**Vol-of-Vol**")
    vov_data = {
        "Metric": ["Vol-of-Vol (annualized)", "Intraday Vol-of-Vol (avg)"],
        "Value": [
            f"{result.vol_of_vol:.2%}" if result.vol_of_vol and not np.isnan(result.vol_of_vol) else "N/A",
            f"{result.vol_of_vol_daily:.4f}" if result.vol_of_vol_daily and not np.isnan(result.vol_of_vol_daily) else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(vov_data), use_container_width=True, hide_index=True)

    st.markdown("**Vol Change Distribution**")
    dist_data = {
        "Metric": ["Mean", "Std Dev", "Skewness", "Kurtosis"],
        "Value": [
            f"{result.vol_change_mean:.6f}" if result.vol_change_mean and not np.isnan(result.vol_change_mean) else "N/A",
            f"{result.vol_change_std:.6f}" if result.vol_change_std and not np.isnan(result.vol_change_std) else "N/A",
            f"{result.vol_change_skewness:.4f}" if result.vol_change_skewness and not np.isnan(result.vol_change_skewness) else "N/A",
            f"{result.vol_change_kurtosis:.4f}" if result.vol_change_kurtosis and not np.isnan(result.vol_change_kurtosis) else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)
