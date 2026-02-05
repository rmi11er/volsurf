"""VRP Analysis page for Streamlit dashboard."""

from datetime import date
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics import RealizedVolCalculator, SurfaceMetrics, VRPCalculator
from volsurf.database.connection import get_connection


def render_vrp_analysis(symbol: str, available_dates: List[date]) -> None:
    """Render the VRP analysis page."""
    tabs = st.tabs(["VRP Time Series", "IV vs RV", "Realized Vol", "Skew Analysis"])

    with tabs[0]:
        render_vrp_timeseries(symbol, available_dates)

    with tabs[1]:
        render_iv_vs_rv(symbol, available_dates)

    with tabs[2]:
        render_realized_vol(symbol, available_dates)

    with tabs[3]:
        render_skew_analysis(symbol, available_dates)


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

    # 10th / 90th percentile bands
    vrp_clean = df["vrp_30d"].dropna()
    if len(vrp_clean) > 10:
        p10 = np.percentile(vrp_clean, 10) * 100
        p90 = np.percentile(vrp_clean, 90) * 100
        fig.add_hline(y=p10, line_dash="dot", line_color="green", opacity=0.5,
                      annotation_text="10th pctile", annotation_position="bottom left")
        fig.add_hline(y=p90, line_dash="dot", line_color="red", opacity=0.5,
                      annotation_text="90th pctile", annotation_position="top left")

    fig.update_layout(
        title=f"{symbol} Variance Risk Premium (IV - RV)",
        xaxis_title="Date",
        yaxis_title="VRP (%)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # Statistics with percentile context
    current = df["vrp_30d"].iloc[-1]
    mean = df["vrp_30d"].mean()
    std = df["vrp_30d"].std()
    zscore = (current - mean) / std if std > 0 else 0

    # Try to get percentile rank from surface_metrics_history
    vrp_pctile = None
    try:
        from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator
        smh_calc = SurfaceMetricsHistoryCalculator()
        vrp_pctile = smh_calc.calculate_percentile_rank(
            symbol, "vrp_30d", current, lookback_days=252
        )
    except Exception:
        pass

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current VRP", f"{current:.2%}")
        if vrp_pctile is not None:
            st.caption(f"{vrp_pctile:.0f}th pctile (252d)")
    with col2:
        st.metric("Mean VRP", f"{mean:.2%}")
    with col3:
        pct_positive = (df["vrp_30d"] > 0).mean()
        st.metric("% Positive", f"{pct_positive:.1%}")
    with col4:
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

    st.plotly_chart(fig, width="stretch")

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

    st.plotly_chart(fig, width="stretch")

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

    st.plotly_chart(fig, width="stretch")

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


def render_skew_analysis(symbol: str, available_dates: List[date]) -> None:
    """Render spot-vol correlation (realized skew) analysis."""
    st.subheader("Spot-Vol Correlation Analysis")

    st.markdown(
        """
    **Theory**: Implied skew prices an expected relationship between spot moves and vol changes.
    Realized skew measures the actual spot-vol beta: how much does ATM vol change per unit spot move?
    """
    )

    metrics = SurfaceMetrics()

    start_date = available_dates[0]
    end_date = available_dates[-1]

    # Get data for analysis
    conn = get_connection()

    # Get underlying prices
    price_query = """
        SELECT date, close
        FROM underlying_prices
        WHERE symbol = ?
          AND date >= ?
          AND date <= ?
        ORDER BY date
    """
    price_df = conn.execute(price_query, [symbol, start_date, end_date]).fetchdf()

    if price_df.empty or len(price_df) < 10:
        st.warning("Insufficient price data for spot-vol analysis.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        tenor = st.selectbox(
            "ATM Vol Tenor (days)",
            options=[7, 14, 21, 30, 45, 60],
            index=3,
            key="spotvol_tenor",
        )
        regression_window = st.selectbox(
            "Regression Window (days)",
            options=[21, 42, 63, 126],
            index=1,
            key="regression_window",
        )

    # Get ATM vol time series
    atm_df = metrics.get_atm_vol_timeseries(symbol, start_date, end_date, tte_target_days=tenor)

    # Get implied skew time series
    skew_df = metrics.get_skew_timeseries(symbol, start_date, end_date, tte_target_days=tenor)

    if atm_df.empty:
        st.warning("No ATM vol data available.")
        return

    # Merge price and vol data
    price_df["date"] = pd.to_datetime(price_df["date"])
    atm_df["date"] = pd.to_datetime(atm_df["date"])

    merged = pd.merge(price_df, atm_df, on="date", how="inner")

    if len(merged) < 10:
        st.warning("Insufficient merged data for analysis.")
        return

    # Calculate returns and vol changes
    merged["spot_return"] = np.log(merged["close"] / merged["close"].shift(1))
    merged["vol_change"] = merged["atm_vol"].diff()
    merged["vol_change_pct"] = merged["atm_vol"].pct_change()
    merged = merged.dropna()

    # --- Spot-Vol Scatter Plot ---
    st.markdown("---")
    st.subheader("Spot-Vol Relationship")

    from scipy import stats as scipy_stats

    # Use regression window for the analysis
    if len(merged) > regression_window:
        window_data = merged.iloc[-regression_window:]
    else:
        window_data = merged

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        window_data["spot_return"], window_data["vol_change"]
    )

    fig = go.Figure()

    # Scatter points - show window data with full period faded in background
    if len(merged) > regression_window:
        # Background: older data (faded)
        older_data = merged.iloc[:-regression_window]
        fig.add_trace(
            go.Scatter(
                x=older_data["spot_return"] * 100,
                y=older_data["vol_change"] * 100,
                mode="markers",
                name="Older Observations",
                marker=dict(size=5, opacity=0.2, color="gray"),
                hovertemplate="Spot Return: %{x:.2f}%<br>Vol Change: %{y:.2f}%<extra></extra>",
            )
        )

    # Window data (highlighted)
    fig.add_trace(
        go.Scatter(
            x=window_data["spot_return"] * 100,
            y=window_data["vol_change"] * 100,
            mode="markers",
            name=f"Last {regression_window}d",
            marker=dict(size=7, opacity=0.7, color="blue"),
            hovertemplate="Spot Return: %{x:.2f}%<br>Vol Change: %{y:.2f}%<extra></extra>",
        )
    )

    # Regression line (based on window)
    x_range = np.linspace(merged["spot_return"].min(), merged["spot_return"].max(), 100)
    y_fit = intercept + slope * x_range

    fig.add_trace(
        go.Scatter(
            x=x_range * 100,
            y=y_fit * 100,
            mode="lines",
            name=f"Regression (β={slope:.4f})",
            line=dict(color="red", width=2),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"{symbol} Spot-Vol Relationship ({tenor}d ATM Vol, {regression_window}d window)",
        xaxis_title="Spot Return (%)",
        yaxis_title="ATM Vol Change (%)",
        hovermode="closest",
    )

    st.plotly_chart(fig, width="stretch")

    # Regression statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Spot-Vol Beta (β)", f"{slope:.4f}", help=f"Vol change per unit spot return ({regression_window}d)")
    with col2:
        st.metric("R²", f"{r_value**2:.3f}", help="Explained variance")
    with col3:
        # Express as: for 1% spot down, vol increases by X%
        vol_per_1pct_down = -slope * 100  # negative because down move
        st.metric("Vol/1% Down", f"{vol_per_1pct_down:.2f}%", help="Expected vol increase for 1% spot decline")
    with col4:
        st.metric("Correlation", f"{r_value:.3f}", help="Spot-vol correlation")

    # --- Rolling Spot-Vol Beta ---
    st.markdown("---")
    st.subheader("Rolling Spot-Vol Beta")

    # Calculate rolling regression
    rolling_betas = []
    rolling_r2s = []
    rolling_dates = []

    for i in range(regression_window, len(merged)):
        window_data = merged.iloc[i - regression_window : i]
        if len(window_data) >= 10:
            slope_r, _, r_val, _, _ = scipy_stats.linregress(
                window_data["spot_return"], window_data["vol_change"]
            )
            rolling_betas.append(slope_r)
            rolling_r2s.append(r_val**2)
            rolling_dates.append(merged.iloc[i]["date"])

    if rolling_betas:
        rolling_df = pd.DataFrame({
            "date": rolling_dates,
            "beta": rolling_betas,
            "r2": rolling_r2s,
        })

        # Also add implied skew for comparison
        if not skew_df.empty:
            skew_df["date"] = pd.to_datetime(skew_df["date"])
            rolling_df = pd.merge(rolling_df, skew_df, on="date", how="left")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rolling_df["date"],
                y=rolling_df["beta"],
                mode="lines",
                name=f"Realized Spot-Vol Beta ({regression_window}d)",
                line=dict(color="blue"),
            )
        )

        # Add implied skew for comparison (scaled)
        if "skew" in rolling_df.columns:
            # Scale implied skew to be comparable
            # Implied skew represents expected vol difference across strikes
            # We can approximate the implied beta from the skew
            fig.add_trace(
                go.Scatter(
                    x=rolling_df["date"],
                    y=rolling_df["skew"] * -1,  # Flip sign: negative skew = positive spot-vol beta
                    mode="lines",
                    name=f"Implied Skew ({tenor}d, inverted)",
                    line=dict(color="orange", dash="dash"),
                    yaxis="y2",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=f"{symbol} Rolling Spot-Vol Beta vs Implied Skew",
            xaxis_title="Date",
            yaxis_title="Spot-Vol Beta",
            yaxis2=dict(
                title="Implied Skew (inverted)",
                overlaying="y",
                side="right",
                tickformat=".1%",
            ),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig, width="stretch")

        # Current values
        col1, col2, col3 = st.columns(3)

        with col1:
            current_beta = rolling_df["beta"].iloc[-1]
            mean_beta = rolling_df["beta"].mean()
            st.metric(
                f"Current Beta ({regression_window}d)",
                f"{current_beta:.4f}",
                delta=f"{current_beta - mean_beta:.4f} vs mean",
            )

        with col2:
            if "skew" in rolling_df.columns and pd.notna(rolling_df["skew"].iloc[-1]):
                current_skew = rolling_df["skew"].iloc[-1]
                st.metric(f"Current Implied Skew ({tenor}d)", f"{current_skew:.2%}")

        with col3:
            # Skew premium: difference between implied and realized
            if "skew" in rolling_df.columns and pd.notna(rolling_df["skew"].iloc[-1]):
                # Implied beta approximation from skew
                # Skew tells us vol difference across strikes; beta tells us vol change per spot move
                # A rough conversion: implied_beta ≈ -skew / sqrt(T)
                tte_years = tenor / 252
                implied_beta_approx = -current_skew / np.sqrt(tte_years)
                skew_premium = implied_beta_approx - current_beta
                st.metric(
                    "Skew Premium",
                    f"{skew_premium:.4f}",
                    help="Implied beta - Realized beta (positive = skew overpriced)",
                )

    # --- Expected vs Actual Vol Moves ---
    st.markdown("---")
    st.subheader("Skew Slide: Expected vs Actual Vol Moves")

    st.markdown(
        """
    Based on the current implied skew, we can estimate the expected vol change for a given spot move.
    This shows whether vol is moving more or less than the skew implies.
    """
    )

    # Calculate expected vol change based on skew
    if not skew_df.empty:
        skew_df["date"] = pd.to_datetime(skew_df["date"])
        analysis_df = pd.merge(merged, skew_df, on="date", how="inner")

        if len(analysis_df) > 1:
            # Expected vol change from skew: Δσ_expected ≈ -skew * Δspot / sqrt(T)
            tte_years = tenor / 252
            analysis_df["expected_vol_change"] = (
                -analysis_df["skew"].shift(1) * analysis_df["spot_return"] / np.sqrt(tte_years)
            )
            analysis_df["vol_surprise"] = analysis_df["vol_change"] - analysis_df["expected_vol_change"]
            analysis_df = analysis_df.dropna()

            if len(analysis_df) > 0:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=analysis_df["date"],
                        y=analysis_df["expected_vol_change"] * 100,
                        mode="lines",
                        name="Expected Vol Change (from skew)",
                        line=dict(color="orange"),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=analysis_df["date"],
                        y=analysis_df["vol_change"] * 100,
                        mode="lines",
                        name="Actual Vol Change",
                        line=dict(color="blue"),
                    )
                )

                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig.update_layout(
                    title=f"{symbol} Expected vs Actual ATM Vol Changes",
                    xaxis_title="Date",
                    yaxis_title="Vol Change (%)",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, width="stretch")

                # Cumulative surprise
                analysis_df["cum_surprise"] = analysis_df["vol_surprise"].cumsum()

                fig2 = px.line(
                    analysis_df,
                    x="date",
                    y=analysis_df["cum_surprise"] * 100,
                    title=f"{symbol} Cumulative Vol Surprise (Actual - Expected)",
                    labels={"y": "Cumulative Surprise (%)"},
                )

                fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig2.update_layout(hovermode="x unified")

                st.plotly_chart(fig2, width="stretch")

                # Statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    mean_surprise = analysis_df["vol_surprise"].mean() * 100
                    st.metric(
                        "Mean Vol Surprise",
                        f"{mean_surprise:.3f}%",
                        help="Positive = vol moved more than skew implied",
                    )

                with col2:
                    hit_rate = (
                        (analysis_df["expected_vol_change"] * analysis_df["vol_change"] > 0).mean()
                    )
                    st.metric(
                        "Direction Hit Rate",
                        f"{hit_rate:.1%}",
                        help="How often skew correctly predicted vol direction",
                    )

                with col3:
                    correlation = analysis_df["expected_vol_change"].corr(analysis_df["vol_change"])
                    st.metric(
                        "Expected/Actual Correlation",
                        f"{correlation:.3f}",
                    )

    # --- Skew Term Structure ---
    st.markdown("---")
    st.subheader("Implied Skew Term Structure")

    tenors = [7, 14, 21, 30, 45, 60, 90]
    skew_term = []

    for t in tenors:
        ts_df = metrics.get_skew_timeseries(symbol, end_date, end_date, tte_target_days=t)
        if not ts_df.empty:
            skew_term.append({"tenor": t, "skew": ts_df["skew"].iloc[0] * 100})

    if skew_term:
        term_df = pd.DataFrame(skew_term)

        fig = px.scatter(
            term_df,
            x="tenor",
            y="skew",
            title=f"{symbol} Implied Skew Term Structure (as of {end_date})",
            labels={"tenor": "Days to Expiry", "skew": "Skew (%)"},
        )
        fig.update_traces(mode="markers+lines")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    # Interpretation
    with st.expander("Spot-Vol Analysis Interpretation"):
        st.markdown(
            """
        **Spot-Vol Beta (Realized Skew)**
        - Measures the actual relationship: how much does ATM vol change per unit spot move?
        - β < 0: Vol rises when spot falls (typical for equities)
        - More negative β = stronger spot-vol inverse relationship

        **Implied Skew**
        - The 25-delta put-call spread prices in an expected spot-vol relationship
        - More negative skew = market expects stronger vol increase on down moves
        - Implied skew reflects both expected moves and risk premium

        **Skew Premium**
        - Difference between implied and realized spot-vol beta
        - Positive premium: Implied skew > realized beta (skew is "expensive")
        - This is the premium skew sellers aim to harvest

        **Expected vs Actual Vol**
        - Using current skew, we estimate expected vol change: Δσ ≈ -skew × ΔS / √T
        - "Vol surprise" = actual - expected
        - Cumulative surprise shows if skew has been systematically over/under-priced

        **Trading Implications**
        - High skew premium + low realized beta = potential skew selling opportunity
        - Cumulative positive surprise = vol has moved more than skew implied
        - Rolling beta trends help time skew trades
        """
        )
