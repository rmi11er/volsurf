"""Macro Context page for Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


def render_macro_context() -> None:
    """Render the Macro Context page with 4 tabs."""
    st.title("Macro Context")

    from volsurf.analytics.macro_context import MacroContextCalculator
    from volsurf.config.settings import get_settings
    from volsurf.database.connection import get_connection

    settings = get_settings()
    calc = MacroContextCalculator()

    # Determine latest date with macro data
    conn = get_connection()
    macro_syms = list(settings.macro_symbols.keys())
    placeholders = ", ".join(["?"] * len(macro_syms))
    row = conn.execute(
        f"SELECT MAX(date) FROM underlying_prices WHERE symbol IN ({placeholders})",
        macro_syms,
    ).fetchone()

    if row is None or row[0] is None:
        st.warning("No macro data found. Run backfill first:")
        st.code("volsurf macro backfill --start 2025-01-01 --end 2026-02-05")
        return

    target_date = row[0]
    st.caption(f"Data as of **{target_date}**")

    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Correlations", "Breadth & Momentum", "Regime"])

    with tab1:
        _render_dashboard(calc, target_date)
    with tab2:
        _render_correlations(calc, settings, target_date)
    with tab3:
        _render_breadth(calc, settings, target_date)
    with tab4:
        _render_regime(calc, target_date)


def _render_dashboard(calc, target_date) -> None:
    """Tab 1: styled table + sparklines."""
    import plotly.graph_objects as go

    data = calc.get_macro_dashboard_data(target_date)

    if data.is_empty():
        st.info("No macro dashboard data available.")
        return

    # Styled table
    st.subheader("Macro Indicators")

    rows_pd = data.drop("sparkline").to_pandas()

    # Format the table
    format_map = {
        "chg_1d": "{:.2%}",
        "chg_5d": "{:.2%}",
        "chg_21d": "{:.2%}",
        "chg_63d": "{:.2%}",
        "pctile_252d": "{:.0f}",
        "close": "{:.2f}",
    }

    def _style_pctile(val):
        """Color extreme percentiles."""
        if val is None or str(val) == "nan":
            return ""
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v <= 10:
            return "background-color: #ffcccc"
        if v >= 90:
            return "background-color: #ccffcc"
        return ""

    styled = rows_pd.style.format(format_map, na_rep="—")
    styled = styled.map(_style_pctile, subset=["pctile_252d"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Sparklines in 3x2 grid
    st.subheader("30-Day Sparklines")
    symbols = data["symbol"].to_list()
    sparklines = data["sparkline"].to_list()

    cols_per_row = 3
    for row_start in range(0, len(symbols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(symbols):
                break
            sym = symbols[idx]
            spark = sparklines[idx]
            with col:
                fig = go.Figure(
                    go.Scatter(
                        y=spark,
                        mode="lines",
                        line=dict(color="#4A90D9", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(74, 144, 217, 0.1)",
                    )
                )
                fig.update_layout(
                    title=dict(text=sym, font=dict(size=14)),
                    height=150,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{sym}")


def _render_correlations(calc, settings, target_date) -> None:
    """Tab 2: heatmap + rolling correlation selector."""
    import plotly.express as px

    st.subheader("Cross-Asset Correlation Matrix")

    window = st.selectbox(
        "Correlation window",
        options=[21, 63, 126],
        index=1,
        format_func=lambda x: f"{x}d",
        key="corr_window",
    )

    # Combine watchlist + macro symbols
    all_syms = list(settings.watchlist) + list(settings.macro_symbols.keys())
    corr_df = calc.get_correlation_matrix(all_syms, target_date, window=window)

    if corr_df.is_empty():
        st.info("Not enough data for correlation matrix.")
        return

    # Convert to pandas for heatmap
    corr_pd = corr_df.to_pandas().set_index("symbol")

    fig = px.imshow(
        corr_pd.values,
        x=corr_pd.columns.tolist(),
        y=corr_pd.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(height=600, title=f"{window}-Day Return Correlations")
    st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation time series
    st.markdown("---")
    st.subheader("Rolling Correlation")

    col1, col2 = st.columns(2)
    with col1:
        wl_sym = st.selectbox(
            "Watchlist symbol",
            options=settings.watchlist,
            key="roll_corr_wl",
        )
    with col2:
        macro_sym = st.selectbox(
            "Macro symbol",
            options=list(settings.macro_symbols.keys()),
            key="roll_corr_macro",
        )

    rolling = calc.get_rolling_correlations(wl_sym, macro_sym, target_date)
    if rolling.is_empty():
        st.info("Not enough data for rolling correlations.")
        return

    rolling_pd = rolling.to_pandas()
    corr_cols = [c for c in rolling_pd.columns if c.startswith("corr_")]

    fig = px.line(
        rolling_pd,
        x="date",
        y=corr_cols,
        title=f"{wl_sym} vs {macro_sym} Rolling Correlation",
        labels={"value": "Correlation", "variable": "Window"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def _render_breadth(calc, settings, target_date) -> None:
    """Tab 3: breadth metrics + per-symbol MA table."""
    import plotly.express as px

    st.subheader("Market Breadth")

    symbols = settings.watchlist

    # Current snapshot
    snapshot = calc.get_current_breadth_snapshot(symbols, target_date)
    if snapshot:
        above = sum(1 for s in snapshot if s["above_ma"])
        total = len(snapshot)
        pct = above / total if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Above 50d MA", f"{above}/{total}", f"{pct:.0%}")
        with col2:
            advancing = sum(1 for s in snapshot if s["close"] > s["ma_50d"])
            st.metric("Advancing", str(advancing))
        with col3:
            declining = total - advancing
            st.metric("Declining", str(declining))

        # Per-symbol MA table
        st.markdown("---")
        st.subheader("Per-Symbol Detail")

        import pandas as pd

        snap_df = pd.DataFrame(snapshot)
        snap_df["distance_to_ma"] = (snap_df["close"] / snap_df["ma_50d"] - 1)

        def _color_above(val):
            if val is True:
                return "background-color: #ccffcc"
            if val is False:
                return "background-color: #ffcccc"
            return ""

        styled = snap_df.style.format({
            "close": "{:.2f}",
            "ma_50d": "{:.2f}",
            "distance_to_ma": "{:.2%}",
        }).map(_color_above, subset=["above_ma"])

        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No breadth data available for watchlist symbols.")

    # Breadth time series
    st.markdown("---")
    st.subheader("Breadth Over Time")

    breadth = calc.get_breadth_indicators(symbols, target_date)
    if not breadth.is_empty():
        breadth_pd = breadth.to_pandas()

        fig = px.area(
            breadth_pd,
            x="date",
            y="pct_above_50d_ma",
            title="% of Watchlist Above 50d MA",
            labels={"pct_above_50d_ma": "% Above 50d MA"},
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough history for breadth time series.")


def _render_regime(calc, target_date) -> None:
    """Tab 4: regime cards + VIX time series with band shading."""
    import plotly.graph_objects as go

    st.subheader("Regime Context")

    regime = calc.get_regime_context(target_date)

    # 4 metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vix = regime.get("vix_level")
        vix_str = f"{vix:.1f}" if vix else "N/A"
        st.metric("VIX", vix_str, regime.get("vix_regime"))

    with col2:
        ratio = regime.get("vix_ts_ratio")
        ratio_str = f"{ratio:.3f}" if ratio else "N/A"
        st.metric("VIX Term Structure", ratio_str, regime.get("vix_ts_label"))

    with col3:
        hyg = regime.get("hyg_21d_return")
        hyg_str = f"{hyg:.2%}" if hyg else "N/A"
        st.metric("Credit (HYG 21d)", hyg_str, regime.get("credit_label"))

    with col4:
        uup = regime.get("uup_21d_return")
        uup_str = f"{uup:.2%}" if uup else "N/A"
        st.metric("USD (UUP 21d)", uup_str, regime.get("usd_label"))

    # Additional context
    col5, col6 = st.columns(2)
    with col5:
        vvix = regime.get("vvix_level")
        vvix_str = f"{vvix:.1f}" if vvix else "N/A"
        st.metric("VVIX (Vol-of-Vol)", vvix_str)
    with col6:
        tlt = regime.get("tlt_63d_return")
        tlt_str = f"{tlt:.2%}" if tlt else "N/A"
        st.metric("Rates (TLT 63d)", tlt_str, regime.get("rate_label"))

    # VIX time series with regime bands
    st.markdown("---")
    st.subheader("VIX History")

    vix_ts = calc.get_vix_timeseries(target_date)
    if not vix_ts.is_empty():
        vix_pd = vix_ts.to_pandas()

        fig = go.Figure()

        # Regime bands
        fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.05,
                      annotation_text="Low", annotation_position="top left")
        fig.add_hrect(y0=15, y1=20, fillcolor="yellow", opacity=0.05,
                      annotation_text="Normal", annotation_position="top left")
        fig.add_hrect(y0=20, y1=30, fillcolor="orange", opacity=0.05,
                      annotation_text="Elevated", annotation_position="top left")
        fig.add_hrect(y0=30, y1=80, fillcolor="red", opacity=0.05,
                      annotation_text="High", annotation_position="top left")

        fig.add_trace(go.Scatter(
            x=vix_pd["date"],
            y=vix_pd["close"],
            mode="lines",
            name="VIX",
            line=dict(color="#4A90D9", width=2),
        ))

        fig.update_layout(
            title="VIX with Regime Bands",
            yaxis_title="VIX Level",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No VIX time series data available.")
