"""Watchlist heat map page for Streamlit dashboard."""

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from volsurf.analytics.live_snapshot import LiveSnapshotManager, LiveSnapshotResult
from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator
from volsurf.config.settings import get_settings


def render_watchlist() -> None:
    """Render the watchlist heat map page."""
    st.title("Watchlist")

    settings = get_settings()
    symbols = settings.watchlist
    calc = SurfaceMetricsHistoryCalculator()

    all_latest = calc.get_all_symbols_latest()

    if all_latest.empty:
        st.warning("No surface metrics history data. Run the BTC-first flow first:")
        st.code(
            "volsurf ingest daily BTC\n"
            "volsurf fit surfaces BTC\n"
            "volsurf analyze realized-vol BTC\n"
            "volsurf analyze vrp BTC\n"
            "volsurf batch analyze --start 2026-01-01 --end 2026-12-31 --symbols BTC"
        )
        return

    # Target date = most recent date with data (for EOD fallback)
    eod_target_date = all_latest["quote_date"].max()
    if hasattr(eod_target_date, "date"):
        eod_target_date = eod_target_date.date()

    # Which symbols have EOD data?
    symbols_with_data = set(all_latest["symbol"].tolist())
    symbols_to_show = [s for s in symbols if s in symbols_with_data]
    missing = [s for s in symbols if s not in symbols_with_data]

    if not symbols_to_show:
        st.warning("No watchlist symbols have surface metrics data yet.")
        return

    # Controls row
    col_info, col_lb, col_refresh = st.columns([3, 1, 1])
    with col_lb:
        lookback = st.selectbox(
            "Percentile lookback",
            options=[30, 60, 90, 252],
            index=3,
            format_func=lambda x: f"{x}d",
            key="wl_lookback",
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_clicked = st.button("Refresh Live", key="wl_refresh")

    # ---- Live snapshot logic ----
    live_result = _maybe_fetch_live(symbols_to_show, refresh_clicked)

    if live_result and live_result.snapshots:
        # Build heatmap from live data
        heatmap_df = _build_live_heatmap_df(
            live_result, symbols_to_show, calc, eod_target_date, lookback,
        )
        ts_label = live_result.timestamp.strftime("%H:%M ET")
        with col_info:
            st.caption(f"Live as of **{ts_label}**")
        if live_result.errors:
            with st.expander(f"{len(live_result.errors)} symbol(s) fell back to EOD"):
                for sym, err in sorted(live_result.errors.items()):
                    st.text(f"{sym}: {err}")
    else:
        # Pure EOD fallback
        heatmap_df = calc.get_watchlist_heatmap_data(symbols_to_show, eod_target_date, lookback_days=lookback)
        reason = ""
        if "live_fallback_reason" in st.session_state:
            reason = f" ({st.session_state.live_fallback_reason})"
        with col_info:
            st.caption(f"Data as of **{eod_target_date}**{reason}")

    if heatmap_df.empty:
        st.warning("No heatmap data returned for the target date.")
        return

    if missing:
        st.info(f"Missing data: {', '.join(missing)}")

    # ---- Heat map table ----
    _render_heatmap_table(heatmap_df)

    # ---- Symbol detail ----
    st.markdown("---")
    detail_sym = st.selectbox("Symbol detail", symbols_to_show, index=0, key="wl_detail")

    enriched = calc.get_enriched_snapshot(detail_sym, eod_target_date)
    if enriched:
        _render_detail_card(detail_sym, enriched, calc, eod_target_date)
    else:
        st.info(f"No enriched snapshot available for {detail_sym} on {eod_target_date}.")


# ---------------------------------------------------------------------------
# Live snapshot helpers
# ---------------------------------------------------------------------------

def _maybe_fetch_live(
    symbols: list[str], force_refresh: bool,
) -> LiveSnapshotResult | None:
    """Auto-fetch live snapshots on first load, or on manual refresh.

    Returns the LiveSnapshotResult if live data is available, else None.
    Caches in st.session_state to survive re-renders.
    """
    # Already cached and no refresh requested
    if "live_result" in st.session_state and not force_refresh:
        return st.session_state.live_result

    mgr = LiveSnapshotManager()

    # Check terminal availability
    try:
        terminal_ok = mgr.check_terminal()
    except Exception:
        terminal_ok = False

    if not terminal_ok:
        st.session_state.live_fallback_reason = "terminal unavailable"
        st.session_state.live_result = None
        return None

    # Fetch with progress bar
    progress = st.progress(0, text="Connecting to Theta Terminal...")
    status_text = st.empty()

    def _on_progress(done: int, total: int, sym: str) -> None:
        progress.progress(done / total, text=f"Fetching live snapshot... ({done}/{total} {sym})")
        status_text.caption(f"{sym} done")

    live_result = mgr.fetch_all(symbols, progress_callback=_on_progress)
    progress.empty()
    status_text.empty()

    if not live_result.snapshots:
        st.session_state.live_fallback_reason = "market closed"
        st.session_state.live_result = None
        return None

    st.session_state.live_result = live_result
    st.session_state.pop("live_fallback_reason", None)
    return live_result


def _build_live_heatmap_df(
    live_result: LiveSnapshotResult,
    symbols: list[str],
    calc: SurfaceMetricsHistoryCalculator,
    eod_target_date: date,
    lookback_days: int,
) -> pd.DataFrame:
    """Build a heatmap DataFrame from live snapshots, falling back to EOD per-symbol.

    Computes percentile ranks and z-scores against the DB history.
    """
    mgr = LiveSnapshotManager()
    enrichment = mgr.enrich_with_percentiles(live_result.snapshots, lookback_days)

    rows = []
    for sym in symbols:
        snap = live_result.snapshots.get(sym)
        if snap is None:
            # Fall back to EOD for this symbol
            eod_df = calc.get_watchlist_heatmap_data([sym], eod_target_date, lookback_days=lookback_days)
            if not eod_df.empty:
                rows.append(eod_df.iloc[0].to_dict())
            continue

        enr = enrichment.get(sym, {})
        pctiles = enr.get("pctile", {})
        zscores = enr.get("zscore", {})

        rows.append({
            "symbol": sym,
            "underlying_close": snap.underlying_close,
            "atm_vol_30d": snap.atm_vol_30d,
            "atm_vol_30d_pctile": pctiles.get("atm_vol_30d"),
            "skew_25d_30d": snap.skew_25d_30d,
            "skew_25d_30d_zscore": zscores.get("skew_25d_30d"),
            "term_slope": snap.term_slope,
            "vrp_30d": snap.vrp_30d,
            "butterfly_30d": snap.butterfly_30d,
            "atm_vol_30d_1d_chg": snap.atm_vol_30d_1d_chg,
            "skew_25d_30d_1d_chg": snap.skew_25d_30d_1d_chg,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heat map table
# ---------------------------------------------------------------------------

def _render_heatmap_table(df: pd.DataFrame) -> None:
    """Styled heatmap table with conditional formatting."""
    display = pd.DataFrame({
        "Symbol": df["symbol"].values,
        "Price": pd.to_numeric(df["underlying_close"], errors="coerce"),
        "ATM 30d": pd.to_numeric(df["atm_vol_30d"], errors="coerce"),
        "Pctile": pd.to_numeric(df.get("atm_vol_30d_pctile"), errors="coerce"),
        "Skew": pd.to_numeric(df["skew_25d_30d"], errors="coerce"),
        "Skew Z": pd.to_numeric(df.get("skew_25d_30d_zscore"), errors="coerce"),
        "Slope": pd.to_numeric(df["term_slope"], errors="coerce"),
        "VRP": pd.to_numeric(df["vrp_30d"], errors="coerce"),
        "Bfly": pd.to_numeric(df["butterfly_30d"], errors="coerce"),
        "Δ ATM": pd.to_numeric(df["atm_vol_30d_1d_chg"], errors="coerce"),
        "Δ Skew": pd.to_numeric(df["skew_25d_30d_1d_chg"], errors="coerce"),
    })

    def _pctile_bg(val):
        if pd.isna(val):
            return ""
        if val < 10 or val > 90:
            return "background-color: #ef5350; color: white"
        if val < 25 or val > 75:
            return "background-color: #fff176"
        return ""

    def _zscore_bg(val):
        if pd.isna(val):
            return ""
        if abs(val) > 2:
            return "background-color: #ef5350; color: white"
        if abs(val) > 1:
            return "background-color: #fff176"
        return ""

    def _delta_color(val):
        if pd.isna(val):
            return ""
        return "color: #e53935" if val > 0 else "color: #43a047" if val < 0 else ""

    styled = (
        display.style
        .applymap(_pctile_bg, subset=["Pctile"])
        .applymap(_zscore_bg, subset=["Skew Z"])
        .applymap(_delta_color, subset=["Δ ATM", "Δ Skew"])
        .format(
            {
                "Price": "${:.2f}",
                "ATM 30d": "{:.1%}",
                "Pctile": "{:.0f}",
                "Skew": "{:.2%}",
                "Skew Z": "{:+.1f}σ",
                "Slope": "{:.2%}",
                "VRP": "{:.2%}",
                "Bfly": "{:.2%}",
                "Δ ATM": "{:+.2%}",
                "Δ Skew": "{:+.4f}",
            },
            na_rep="—",
        )
    )

    st.dataframe(styled, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Detail card
# ---------------------------------------------------------------------------

def _render_detail_card(
    symbol: str, enriched: dict, calc: SurfaceMetricsHistoryCalculator, target_date: date,
) -> None:
    snap = enriched["snapshot"]
    pctiles = enriched["percentiles"]

    # Key metrics with deltas
    cols = st.columns(5)
    _metric_col(cols[0], "ATM Vol 30d", snap.atm_vol_30d, snap.atm_vol_30d_1d_chg,
                pctiles.get("atm_vol_30d", {}), fmt=".2%", delta_fmt=".2%")
    _metric_col(cols[1], "Skew 30d", snap.skew_25d_30d, snap.skew_25d_30d_1d_chg,
                pctiles.get("skew_25d_30d", {}), fmt=".4f", delta_fmt=".4f")
    _metric_col(cols[2], "Term Slope", snap.term_slope, snap.term_slope_1d_chg,
                pctiles.get("term_slope", {}), fmt=".4f", delta_fmt=".4f")
    _metric_col(cols[3], "Butterfly", snap.butterfly_30d, None,
                pctiles.get("butterfly_30d", {}), fmt=".4f")
    _metric_col(cols[4], "VRP 30d", snap.vrp_30d, None,
                pctiles.get("vrp_30d", {}), fmt=".2%")

    # Percentile breakdown across lookback windows
    st.markdown("##### Percentile Ranks by Lookback Window")

    for metric_key, label in [
        ("atm_vol_30d", "ATM Vol 30d"),
        ("skew_25d_30d", "Skew 30d"),
        ("term_slope", "Term Slope"),
        ("butterfly_30d", "Butterfly"),
        ("vrp_30d", "VRP 30d"),
    ]:
        if metric_key not in pctiles:
            continue
        p_dict = pctiles[metric_key]
        windows = sorted(p_dict.keys())
        row_cols = st.columns([1.2] + [1] * len(windows))
        with row_cols[0]:
            st.markdown(f"**{label}**")
        for i, w in enumerate(windows):
            with row_cols[i + 1]:
                val = p_dict[w]
                if val is not None:
                    st.progress(val / 100.0, text=f"{w}d: {val:.0f}th")
                else:
                    st.caption(f"{w}d: —")

    # 30-day sparklines
    st.markdown("##### 30-Day Trend")
    ts = calc.get_timeseries(symbol, target_date - timedelta(days=45), target_date)
    if not ts.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(ts, x="quote_date", y="atm_vol_30d", title="ATM Vol 30d")
            fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(ts, x="quote_date", y="skew_25d_30d", title="Skew 30d")
            fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)


def _metric_col(container, label, value, delta_value, pctile_dict, fmt=".2%", delta_fmt=None):
    """Render st.metric with optional percentile caption."""
    with container:
        val_str = f"{value:{fmt}}" if value is not None else "—"
        delta_str = None
        if delta_value is not None and delta_fmt:
            delta_str = f"{delta_value:{delta_fmt}}"
        st.metric(label, val_str, delta=delta_str)
        p252 = pctile_dict.get(252)
        if p252 is not None:
            st.caption(f"{p252:.0f}th pctile (252d)")
