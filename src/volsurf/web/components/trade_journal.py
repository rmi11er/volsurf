"""Trade Journal web component for Streamlit dashboard."""

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytics.trade_journal import (
    VALID_CATEGORIES,
    VALID_DIRECTIONS,
    TradeEntry,
    TradeJournalManager,
    TradeResolution,
)


def render_trade_journal() -> None:
    """Render the trade journal page."""
    st.title("Trade Journal")

    manager = TradeJournalManager()

    tabs = st.tabs(["New Trade", "Open Trades", "Trade History", "Performance"])

    with tabs[0]:
        _render_new_trade(manager)

    with tabs[1]:
        _render_open_trades(manager)

    with tabs[2]:
        _render_trade_history(manager)

    with tabs[3]:
        _render_performance(manager)


# ---------------------------------------------------------------------------
# Tab 1: New Trade Entry
# ---------------------------------------------------------------------------

def _render_new_trade(manager: TradeJournalManager) -> None:
    st.subheader("Log New Trade")

    # Get available symbols from settings
    from volsurf.config.settings import get_settings
    symbols = get_settings().watchlist

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox("Symbol", symbols, key="tj_symbol")
        category = st.selectbox(
            "Category",
            sorted(VALID_CATEGORIES),
            format_func=lambda x: x.replace("_", " ").title(),
            key="tj_category",
        )
        direction = st.selectbox(
            "Direction",
            sorted(VALID_DIRECTIONS),
            format_func=lambda x: x.replace("_", " ").title(),
            key="tj_direction",
        )

    with col2:
        entry_date = st.date_input("Entry Date", value=date.today(), key="tj_entry_date")
        horizon = st.number_input(
            "Time Horizon (days)", min_value=1, max_value=365, value=10, key="tj_horizon"
        )
        tags = st.text_input("Tags (comma-separated)", key="tj_tags")

    thesis = st.text_area("Thesis", placeholder="Why are you making this trade?", key="tj_thesis")
    structure = st.text_input(
        "Structure Description",
        placeholder="e.g., Long 30d 25d put spread",
        key="tj_structure",
    )

    col_t, col_s = st.columns(2)
    with col_t:
        target = st.text_input("Target", placeholder="e.g., ATM vol to 25%", key="tj_target")
    with col_s:
        stop = st.text_input("Stop", placeholder="e.g., Close if vol > 35%", key="tj_stop")

    # Show current market context
    _show_market_context(symbol, entry_date)

    if st.button("Submit Trade", type="primary", key="tj_submit"):
        if not thesis:
            st.error("Thesis is required.")
            return

        entry = TradeEntry(
            symbol=symbol,
            thesis=thesis,
            category=category,
            direction=direction,
            entry_date=entry_date,
            structure_description=structure or None,
            target_description=target or None,
            stop_description=stop or None,
            time_horizon_days=horizon,
            tags=tags or None,
        )

        try:
            trade_id = manager.create_trade(entry)
            st.success(f"Trade #{trade_id} created successfully!")
            st.rerun()
        except ValueError as e:
            st.error(str(e))


def _show_market_context(symbol: str, target_date: date) -> None:
    """Show current market context for the selected symbol."""
    try:
        from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator

        calc = SurfaceMetricsHistoryCalculator()
        enriched = calc.get_enriched_snapshot(symbol, target_date)
        if not enriched:
            return

        snap = enriched["snapshot"]
        pctiles = enriched.get("percentiles", {})

        st.markdown("##### Market Context")
        cols = st.columns(5)

        with cols[0]:
            val = f"{snap.atm_vol_30d:.2%}" if snap.atm_vol_30d else "—"
            st.metric("ATM Vol 30d", val)
            p = pctiles.get("atm_vol_30d", {}).get(252)
            if p is not None:
                st.caption(f"{p:.0f}th pctile")

        with cols[1]:
            val = f"{snap.skew_25d_30d:.4f}" if snap.skew_25d_30d else "—"
            st.metric("Skew 30d", val)

        with cols[2]:
            val = f"{snap.term_slope:.4f}" if snap.term_slope else "—"
            st.metric("Term Slope", val)

        with cols[3]:
            val = f"{snap.vrp_30d:.2%}" if snap.vrp_30d else "—"
            st.metric("VRP 30d", val)

        with cols[4]:
            val = f"${snap.underlying_close:.2f}" if snap.underlying_close else "—"
            st.metric("Price", val)

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 2: Open Trades
# ---------------------------------------------------------------------------

def _render_open_trades(manager: TradeJournalManager) -> None:
    st.subheader("Open Trades")

    df = manager.get_open_trades()

    if df.empty:
        st.info("No open trades. Use the 'New Trade' tab to log one.")
        return

    st.caption(f"{len(df)} open trade(s)")

    for _, row in df.iterrows():
        trade_id = int(row["trade_id"])
        with st.expander(
            f"#{trade_id} — {row['symbol']} {row['direction']} ({row['category']}) — {row['entry_date']}"
        ):
            st.markdown(f"**Thesis:** {row['thesis']}")
            if row.get("structure_description"):
                st.markdown(f"**Structure:** {row['structure_description']}")

            cols = st.columns(4)
            with cols[0]:
                iv = f"{row['entry_iv_level']:.2%}" if row.get("entry_iv_level") is not None else "—"
                st.metric("Entry IV", iv)
            with cols[1]:
                skew = f"{row['entry_skew_level']:.4f}" if row.get("entry_skew_level") is not None else "—"
                st.metric("Entry Skew", skew)
            with cols[2]:
                price = f"${row['entry_underlying_price']:.2f}" if row.get("entry_underlying_price") is not None else "—"
                st.metric("Entry Price", price)
            with cols[3]:
                horizon = f"{int(row['time_horizon_days'])}d" if row.get("time_horizon_days") is not None else "—"
                st.metric("Horizon", horizon)

            # Resolve form
            st.markdown("---")
            st.markdown("**Resolve this trade:**")
            resolve_cols = st.columns([1, 1, 2])
            with resolve_cols[0]:
                outcome = st.selectbox(
                    "Outcome",
                    ["win", "loss", "scratch"],
                    key=f"resolve_outcome_{trade_id}",
                )
            with resolve_cols[1]:
                pnl = st.number_input(
                    "P&L (vol pts)",
                    value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"resolve_pnl_{trade_id}",
                    help="Leave 0 for auto-computation",
                )
            with resolve_cols[2]:
                notes = st.text_input("Notes", key=f"resolve_notes_{trade_id}")

            if st.button("Resolve", key=f"resolve_btn_{trade_id}"):
                resolution = TradeResolution(
                    trade_id=trade_id,
                    outcome=outcome,
                    pnl_vol_points=pnl if pnl != 0 else None,
                    outcome_notes=notes or None,
                )
                try:
                    manager.resolve_trade(resolution)
                    st.success(f"Trade #{trade_id} resolved: {outcome}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # Auto-resolve button
    st.markdown("---")
    if st.button("Auto-Resolve Expired Trades"):
        count = manager.auto_resolve_expired()
        if count > 0:
            st.success(f"Auto-resolved {count} trade(s)")
            st.rerun()
        else:
            st.info("No expired trades to resolve.")


# ---------------------------------------------------------------------------
# Tab 3: Trade History
# ---------------------------------------------------------------------------

def _render_trade_history(manager: TradeJournalManager) -> None:
    st.subheader("Trade History")

    from volsurf.config.settings import get_settings
    symbols = get_settings().watchlist

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_symbol = st.selectbox("Symbol", ["All"] + symbols, key="th_symbol_filter")
    with col2:
        filter_category = st.selectbox(
            "Category",
            ["All"] + sorted(VALID_CATEGORIES),
            format_func=lambda x: x.replace("_", " ").title() if x != "All" else "All",
            key="th_category",
        )
    with col3:
        limit = st.number_input("Show last", min_value=5, max_value=500, value=50, key="th_limit")

    df = manager.get_trade_history(
        symbol=filter_symbol if filter_symbol != "All" else None,
        category=filter_category if filter_category != "All" else None,
        limit=limit,
    )

    if df.empty:
        st.info("No trades found matching filters.")
        return

    # Summary stats for the filtered view
    total = len(df)
    wins = (df["outcome"] == "win").sum()
    losses = (df["outcome"] == "loss").sum()
    open_count = (df["outcome"] == "open").sum()

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total", total)
    with cols[1]:
        st.metric("Wins", int(wins))
    with cols[2]:
        st.metric("Losses", int(losses))
    with cols[3]:
        resolved = total - open_count
        wr = wins / resolved if resolved > 0 else 0
        st.metric("Win Rate", f"{wr:.0%}" if resolved > 0 else "—")

    # Display table
    display_df = df[["trade_id", "entry_date", "exit_date", "symbol", "direction",
                      "category", "outcome", "pnl_vol_points"]].copy()
    display_df.columns = ["ID", "Entry", "Exit", "Symbol", "Direction", "Category", "Outcome", "P&L"]

    def _outcome_color(val):
        if val == "win":
            return "color: #43a047"
        if val == "loss":
            return "color: #e53935"
        if val == "scratch":
            return "color: #ffa726"
        return "color: gray"

    styled = (
        display_df.style
        .applymap(_outcome_color, subset=["Outcome"])
        .format({"P&L": "{:.4f}", "ID": "{:.0f}"}, na_rep="—")
    )

    st.dataframe(styled, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Performance Analytics
# ---------------------------------------------------------------------------

def _render_performance(manager: TradeJournalManager) -> None:
    st.subheader("Performance Analytics")

    summary = manager.get_performance_summary()

    if summary["total_trades"] == 0:
        st.info("No trades to analyze. Start by logging trades in the 'New Trade' tab.")
        return

    # Overview metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Trades", summary["total_trades"])
    with cols[1]:
        st.metric("Win Rate", f"{summary['win_rate']:.0%}" if summary["win_rate"] is not None else "—")
    with cols[2]:
        st.metric("Avg P&L", f"{summary['avg_pnl']:.4f}" if summary["avg_pnl"] is not None else "—")
    with cols[3]:
        st.metric("Avg Win", f"{summary['avg_win']:.4f}" if summary["avg_win"] is not None else "—")
    with cols[4]:
        st.metric("Avg Loss", f"{summary['avg_loss']:.4f}" if summary["avg_loss"] is not None else "—")

    # Win rate by category
    st.markdown("---")
    cat_df = manager.get_category_performance()

    if not cat_df.empty and cat_df["total"].sum() > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Win Rate by Category")
            resolved_cats = cat_df[cat_df["win_rate_pct"].notna()].copy()
            if not resolved_cats.empty:
                fig = px.bar(
                    resolved_cats,
                    x="category",
                    y="win_rate_pct",
                    title="Win Rate by Category",
                    labels={"win_rate_pct": "Win Rate (%)", "category": ""},
                    text="win_rate_pct",
                )
                fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                fig.update_layout(height=350, yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### Trade Count by Category")
            fig = px.bar(
                cat_df,
                x="category",
                y="total",
                title="Trades by Category",
                labels={"total": "Count", "category": ""},
                color="category",
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Cumulative P&L over time
    st.markdown("---")
    st.markdown("##### Cumulative P&L")

    history = manager.get_trade_history(limit=500)
    resolved = history[history["outcome"] != "open"].copy()

    if not resolved.empty and resolved["pnl_vol_points"].notna().any():
        resolved = resolved.sort_values("exit_date")
        resolved["cum_pnl"] = resolved["pnl_vol_points"].cumsum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=resolved["exit_date"],
                y=resolved["cum_pnl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#1976d2"),
                hovertemplate="Date: %{x}<br>Cum P&L: %{y:.4f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="Cumulative P&L (vol points)",
            xaxis_title="Exit Date",
            yaxis_title="Cumulative P&L",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly summary
    if not resolved.empty and resolved["pnl_vol_points"].notna().any():
        st.markdown("---")
        st.markdown("##### Monthly Summary")

        resolved["month"] = pd.to_datetime(resolved["exit_date"]).dt.to_period("M").astype(str)
        monthly = resolved.groupby("month").agg(
            trades=("pnl_vol_points", "count"),
            wins=("pnl_direction_correct", "sum"),
            total_pnl=("pnl_vol_points", "sum"),
            avg_pnl=("pnl_vol_points", "mean"),
        ).reset_index()
        monthly["win_rate"] = (monthly["wins"] / monthly["trades"] * 100).round(0)
        monthly.columns = ["Month", "Trades", "Wins", "Total P&L", "Avg P&L", "Win Rate %"]

        st.dataframe(
            monthly.style.format({
                "Total P&L": "{:.4f}",
                "Avg P&L": "{:.4f}",
                "Win Rate %": "{:.0f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )
