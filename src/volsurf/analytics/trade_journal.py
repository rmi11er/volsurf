"""Trade Journal manager for logging and tracking hypothetical trades."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection


VALID_CATEGORIES = frozenset({
    "surface_anomaly",
    "event_mispricing",
    "rv_vs_iv",
    "skew_trade",
    "term_structure",
    "directional_vol",
})

VALID_DIRECTIONS = frozenset({
    "long_vol",
    "short_vol",
    "long_skew",
    "short_skew",
    "calendar",
    "other",
})

VALID_OUTCOMES = frozenset({"win", "loss", "scratch", "open"})


@dataclass
class TradeEntry:
    """Input for creating a new trade."""

    symbol: str
    thesis: str
    category: str
    direction: str
    entry_date: Optional[date] = None  # defaults to today
    structure_description: Optional[str] = None
    entry_iv_level: Optional[float] = None
    entry_skew_level: Optional[float] = None
    entry_underlying_price: Optional[float] = None
    entry_metric_value: Optional[float] = None
    entry_metric_name: Optional[str] = None
    target_description: Optional[str] = None
    stop_description: Optional[str] = None
    time_horizon_days: Optional[int] = None
    tags: Optional[str] = None


@dataclass
class TradeResolution:
    """Input for resolving a trade."""

    trade_id: int
    outcome: str  # 'win', 'loss', 'scratch'
    exit_date: Optional[date] = None  # defaults to today
    pnl_vol_points: Optional[float] = None
    outcome_notes: Optional[str] = None


class TradeJournalManager:
    """Manager for trade journal CRUD and analytics."""

    def create_trade(self, entry: TradeEntry) -> int:
        """Create a new trade journal entry.

        Auto-populates market context from surface_metrics_history.

        Returns:
            trade_id of the created trade.
        """
        if entry.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{entry.category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )
        if entry.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{entry.direction}'. Must be one of: {sorted(VALID_DIRECTIONS)}"
            )

        entry_date = entry.entry_date or date.today()
        conn = get_connection()

        # Auto-populate context from surface_metrics_history
        atm_vol_pctile = None
        skew_zscore = None
        vrp = None
        iv_level = entry.entry_iv_level
        skew_level = entry.entry_skew_level
        underlying_price = entry.entry_underlying_price

        try:
            from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator

            smh = SurfaceMetricsHistoryCalculator()
            enriched = smh.get_enriched_snapshot(entry.symbol, entry_date)
            if enriched:
                snap = enriched["snapshot"]
                pctiles = enriched.get("percentiles", {})

                # Fill in entry levels if not provided
                if iv_level is None and snap.atm_vol_30d is not None:
                    iv_level = snap.atm_vol_30d
                if skew_level is None and snap.skew_25d_30d is not None:
                    skew_level = snap.skew_25d_30d
                if underlying_price is None and snap.underlying_close is not None:
                    underlying_price = snap.underlying_close

                # Context fields
                p_dict = pctiles.get("atm_vol_30d", {})
                atm_vol_pctile = p_dict.get(252)
                z_dict = enriched.get("zscores", {})
                skew_zscore = z_dict.get("skew_25d_30d", {}).get(252)
                vrp = snap.vrp_30d
        except Exception as e:
            logger.debug(f"Could not auto-populate context: {e}")

        # Insert
        result = conn.execute(
            """
            INSERT INTO trade_journal (
                trade_id, entry_date, symbol, thesis, category, direction,
                structure_description,
                entry_iv_level, entry_skew_level, entry_underlying_price,
                entry_metric_value, entry_metric_name,
                target_description, stop_description, time_horizon_days,
                entry_atm_vol_pctile, entry_skew_zscore, entry_vrp,
                tags, outcome
            ) VALUES (
                nextval('seq_trade_id'), ?, ?, ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, 'open'
            )
            RETURNING trade_id
            """,
            [
                entry_date,
                entry.symbol,
                entry.thesis,
                entry.category,
                entry.direction,
                entry.structure_description,
                iv_level,
                skew_level,
                underlying_price,
                entry.entry_metric_value,
                entry.entry_metric_name,
                entry.target_description,
                entry.stop_description,
                entry.time_horizon_days,
                atm_vol_pctile,
                skew_zscore,
                vrp,
                entry.tags,
            ],
        ).fetchone()

        trade_id = result[0]
        logger.info(f"Created trade #{trade_id}: {entry.symbol} {entry.direction} ({entry.category})")
        return trade_id

    def resolve_trade(self, resolution: TradeResolution) -> None:
        """Resolve an open trade."""
        if resolution.outcome not in VALID_OUTCOMES or resolution.outcome == "open":
            raise ValueError(
                f"Invalid outcome '{resolution.outcome}'. Must be one of: win, loss, scratch"
            )

        exit_date = resolution.exit_date or date.today()
        conn = get_connection()

        # Get the trade to find symbol
        trade = conn.execute(
            "SELECT symbol, entry_iv_level, entry_skew_level, direction FROM trade_journal WHERE trade_id = ?",
            [resolution.trade_id],
        ).fetchone()

        if trade is None:
            raise ValueError(f"Trade #{resolution.trade_id} not found")

        symbol = trade[0]
        entry_iv = trade[1]
        entry_skew = trade[2]
        direction = trade[3]

        # Get exit market levels
        exit_iv = None
        exit_skew = None
        exit_underlying = None
        pnl_vol_points = resolution.pnl_vol_points
        pnl_direction_correct = None

        try:
            from volsurf.analytics.surface_metrics_history import SurfaceMetricsHistoryCalculator

            smh = SurfaceMetricsHistoryCalculator()
            snap_data = smh.get_latest_snapshot(symbol, before_date=exit_date + timedelta(days=1))
            if snap_data:
                exit_iv = snap_data.atm_vol_30d
                exit_skew = snap_data.skew_25d_30d
                exit_underlying = snap_data.underlying_close

                # Auto-compute P&L in vol points if not provided
                if pnl_vol_points is None and entry_iv is not None and exit_iv is not None:
                    if direction in ("long_vol", "short_vol"):
                        vol_diff = exit_iv - entry_iv
                        pnl_vol_points = vol_diff if direction == "long_vol" else -vol_diff
                    elif direction in ("long_skew", "short_skew"):
                        if entry_skew is not None and exit_skew is not None:
                            skew_diff = exit_skew - entry_skew
                            pnl_vol_points = skew_diff if direction == "long_skew" else -skew_diff

                # Direction correctness
                if pnl_vol_points is not None:
                    pnl_direction_correct = pnl_vol_points > 0
        except Exception as e:
            logger.debug(f"Could not auto-populate exit context: {e}")

        conn.execute(
            """
            UPDATE trade_journal SET
                exit_date = ?,
                exit_iv_level = ?,
                exit_skew_level = ?,
                exit_underlying_price = ?,
                exit_metric_value = ?,
                pnl_vol_points = ?,
                pnl_direction_correct = ?,
                outcome = ?,
                outcome_notes = ?,
                updated_timestamp = now()
            WHERE trade_id = ?
            """,
            [
                exit_date,
                exit_iv,
                exit_skew,
                exit_underlying,
                None,  # exit_metric_value — set manually if needed
                pnl_vol_points,
                pnl_direction_correct,
                resolution.outcome,
                resolution.outcome_notes,
                resolution.trade_id,
            ],
        )
        logger.info(f"Resolved trade #{resolution.trade_id}: {resolution.outcome}")

    def auto_resolve_expired(self) -> int:
        """Auto-resolve open trades past their time horizon.

        Returns:
            Count of trades auto-resolved.
        """
        conn = get_connection()
        today = date.today()

        expired = conn.execute(
            """
            SELECT trade_id, symbol, entry_date, time_horizon_days, direction,
                   entry_iv_level, entry_skew_level
            FROM trade_journal
            WHERE outcome = 'open'
              AND time_horizon_days IS NOT NULL
              AND entry_date + INTERVAL (time_horizon_days) DAY <= ?
            """,
            [today],
        ).fetchall()

        count = 0
        for row in expired:
            trade_id = row[0]
            try:
                # Determine outcome based on P&L direction
                resolution = TradeResolution(
                    trade_id=trade_id,
                    outcome="scratch",  # default; updated below if we can compute P&L
                    exit_date=today,
                    outcome_notes="Auto-resolved: time horizon expired",
                )
                self.resolve_trade(resolution)

                # Re-check: if pnl was computed, update outcome
                result = conn.execute(
                    "SELECT pnl_vol_points FROM trade_journal WHERE trade_id = ?",
                    [trade_id],
                ).fetchone()
                if result and result[0] is not None:
                    pnl = float(result[0])
                    if abs(pnl) < 0.001:  # less than 0.1 vol point
                        outcome = "scratch"
                    elif pnl > 0:
                        outcome = "win"
                    else:
                        outcome = "loss"
                    conn.execute(
                        "UPDATE trade_journal SET outcome = ? WHERE trade_id = ?",
                        [outcome, trade_id],
                    )

                count += 1
            except Exception as e:
                logger.warning(f"Failed to auto-resolve trade #{trade_id}: {e}")

        logger.info(f"Auto-resolved {count} expired trades")
        return count

    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades."""
        conn = get_connection()
        return conn.execute(
            """
            SELECT trade_id, entry_date, symbol, category, direction,
                   thesis, structure_description,
                   entry_iv_level, entry_skew_level, entry_underlying_price,
                   time_horizon_days, tags,
                   entry_atm_vol_pctile, entry_vrp
            FROM trade_journal
            WHERE outcome = 'open'
            ORDER BY entry_date DESC
            """
        ).fetchdf()

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        category: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Get trade history with optional filters."""
        conn = get_connection()

        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if start_date:
            conditions.append("entry_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("entry_date <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        return conn.execute(
            f"""
            SELECT trade_id, entry_date, exit_date, symbol, category, direction,
                   thesis, outcome, pnl_vol_points, pnl_direction_correct,
                   entry_iv_level, exit_iv_level,
                   entry_underlying_price, exit_underlying_price,
                   time_horizon_days, tags, outcome_notes
            FROM trade_journal
            {where}
            ORDER BY entry_date DESC
            LIMIT ?
            """,
            params,
        ).fetchdf()

    def get_performance_summary(
        self,
        category: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> dict:
        """Get performance summary statistics.

        Returns:
            Dict with total_trades, wins, losses, scratches, open,
            win_rate, avg_pnl, avg_win, avg_loss.
        """
        conn = get_connection()

        conditions = ["1=1"]
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        where = " AND ".join(conditions)

        row = conn.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                COUNT(*) FILTER (WHERE outcome = 'loss') as losses,
                COUNT(*) FILTER (WHERE outcome = 'scratch') as scratches,
                COUNT(*) FILTER (WHERE outcome = 'open') as open_count,
                AVG(pnl_vol_points) FILTER (WHERE outcome != 'open') as avg_pnl,
                AVG(pnl_vol_points) FILTER (WHERE outcome = 'win') as avg_win,
                AVG(pnl_vol_points) FILTER (WHERE outcome = 'loss') as avg_loss
            FROM trade_journal
            WHERE {where}
            """,
            params,
        ).fetchone()

        total = row[0]
        wins = row[1]
        losses = row[2]
        scratches = row[3]
        open_count = row[4]
        resolved = wins + losses + scratches

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "open": open_count,
            "win_rate": wins / resolved if resolved > 0 else None,
            "avg_pnl": float(row[5]) if row[5] is not None else None,
            "avg_win": float(row[6]) if row[6] is not None else None,
            "avg_loss": float(row[7]) if row[7] is not None else None,
        }

    def get_category_performance(self) -> pd.DataFrame:
        """Get performance breakdown by category."""
        conn = get_connection()
        return conn.execute(
            """
            SELECT
                category,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                COUNT(*) FILTER (WHERE outcome = 'loss') as losses,
                COUNT(*) FILTER (WHERE outcome = 'scratch') as scratches,
                COUNT(*) FILTER (WHERE outcome = 'open') as open_count,
                ROUND(
                    COUNT(*) FILTER (WHERE outcome = 'win') * 100.0 /
                    NULLIF(COUNT(*) FILTER (WHERE outcome != 'open'), 0),
                    1
                ) as win_rate_pct,
                ROUND(AVG(pnl_vol_points) FILTER (WHERE outcome != 'open'), 6) as avg_pnl
            FROM trade_journal
            GROUP BY category
            ORDER BY total DESC
            """
        ).fetchdf()

    def get_trade(self, trade_id: int) -> Optional[dict]:
        """Get a single trade by ID."""
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM trade_journal WHERE trade_id = ?",
            [trade_id],
        ).fetchone()

        if row is None:
            return None

        columns = [desc[0] for desc in conn.description]
        return dict(zip(columns, row))
