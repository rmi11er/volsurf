"""Surface metrics history — denormalized daily snapshots for heat map and percentile calculations."""

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.analytics.surface_metrics import get_atm_vol_from_svi, get_vol_at_moneyness
from volsurf.database.connection import get_connection

# Columns that support percentile/z-score queries (validated to prevent SQL injection)
RANKABLE_METRICS = frozenset({
    "atm_vol_30d", "atm_vol_60d", "atm_vol_90d",
    "skew_25d_30d", "skew_25d_60d", "skew_25d_90d",
    "term_slope", "butterfly_30d",
    "rv_21d", "rv_63d", "vrp_30d",
})

# norm.ppf(0.75) — used for 25-delta log-moneyness approximation
_NORM_PPF_75 = 0.6745


@dataclass
class SurfaceMetricsSnapshot:
    """One row of the surface_metrics_history table."""

    symbol: str
    quote_date: date

    atm_vol_30d: Optional[float] = None
    atm_vol_60d: Optional[float] = None
    atm_vol_90d: Optional[float] = None

    skew_25d_30d: Optional[float] = None
    skew_25d_60d: Optional[float] = None
    skew_25d_90d: Optional[float] = None

    term_slope: Optional[float] = None
    butterfly_30d: Optional[float] = None

    rv_21d: Optional[float] = None
    rv_63d: Optional[float] = None
    vrp_30d: Optional[float] = None
    underlying_close: Optional[float] = None

    atm_vol_30d_1d_chg: Optional[float] = None
    skew_25d_30d_1d_chg: Optional[float] = None
    term_slope_1d_chg: Optional[float] = None


class SurfaceMetricsHistoryCalculator:
    """Computes and stores daily surface metric snapshots at standard tenors."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(value: Optional[float]) -> Optional[float]:
        """Convert NaN/inf to None for DB storage."""
        if value is None:
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _interpolate_metric(
        tte_values: np.ndarray, metric_values: np.ndarray, target_tte_years: float,
    ) -> Optional[float]:
        """Interpolate metric to target tenor, flat-extrapolating at boundaries."""
        valid = ~np.isnan(metric_values)
        if not valid.any():
            return None
        tte_v = tte_values[valid]
        met_v = metric_values[valid]
        if target_tte_years <= tte_v[0]:
            return float(met_v[0])
        if target_tte_years >= tte_v[-1]:
            return float(met_v[-1])
        return float(np.interp(target_tte_years, tte_v, met_v))

    @staticmethod
    def _compute_butterfly(
        a: float, b: float, rho: float, m: float, sigma: float, tte_years: float,
    ) -> float:
        """Butterfly = IV(25d put) + IV(25d call) - 2*ATM for one expiration."""
        atm_vol = get_atm_vol_from_svi(a, b, rho, m, sigma, tte_years)
        if np.isnan(atm_vol) or atm_vol <= 0:
            return np.nan
        k_25d = _NORM_PPF_75 * atm_vol * np.sqrt(tte_years)
        iv_put = get_vol_at_moneyness(a, b, rho, m, sigma, tte_years, -k_25d)
        iv_call = get_vol_at_moneyness(a, b, rho, m, sigma, tte_years, k_25d)
        return iv_put + iv_call - 2.0 * atm_vol

    @staticmethod
    def _float_or_none(val) -> Optional[float]:
        if val is None:
            return None
        v = float(val)
        if math.isnan(v):
            return None
        return v

    def _row_to_snapshot(self, row) -> SurfaceMetricsSnapshot:
        """Convert a DataFrame row to SurfaceMetricsSnapshot."""
        f = self._float_or_none
        qd = row["quote_date"]
        if hasattr(qd, "date"):
            qd = qd.date()
        return SurfaceMetricsSnapshot(
            symbol=str(row["symbol"]),
            quote_date=qd,
            atm_vol_30d=f(row.get("atm_vol_30d")),
            atm_vol_60d=f(row.get("atm_vol_60d")),
            atm_vol_90d=f(row.get("atm_vol_90d")),
            skew_25d_30d=f(row.get("skew_25d_30d")),
            skew_25d_60d=f(row.get("skew_25d_60d")),
            skew_25d_90d=f(row.get("skew_25d_90d")),
            term_slope=f(row.get("term_slope")),
            butterfly_30d=f(row.get("butterfly_30d")),
            rv_21d=f(row.get("rv_21d")),
            rv_63d=f(row.get("rv_63d")),
            vrp_30d=f(row.get("vrp_30d")),
            underlying_close=f(row.get("underlying_close")),
            atm_vol_30d_1d_chg=f(row.get("atm_vol_30d_1d_chg")),
            skew_25d_30d_1d_chg=f(row.get("skew_25d_30d_1d_chg")),
            term_slope_1d_chg=f(row.get("term_slope_1d_chg")),
        )

    @staticmethod
    def _validate_metric(metric: str) -> None:
        if metric not in RANKABLE_METRICS:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {sorted(RANKABLE_METRICS)}")

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def calculate_for_date(self, symbol: str, target_date: date) -> SurfaceMetricsSnapshot:
        """Calculate surface metrics snapshot for a single date."""
        conn = get_connection()
        snap = SurfaceMetricsSnapshot(symbol=symbol, quote_date=target_date)

        # Fitted surfaces for the date
        df = conn.execute(
            """
            SELECT tte_years, atm_vol, skew_25delta,
                   svi_a, svi_b, svi_rho, svi_m, svi_sigma
            FROM fitted_surfaces
            WHERE symbol = ? AND quote_date = ? AND atm_vol IS NOT NULL
            ORDER BY tte_years
            """,
            [symbol, target_date],
        ).fetchdf()

        if df.empty:
            return snap

        tte = df["tte_years"].values.astype(float)
        atm_vols = df["atm_vol"].values.astype(float)
        skews = df["skew_25delta"].values.astype(float)

        # Butterfly per expiration
        butterflies = np.array([
            self._compute_butterfly(
                float(r["svi_a"]), float(r["svi_b"]), float(r["svi_rho"]),
                float(r["svi_m"]), float(r["svi_sigma"]), float(r["tte_years"]),
            )
            for _, r in df.iterrows()
        ])

        # Interpolate to standard tenors
        for tenor_days, atm_attr, skew_attr in [
            (30, "atm_vol_30d", "skew_25d_30d"),
            (60, "atm_vol_60d", "skew_25d_60d"),
            (90, "atm_vol_90d", "skew_25d_90d"),
        ]:
            target_tte = tenor_days / 252.0
            setattr(snap, atm_attr, self._sanitize(self._interpolate_metric(tte, atm_vols, target_tte)))
            setattr(snap, skew_attr, self._sanitize(self._interpolate_metric(tte, skews, target_tte)))

        snap.butterfly_30d = self._sanitize(self._interpolate_metric(tte, butterflies, 30 / 252.0))

        if snap.atm_vol_90d is not None and snap.atm_vol_30d is not None:
            snap.term_slope = self._sanitize(snap.atm_vol_90d - snap.atm_vol_30d)

        # Cross-reference: realized vol
        rv_row = conn.execute(
            "SELECT rv_21d, rv_63d FROM realized_volatility WHERE symbol = ? AND date = ?",
            [symbol, target_date],
        ).fetchone()
        if rv_row:
            snap.rv_21d = self._sanitize(self._float_or_none(rv_row[0]))
            snap.rv_63d = self._sanitize(self._float_or_none(rv_row[1]))

        # Cross-reference: VRP
        vrp_row = conn.execute(
            "SELECT vrp_30d FROM vrp_metrics WHERE symbol = ? AND date = ?",
            [symbol, target_date],
        ).fetchone()
        if vrp_row and vrp_row[0] is not None:
            snap.vrp_30d = self._sanitize(float(vrp_row[0]))

        # Cross-reference: underlying close
        price_row = conn.execute(
            "SELECT close FROM underlying_prices WHERE symbol = ? AND date = ?",
            [symbol, target_date],
        ).fetchone()
        if price_row and price_row[0] is not None:
            snap.underlying_close = float(price_row[0])

        return snap

    def calculate_day_over_day(
        self, current: SurfaceMetricsSnapshot, previous: Optional[SurfaceMetricsSnapshot],
    ) -> None:
        """Fill _1d_chg fields on *current* by diffing against *previous*."""
        if previous is None:
            return
        if current.atm_vol_30d is not None and previous.atm_vol_30d is not None:
            current.atm_vol_30d_1d_chg = current.atm_vol_30d - previous.atm_vol_30d
        if current.skew_25d_30d is not None and previous.skew_25d_30d is not None:
            current.skew_25d_30d_1d_chg = current.skew_25d_30d - previous.skew_25d_30d
        if current.term_slope is not None and previous.term_slope is not None:
            current.term_slope_1d_chg = current.term_slope - previous.term_slope

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_result(self, snap: SurfaceMetricsSnapshot) -> None:
        """Upsert a snapshot into surface_metrics_history."""
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO surface_metrics_history (
                smh_id, symbol, quote_date,
                atm_vol_30d, atm_vol_60d, atm_vol_90d,
                skew_25d_30d, skew_25d_60d, skew_25d_90d,
                term_slope, butterfly_30d,
                rv_21d, rv_63d, vrp_30d, underlying_close,
                atm_vol_30d_1d_chg, skew_25d_30d_1d_chg, term_slope_1d_chg,
                calculation_timestamp
            ) VALUES (
                nextval('seq_smh_id'), ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                now()
            )
            ON CONFLICT (symbol, quote_date) DO UPDATE SET
                atm_vol_30d = excluded.atm_vol_30d,
                atm_vol_60d = excluded.atm_vol_60d,
                atm_vol_90d = excluded.atm_vol_90d,
                skew_25d_30d = excluded.skew_25d_30d,
                skew_25d_60d = excluded.skew_25d_60d,
                skew_25d_90d = excluded.skew_25d_90d,
                term_slope = excluded.term_slope,
                butterfly_30d = excluded.butterfly_30d,
                rv_21d = excluded.rv_21d,
                rv_63d = excluded.rv_63d,
                vrp_30d = excluded.vrp_30d,
                underlying_close = excluded.underlying_close,
                atm_vol_30d_1d_chg = excluded.atm_vol_30d_1d_chg,
                skew_25d_30d_1d_chg = excluded.skew_25d_30d_1d_chg,
                term_slope_1d_chg = excluded.term_slope_1d_chg,
                calculation_timestamp = now()
            """,
            [
                snap.symbol, snap.quote_date,
                snap.atm_vol_30d, snap.atm_vol_60d, snap.atm_vol_90d,
                snap.skew_25d_30d, snap.skew_25d_60d, snap.skew_25d_90d,
                snap.term_slope, snap.butterfly_30d,
                snap.rv_21d, snap.rv_63d, snap.vrp_30d, snap.underlying_close,
                snap.atm_vol_30d_1d_chg, snap.skew_25d_30d_1d_chg, snap.term_slope_1d_chg,
            ],
        )

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(
        self, symbol: str, start_date: date, end_date: date, store: bool = True,
    ) -> int:
        """Calculate and store snapshots for a date range."""
        conn = get_connection()

        dates_df = conn.execute(
            """
            SELECT DISTINCT quote_date
            FROM fitted_surfaces
            WHERE symbol = ? AND quote_date >= ? AND quote_date <= ?
            ORDER BY quote_date
            """,
            [symbol, start_date, end_date],
        ).fetchdf()

        if dates_df.empty:
            logger.warning(f"No fitted surfaces for {symbol} in range")
            return 0

        # Seed previous snapshot for day-over-day on first date
        first_date = dates_df["quote_date"].iloc[0]
        if hasattr(first_date, "date"):
            first_date = first_date.date()
        prev_snap = self.get_latest_snapshot(symbol, before_date=first_date)

        count = 0
        for _, row in dates_df.iterrows():
            td = row["quote_date"]
            if hasattr(td, "date"):
                td = td.date()

            snap = self.calculate_for_date(symbol, td)

            if snap.atm_vol_30d is None:
                prev_snap = None
                continue

            self.calculate_day_over_day(snap, prev_snap)

            if store:
                self.store_result(snap)

            prev_snap = snap
            count += 1

        logger.info(f"Surface metrics history for {symbol}: {count} dates")
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_timeseries(
        self, symbol: str, start_date: date, end_date: date,
    ) -> pd.DataFrame:
        """Get surface metrics time series from the database."""
        conn = get_connection()
        return conn.execute(
            """
            SELECT * FROM surface_metrics_history
            WHERE symbol = ? AND quote_date >= ? AND quote_date <= ?
            ORDER BY quote_date
            """,
            [symbol, start_date, end_date],
        ).fetchdf()

    def get_latest_snapshot(
        self, symbol: str, before_date: Optional[date] = None,
    ) -> Optional[SurfaceMetricsSnapshot]:
        """Get the most recent snapshot for a symbol, optionally before a date."""
        conn = get_connection()
        if before_date:
            df = conn.execute(
                """
                SELECT * FROM surface_metrics_history
                WHERE symbol = ? AND quote_date < ?
                ORDER BY quote_date DESC LIMIT 1
                """,
                [symbol, before_date],
            ).fetchdf()
        else:
            df = conn.execute(
                """
                SELECT * FROM surface_metrics_history
                WHERE symbol = ?
                ORDER BY quote_date DESC LIMIT 1
                """,
                [symbol],
            ).fetchdf()
        if df.empty:
            return None
        return self._row_to_snapshot(df.iloc[0])

    def get_all_symbols_latest(self) -> pd.DataFrame:
        """Latest snapshot per symbol (one row each). Powers the heat map base layer."""
        conn = get_connection()
        return conn.execute(
            """
            SELECT s.*
            FROM surface_metrics_history s
            INNER JOIN (
                SELECT symbol, MAX(quote_date) AS max_date
                FROM surface_metrics_history
                GROUP BY symbol
            ) latest ON s.symbol = latest.symbol AND s.quote_date = latest.max_date
            ORDER BY s.symbol
            """
        ).fetchdf()

    # ------------------------------------------------------------------
    # Percentile / z-score (single-symbol helpers)
    # ------------------------------------------------------------------

    def calculate_percentile_rank(
        self,
        symbol: str,
        metric: str,
        current_value: float,
        target_date: date,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """Percentile rank (0-100) of *current_value* vs trailing *lookback_days* trading days."""
        self._validate_metric(metric)
        conn = get_connection()
        # Use ROW_NUMBER to get exactly `lookback_days` trading days before target_date
        result = conn.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE val < ?) * 100.0 / NULLIF(COUNT(val), 0)
            FROM (
                SELECT {metric} AS val FROM (
                    SELECT {metric},
                           ROW_NUMBER() OVER (ORDER BY quote_date DESC) AS rn
                    FROM surface_metrics_history
                    WHERE symbol = ? AND quote_date < ? AND {metric} IS NOT NULL
                ) WHERE rn <= ?
            )
            """,
            [current_value, symbol, target_date, lookback_days],
        ).fetchone()
        if result is None or result[0] is None:
            return None
        return float(result[0])

    def calculate_zscore(
        self,
        symbol: str,
        metric: str,
        current_value: float,
        target_date: date,
        lookback_days: int = 252,
    ) -> Optional[float]:
        """Z-score of *current_value* relative to trailing *lookback_days* trading days."""
        self._validate_metric(metric)
        conn = get_connection()
        result = conn.execute(
            f"""
            SELECT AVG(val), STDDEV_SAMP(val)
            FROM (
                SELECT {metric} AS val FROM (
                    SELECT {metric},
                           ROW_NUMBER() OVER (ORDER BY quote_date DESC) AS rn
                    FROM surface_metrics_history
                    WHERE symbol = ? AND quote_date < ? AND {metric} IS NOT NULL
                ) WHERE rn <= ?
            )
            """,
            [symbol, target_date, lookback_days],
        ).fetchone()
        if result is None or result[0] is None or result[1] is None or float(result[1]) == 0:
            return None
        return (current_value - float(result[0])) / float(result[1])

    # ------------------------------------------------------------------
    # Enriched views (power the UI)
    # ------------------------------------------------------------------

    def get_enriched_snapshot(
        self,
        symbol: str,
        target_date: date,
        lookback_windows: list[int] | None = None,
    ) -> Optional[dict]:
        """Snapshot + percentile ranks + z-scores at multiple lookback windows.

        Returns dict with keys: snapshot, percentiles, zscores.
        Each of percentiles/zscores maps metric_name → {window: value}.
        """
        if lookback_windows is None:
            lookback_windows = [30, 60, 90, 252]

        conn = get_connection()
        df = conn.execute(
            "SELECT * FROM surface_metrics_history WHERE symbol = ? AND quote_date = ?",
            [symbol, target_date],
        ).fetchdf()
        if df.empty:
            return None
        snap = self._row_to_snapshot(df.iloc[0])

        key_metrics = ["atm_vol_30d", "skew_25d_30d", "term_slope", "butterfly_30d", "vrp_30d"]
        percentiles: dict[str, dict[int, Optional[float]]] = {}
        zscores: dict[str, dict[int, Optional[float]]] = {}

        for metric in key_metrics:
            val = getattr(snap, metric, None)
            if val is None:
                continue
            p: dict[int, Optional[float]] = {}
            z: dict[int, Optional[float]] = {}
            for w in lookback_windows:
                p[w] = self.calculate_percentile_rank(symbol, metric, val, target_date, w)
                z[w] = self.calculate_zscore(symbol, metric, val, target_date, w)
            percentiles[metric] = p
            zscores[metric] = z

        return {"snapshot": snap, "percentiles": percentiles, "zscores": zscores}

    def get_watchlist_heatmap_data(
        self,
        symbols: list[str],
        target_date: date,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Latest snapshot + percentiles + z-scores for all symbols.

        One efficient query per metric (5 total), processing all symbols in parallel via
        ROW_NUMBER window function in DuckDB.
        """
        conn = get_connection()
        placeholders = ", ".join(["?"] * len(symbols))

        # Base snapshots for the target date
        base_df = conn.execute(
            f"""
            SELECT * FROM surface_metrics_history
            WHERE symbol IN ({placeholders}) AND quote_date = ?
            ORDER BY symbol
            """,
            [*symbols, target_date],
        ).fetchdf()

        if base_df.empty:
            return pd.DataFrame()

        # Batch compute percentiles + z-scores per metric across all symbols
        key_metrics = ["atm_vol_30d", "skew_25d_30d", "term_slope", "butterfly_30d", "vrp_30d"]

        for metric in key_metrics:
            stats_df = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT symbol, {metric} AS val,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol ORDER BY quote_date DESC
                           ) AS rn
                    FROM surface_metrics_history
                    WHERE symbol IN ({placeholders})
                      AND quote_date < ?
                      AND {metric} IS NOT NULL
                ),
                hist AS (
                    SELECT symbol, val FROM ranked WHERE rn <= ?
                ),
                current_vals AS (
                    SELECT symbol, {metric} AS val
                    FROM surface_metrics_history
                    WHERE symbol IN ({placeholders}) AND quote_date = ?
                      AND {metric} IS NOT NULL
                )
                SELECT
                    c.symbol,
                    COUNT(h.val) FILTER (WHERE h.val < c.val) * 100.0
                        / NULLIF(COUNT(h.val), 0) AS pctile,
                    CASE WHEN STDDEV_SAMP(h.val) > 0
                         THEN (c.val - AVG(h.val)) / STDDEV_SAMP(h.val)
                    END AS zscore
                FROM current_vals c
                LEFT JOIN hist h ON h.symbol = c.symbol
                GROUP BY c.symbol, c.val
                """,
                [*symbols, target_date, lookback_days, *symbols, target_date],
            ).fetchdf()

            pctile_col = f"{metric}_pctile"
            zscore_col = f"{metric}_zscore"
            base_df[pctile_col] = None
            base_df[zscore_col] = None

            if not stats_df.empty:
                stats_map = {row["symbol"]: row for _, row in stats_df.iterrows()}
                for idx, row in base_df.iterrows():
                    sym = row["symbol"]
                    if sym in stats_map:
                        s = stats_map[sym]
                        base_df.at[idx, pctile_col] = self._float_or_none(s["pctile"])
                        base_df.at[idx, zscore_col] = self._float_or_none(s["zscore"])

        return base_df
