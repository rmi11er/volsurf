"""Variance Risk Premium (VRP) calculation."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.analytics.realized_vol import RealizedVolCalculator
from volsurf.database.connection import get_connection


@dataclass
class VRPResult:
    """Result of VRP calculation."""

    symbol: str
    date: date
    vrp_30d: Optional[float] = None
    vrp_60d: Optional[float] = None
    vrp_90d: Optional[float] = None
    implied_vol_30d: Optional[float] = None
    realized_vol_30d: Optional[float] = None
    vrp_zscore: Optional[float] = None


class VRPCalculator:
    """Calculator for Variance Risk Premium metrics."""

    def __init__(self, rv_calculator: Optional[RealizedVolCalculator] = None):
        self.rv_calculator = rv_calculator or RealizedVolCalculator()

    def get_atm_implied_vol(
        self, symbol: str, target_date: date, target_tte_days: int = 30
    ) -> Optional[float]:
        """
        Get ATM implied volatility for a target tenor.

        Uses linear interpolation between available expirations.

        Args:
            symbol: Ticker symbol
            target_date: Quote date
            target_tte_days: Target time to expiration in days

        Returns:
            Interpolated ATM implied vol, or None if not available
        """
        conn = get_connection()

        # Get fitted surfaces for the date, sorted by TTE
        query = """
            SELECT tte_years, atm_vol
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date = ?
              AND atm_vol IS NOT NULL
            ORDER BY tte_years
        """
        df = conn.execute(query, [symbol, target_date]).fetchdf()

        if df.empty:
            return None

        target_tte_years = target_tte_days / 252.0

        # If we have exact match
        tte_values = df["tte_years"].values
        atm_vols = df["atm_vol"].values

        # Linear interpolation
        if target_tte_years <= tte_values[0]:
            # Extrapolate using first available
            return float(atm_vols[0])
        elif target_tte_years >= tte_values[-1]:
            # Extrapolate using last available
            return float(atm_vols[-1])
        else:
            # Interpolate
            return float(np.interp(target_tte_years, tte_values, atm_vols))

    def get_realized_vol(
        self, symbol: str, target_date: date, window_days: int = 21
    ) -> Optional[float]:
        """
        Get realized volatility for a specific window.

        Args:
            symbol: Ticker symbol
            target_date: Date to get RV for
            window_days: Window size (10, 21, 63, or 252)

        Returns:
            Realized vol or None if not available
        """
        conn = get_connection()

        # Map window to column name
        window_col_map = {10: "rv_10d", 21: "rv_21d", 63: "rv_63d", 252: "rv_252d"}
        col = window_col_map.get(window_days, "rv_21d")

        query = f"""
            SELECT {col} as rv
            FROM realized_volatility
            WHERE symbol = ?
              AND date = ?
        """
        df = conn.execute(query, [symbol, target_date]).fetchdf()

        if df.empty or df["rv"].iloc[0] is None:
            return None

        return float(df["rv"].iloc[0])

    def calculate_vrp_zscore(
        self, symbol: str, target_date: date, lookback_days: int = 252
    ) -> Optional[float]:
        """
        Calculate z-score of current VRP vs historical.

        Args:
            symbol: Ticker symbol
            target_date: Date to calculate for
            lookback_days: Historical window for mean/std

        Returns:
            Z-score or None if insufficient data
        """
        conn = get_connection()

        # Get historical VRP values
        query = """
            SELECT vrp_30d
            FROM vrp_metrics
            WHERE symbol = ?
              AND date < ?
              AND vrp_30d IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
        """
        df = conn.execute(query, [symbol, target_date, lookback_days]).fetchdf()

        if len(df) < 20:  # Require minimum history
            return None

        historical_vrp = df["vrp_30d"].values
        mean_vrp = np.mean(historical_vrp)
        std_vrp = np.std(historical_vrp)

        if std_vrp == 0:
            return None

        # Get current VRP
        current_query = """
            SELECT vrp_30d
            FROM vrp_metrics
            WHERE symbol = ?
              AND date = ?
        """
        current_df = conn.execute(current_query, [symbol, target_date]).fetchdf()

        if current_df.empty or current_df["vrp_30d"].iloc[0] is None:
            return None

        current_vrp = current_df["vrp_30d"].iloc[0]
        return float((current_vrp - mean_vrp) / std_vrp)

    def calculate_for_date(
        self, symbol: str, target_date: date, compute_zscore: bool = False
    ) -> VRPResult:
        """
        Calculate VRP metrics for a specific date.

        VRP = Implied Vol - Realized Vol

        Args:
            symbol: Ticker symbol
            target_date: Date to calculate for
            compute_zscore: Whether to compute z-score (requires history)

        Returns:
            VRPResult with all metrics
        """
        result = VRPResult(symbol=symbol, date=target_date)

        # Get implied vol at different tenors
        iv_30d = self.get_atm_implied_vol(symbol, target_date, 30)
        iv_60d = self.get_atm_implied_vol(symbol, target_date, 60)
        iv_90d = self.get_atm_implied_vol(symbol, target_date, 90)

        # Get realized vol (use 21d for 30d IV, 63d for 90d IV)
        rv_21d = self.get_realized_vol(symbol, target_date, 21)
        rv_63d = self.get_realized_vol(symbol, target_date, 63)

        # Calculate VRP at each horizon
        if iv_30d is not None and rv_21d is not None:
            result.vrp_30d = iv_30d - rv_21d
            result.implied_vol_30d = iv_30d
            result.realized_vol_30d = rv_21d

        if iv_60d is not None and rv_21d is not None:
            result.vrp_60d = iv_60d - rv_21d

        if iv_90d is not None and rv_63d is not None:
            result.vrp_90d = iv_90d - rv_63d

        return result

    def store_result(self, result: VRPResult) -> None:
        """Store VRP result in database."""
        conn = get_connection()

        conn.execute(
            """
            INSERT INTO vrp_metrics (
                vrp_id, symbol, date,
                vrp_30d, vrp_60d, vrp_90d,
                implied_vol_30d, realized_vol_30d,
                vrp_zscore,
                calculation_timestamp
            ) VALUES (
                nextval('seq_vrp_id'), ?, ?,
                ?, ?, ?,
                ?, ?,
                ?,
                now()
            )
            ON CONFLICT (symbol, date) DO UPDATE SET
                vrp_30d = excluded.vrp_30d,
                vrp_60d = excluded.vrp_60d,
                vrp_90d = excluded.vrp_90d,
                implied_vol_30d = excluded.implied_vol_30d,
                realized_vol_30d = excluded.realized_vol_30d,
                vrp_zscore = excluded.vrp_zscore,
                calculation_timestamp = now()
            """,
            [
                result.symbol,
                result.date,
                result.vrp_30d,
                result.vrp_60d,
                result.vrp_90d,
                result.implied_vol_30d,
                result.realized_vol_30d,
                result.vrp_zscore,
            ],
        )

    def backfill(
        self, symbol: str, start_date: date, end_date: date, store: bool = True
    ) -> int:
        """
        Calculate and optionally store VRP for a date range.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range
            store: Whether to store results in database

        Returns:
            Number of dates processed
        """
        conn = get_connection()

        # Get all dates with fitted surfaces in range
        query = """
            SELECT DISTINCT quote_date
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date >= ?
              AND quote_date <= ?
            ORDER BY quote_date
        """
        dates_df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()

        if dates_df.empty:
            logger.warning(f"No fitted surfaces found for {symbol} in date range")
            return 0

        count = 0
        for _, row in dates_df.iterrows():
            target_date = row["quote_date"]
            if hasattr(target_date, "date"):
                target_date = target_date.date()

            result = self.calculate_for_date(symbol, target_date)

            # Only store if we have at least some data
            if result.vrp_30d is not None and store:
                self.store_result(result)
                count += 1

        # Update z-scores in a second pass (needs history)
        if store and count > 0:
            self._update_zscores(symbol, start_date, end_date)

        logger.info(f"Calculated VRP for {count} dates")
        return count

    def _update_zscores(self, symbol: str, start_date: date, end_date: date) -> None:
        """Update z-scores for stored VRP records."""
        conn = get_connection()

        # Get all dates in range
        query = """
            SELECT date
            FROM vrp_metrics
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        dates_df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()

        for _, row in dates_df.iterrows():
            target_date = row["date"]
            if hasattr(target_date, "date"):
                target_date = target_date.date()

            zscore = self.calculate_vrp_zscore(symbol, target_date)
            if zscore is not None:
                conn.execute(
                    """
                    UPDATE vrp_metrics
                    SET vrp_zscore = ?
                    WHERE symbol = ? AND date = ?
                    """,
                    [zscore, symbol, target_date],
                )

    def get_vrp_timeseries(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Get VRP time series from database.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range

        Returns:
            DataFrame with VRP metrics
        """
        conn = get_connection()
        query = """
            SELECT date, vrp_30d, vrp_60d, vrp_90d,
                   implied_vol_30d, realized_vol_30d, vrp_zscore
            FROM vrp_metrics
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        return conn.execute(query, [symbol, start_date, end_date]).fetchdf()
