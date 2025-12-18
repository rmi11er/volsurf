"""Realized volatility calculation."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection


@dataclass
class RealizedVolResult:
    """Result of realized volatility calculation."""

    symbol: str
    date: date
    rv_10d: Optional[float] = None
    rv_21d: Optional[float] = None
    rv_63d: Optional[float] = None
    rv_252d: Optional[float] = None
    parkinson_10d: Optional[float] = None
    parkinson_21d: Optional[float] = None
    gk_10d: Optional[float] = None
    gk_21d: Optional[float] = None


def calculate_close_to_close_vol(
    returns: np.ndarray, annualization_factor: int = 252
) -> float:
    """
    Calculate close-to-close realized volatility.

    Formula: RV = sqrt(annualization * sum(r_t^2) / N)

    Args:
        returns: Array of log returns
        annualization_factor: Trading days per year (default 252)

    Returns:
        Annualized volatility as a decimal (e.g., 0.15 for 15%)
    """
    if len(returns) == 0:
        return np.nan
    variance = np.mean(returns**2)
    return np.sqrt(variance * annualization_factor)


def calculate_parkinson_vol(
    high: np.ndarray, low: np.ndarray, annualization_factor: int = 252
) -> float:
    """
    Calculate Parkinson (high-low) volatility estimator.

    More efficient than close-to-close as it uses intraday range.
    Formula: RV = sqrt(annualization / (4 * N * ln(2)) * sum(ln(H/L)^2))

    Args:
        high: Array of high prices
        low: Array of low prices
        annualization_factor: Trading days per year

    Returns:
        Annualized volatility
    """
    if len(high) == 0 or len(low) == 0:
        return np.nan

    # Avoid division by zero
    valid_mask = (low > 0) & (high > 0)
    high = high[valid_mask]
    low = low[valid_mask]

    if len(high) == 0:
        return np.nan

    log_hl_squared = np.log(high / low) ** 2
    n = len(log_hl_squared)
    variance = np.sum(log_hl_squared) / (4 * n * np.log(2))
    return np.sqrt(variance * annualization_factor)


def calculate_garman_klass_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """
    Calculate Garman-Klass volatility estimator.

    More efficient than Parkinson, uses all OHLC data.
    Formula: RV = sqrt(annualization/N * sum(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))

    Args:
        open_: Array of open prices
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        annualization_factor: Trading days per year

    Returns:
        Annualized volatility
    """
    if len(open_) == 0:
        return np.nan

    # Avoid division by zero
    valid_mask = (open_ > 0) & (low > 0) & (high > 0) & (close > 0)
    open_ = open_[valid_mask]
    high = high[valid_mask]
    low = low[valid_mask]
    close = close[valid_mask]

    if len(open_) == 0:
        return np.nan

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    variance_term = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    variance = np.mean(variance_term)

    # Garman-Klass can give negative variance in some edge cases
    if variance < 0:
        return np.nan

    return np.sqrt(variance * annualization_factor)


class RealizedVolCalculator:
    """Calculator for realized volatility metrics."""

    def __init__(self, annualization_factor: int = 252):
        self.annualization_factor = annualization_factor

    def get_price_history(
        self, symbol: str, end_date: date, lookback_days: int = 300
    ) -> pd.DataFrame:
        """
        Get underlying price history from database.

        Args:
            symbol: Ticker symbol
            end_date: End date for history
            lookback_days: Number of calendar days to look back

        Returns:
            DataFrame with OHLC data
        """
        conn = get_connection()
        query = """
            SELECT date, open, high, low, close
            FROM underlying_prices
            WHERE symbol = ?
              AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = conn.execute(query, [symbol, end_date, lookback_days]).fetchdf()
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def calculate_for_date(self, symbol: str, target_date: date) -> RealizedVolResult:
        """
        Calculate all realized vol metrics for a specific date.

        Args:
            symbol: Ticker symbol
            target_date: Date to calculate for

        Returns:
            RealizedVolResult with all metrics
        """
        # Get enough history for longest window (252 days) plus buffer
        df = self.get_price_history(symbol, target_date, lookback_days=300)

        if df.empty:
            logger.warning(f"No price data found for {symbol} before {target_date}")
            return RealizedVolResult(symbol=symbol, date=target_date)

        # Filter to only include data up to target_date
        df = df[df["date"] <= pd.Timestamp(target_date)]

        if len(df) < 10:
            logger.warning(
                f"Insufficient price data for {symbol} on {target_date}: {len(df)} days"
            )
            return RealizedVolResult(symbol=symbol, date=target_date)

        # Calculate log returns
        close_prices = df["close"].values
        log_returns = np.diff(np.log(close_prices))

        # Get OHLC arrays for range-based estimators
        open_prices = df["open"].values[1:]  # Align with returns
        high_prices = df["high"].values[1:]
        low_prices = df["low"].values[1:]

        result = RealizedVolResult(symbol=symbol, date=target_date)

        # Close-to-close volatility at different windows
        windows = [(10, "rv_10d"), (21, "rv_21d"), (63, "rv_63d"), (252, "rv_252d")]
        for window, attr in windows:
            if len(log_returns) >= window:
                rv = calculate_close_to_close_vol(
                    log_returns[-window:], self.annualization_factor
                )
                setattr(result, attr, rv)

        # Parkinson volatility
        parkinson_windows = [(10, "parkinson_10d"), (21, "parkinson_21d")]
        for window, attr in parkinson_windows:
            if len(high_prices) >= window:
                rv = calculate_parkinson_vol(
                    high_prices[-window:],
                    low_prices[-window:],
                    self.annualization_factor,
                )
                setattr(result, attr, rv)

        # Garman-Klass volatility
        gk_windows = [(10, "gk_10d"), (21, "gk_21d")]
        for window, attr in gk_windows:
            if len(open_prices) >= window:
                rv = calculate_garman_klass_vol(
                    open_prices[-window:],
                    high_prices[-window:],
                    low_prices[-window:],
                    close_prices[1:][-window:],  # Align close with returns
                    self.annualization_factor,
                )
                setattr(result, attr, rv)

        return result

    def store_result(self, result: RealizedVolResult) -> None:
        """Store realized vol result in database."""
        conn = get_connection()

        # Upsert into realized_volatility table
        conn.execute(
            """
            INSERT INTO realized_volatility (
                rv_id, symbol, date,
                rv_10d, rv_21d, rv_63d, rv_252d,
                parkinson_10d, parkinson_21d,
                gk_10d, gk_21d,
                calculation_timestamp
            ) VALUES (
                nextval('seq_rv_id'), ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                now()
            )
            ON CONFLICT (symbol, date) DO UPDATE SET
                rv_10d = excluded.rv_10d,
                rv_21d = excluded.rv_21d,
                rv_63d = excluded.rv_63d,
                rv_252d = excluded.rv_252d,
                parkinson_10d = excluded.parkinson_10d,
                parkinson_21d = excluded.parkinson_21d,
                gk_10d = excluded.gk_10d,
                gk_21d = excluded.gk_21d,
                calculation_timestamp = now()
            """,
            [
                result.symbol,
                result.date,
                result.rv_10d,
                result.rv_21d,
                result.rv_63d,
                result.rv_252d,
                result.parkinson_10d,
                result.parkinson_21d,
                result.gk_10d,
                result.gk_21d,
            ],
        )

    def backfill(
        self, symbol: str, start_date: date, end_date: date, store: bool = True
    ) -> int:
        """
        Calculate and optionally store realized vol for a date range.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range
            store: Whether to store results in database

        Returns:
            Number of dates processed
        """
        conn = get_connection()

        # Get all dates with underlying prices in range
        query = """
            SELECT DISTINCT date
            FROM underlying_prices
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        dates_df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()

        if dates_df.empty:
            logger.warning(f"No underlying prices found for {symbol} in date range")
            return 0

        count = 0
        for _, row in dates_df.iterrows():
            target_date = row["date"]
            if hasattr(target_date, "date"):
                target_date = target_date.date()

            result = self.calculate_for_date(symbol, target_date)

            # Only store if we have at least some data
            if result.rv_10d is not None and store:
                self.store_result(result)
                count += 1

        logger.info(f"Calculated realized vol for {count} dates")
        return count

    def get_stored_vol(self, symbol: str, target_date: date) -> Optional[dict]:
        """Get stored realized vol from database."""
        conn = get_connection()
        query = """
            SELECT *
            FROM realized_volatility
            WHERE symbol = ?
              AND date = ?
        """
        df = conn.execute(query, [symbol, target_date]).fetchdf()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_vol_timeseries(
        self, symbol: str, start_date: date, end_date: date, metric: str = "rv_21d"
    ) -> pd.DataFrame:
        """
        Get realized vol time series from database.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range
            metric: Which vol metric to retrieve

        Returns:
            DataFrame with date and vol columns
        """
        conn = get_connection()
        query = f"""
            SELECT date, {metric} as realized_vol
            FROM realized_volatility
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        return conn.execute(query, [symbol, start_date, end_date]).fetchdf()
