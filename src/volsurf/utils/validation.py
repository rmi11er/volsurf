"""Data validation and monitoring utilities."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection


@dataclass
class DataQualityReport:
    """Report on data quality for a symbol/date."""

    symbol: str
    quote_date: date

    # Raw data stats
    total_options: int = 0
    liquid_options: int = 0
    num_expirations: int = 0

    # Quality issues
    missing_bid: int = 0
    missing_ask: int = 0
    invalid_spread: int = 0  # bid > ask
    zero_bid: int = 0
    missing_iv: int = 0
    extreme_iv: int = 0  # IV < 5% or > 200%
    low_oi: int = 0  # OI < threshold

    # Derived metrics
    liquidity_ratio: float = 0.0
    avg_spread_pct: Optional[float] = None

    def is_healthy(self) -> bool:
        """Check if data quality is acceptable."""
        return (
            self.total_options > 0
            and self.liquidity_ratio > 0.1
            and self.invalid_spread == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quote_date": self.quote_date.isoformat(),
            "total_options": self.total_options,
            "liquid_options": self.liquid_options,
            "num_expirations": self.num_expirations,
            "missing_bid": self.missing_bid,
            "missing_ask": self.missing_ask,
            "invalid_spread": self.invalid_spread,
            "zero_bid": self.zero_bid,
            "missing_iv": self.missing_iv,
            "extreme_iv": self.extreme_iv,
            "low_oi": self.low_oi,
            "liquidity_ratio": self.liquidity_ratio,
            "avg_spread_pct": self.avg_spread_pct,
            "is_healthy": self.is_healthy(),
        }


@dataclass
class FitQualityReport:
    """Report on surface fit quality for a symbol/date."""

    symbol: str
    quote_date: date

    # Fit stats
    num_surfaces: int = 0
    successful_fits: int = 0
    failed_fits: int = 0

    # Quality metrics
    avg_rmse: Optional[float] = None
    max_rmse: Optional[float] = None
    avg_points_per_fit: Optional[float] = None

    # Arbitrage stats
    arb_free_surfaces: int = 0
    butterfly_violations: int = 0
    calendar_violations: int = 0

    def is_healthy(self) -> bool:
        """Check if fit quality is acceptable."""
        return (
            self.num_surfaces > 0
            and self.successful_fits == self.num_surfaces
            and (self.avg_rmse is None or self.avg_rmse < 0.01)  # 1% RMSE threshold
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quote_date": self.quote_date.isoformat(),
            "num_surfaces": self.num_surfaces,
            "successful_fits": self.successful_fits,
            "failed_fits": self.failed_fits,
            "avg_rmse": self.avg_rmse,
            "max_rmse": self.max_rmse,
            "avg_points_per_fit": self.avg_points_per_fit,
            "arb_free_surfaces": self.arb_free_surfaces,
            "butterfly_violations": self.butterfly_violations,
            "calendar_violations": self.calendar_violations,
            "is_healthy": self.is_healthy(),
        }


class DataValidator:
    """Validate data quality and generate reports."""

    def __init__(self, min_oi_threshold: int = 50, iv_bounds: tuple = (0.05, 2.0)):
        """
        Initialize validator.

        Args:
            min_oi_threshold: Minimum open interest for "healthy" option
            iv_bounds: (min, max) bounds for valid IV
        """
        self.min_oi_threshold = min_oi_threshold
        self.iv_min, self.iv_max = iv_bounds

    def validate_date(self, symbol: str, quote_date: date) -> DataQualityReport:
        """
        Validate data quality for a specific date.

        Args:
            symbol: Ticker symbol
            quote_date: Date to validate

        Returns:
            DataQualityReport with quality metrics
        """
        conn = get_connection()

        report = DataQualityReport(symbol=symbol, quote_date=quote_date)

        # Get raw data
        df = conn.execute(
            """
            SELECT
                bid, ask, mid, open_interest, implied_volatility, is_liquid
            FROM raw_options_chains
            WHERE symbol = ? AND quote_date = ?
            """,
            [symbol, quote_date],
        ).fetchdf()

        if df.empty:
            return report

        report.total_options = len(df)
        report.liquid_options = int(df["is_liquid"].sum())
        report.liquidity_ratio = report.liquid_options / report.total_options

        # Count issues
        report.missing_bid = int(df["bid"].isna().sum())
        report.missing_ask = int(df["ask"].isna().sum())
        report.invalid_spread = int((df["bid"] > df["ask"]).sum())
        report.zero_bid = int((df["bid"] == 0).sum())
        report.low_oi = int((df["open_interest"] < self.min_oi_threshold).sum())

        if "implied_volatility" in df.columns:
            iv = df["implied_volatility"]
            report.missing_iv = int(iv.isna().sum())
            valid_iv = iv.dropna()
            report.extreme_iv = int(
                ((valid_iv < self.iv_min) | (valid_iv > self.iv_max)).sum()
            )

        # Calculate average spread
        valid_spreads = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid"] > 0)]
        if len(valid_spreads) > 0:
            spread_pct = (valid_spreads["ask"] - valid_spreads["bid"]) / valid_spreads["mid"]
            report.avg_spread_pct = float(spread_pct.mean())

        # Count expirations
        exp_count = conn.execute(
            """
            SELECT COUNT(DISTINCT expiration_date)
            FROM raw_options_chains
            WHERE symbol = ? AND quote_date = ?
            """,
            [symbol, quote_date],
        ).fetchone()
        report.num_expirations = exp_count[0] if exp_count else 0

        return report

    def validate_fits(self, symbol: str, quote_date: date) -> FitQualityReport:
        """
        Validate surface fit quality for a specific date.

        Args:
            symbol: Ticker symbol
            quote_date: Date to validate

        Returns:
            FitQualityReport with quality metrics
        """
        conn = get_connection()

        report = FitQualityReport(symbol=symbol, quote_date=quote_date)

        df = conn.execute(
            """
            SELECT
                rmse, num_points, passes_no_arbitrage,
                butterfly_arbitrage_violations, calendar_arbitrage_violations
            FROM fitted_surfaces
            WHERE symbol = ? AND quote_date = ?
            """,
            [symbol, quote_date],
        ).fetchdf()

        if df.empty:
            return report

        report.num_surfaces = len(df)
        report.successful_fits = len(df)  # All stored fits are successful
        report.failed_fits = 0

        report.avg_rmse = float(df["rmse"].mean())
        report.max_rmse = float(df["rmse"].max())
        report.avg_points_per_fit = float(df["num_points"].mean())

        if "passes_no_arbitrage" in df.columns:
            report.arb_free_surfaces = int(df["passes_no_arbitrage"].sum())
        if "butterfly_arbitrage_violations" in df.columns:
            report.butterfly_violations = int(df["butterfly_arbitrage_violations"].sum())
        if "calendar_arbitrage_violations" in df.columns:
            report.calendar_violations = int(df["calendar_arbitrage_violations"].sum())

        return report

    def generate_health_report(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Generate health report across a date range.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range

        Returns:
            DataFrame with daily health metrics
        """
        conn = get_connection()

        # Get all dates with data
        dates_df = conn.execute(
            """
            SELECT DISTINCT quote_date
            FROM raw_options_chains
            WHERE symbol = ?
              AND quote_date >= ?
              AND quote_date <= ?
            ORDER BY quote_date
            """,
            [symbol, start_date, end_date],
        ).fetchdf()

        reports = []
        for _, row in dates_df.iterrows():
            d = row["quote_date"]
            if hasattr(d, "date"):
                d = d.date()

            data_report = self.validate_date(symbol, d)
            fit_report = self.validate_fits(symbol, d)

            reports.append({
                "date": d,
                "total_options": data_report.total_options,
                "liquid_options": data_report.liquid_options,
                "liquidity_ratio": data_report.liquidity_ratio,
                "num_expirations": data_report.num_expirations,
                "num_surfaces": fit_report.num_surfaces,
                "avg_rmse": fit_report.avg_rmse,
                "data_healthy": data_report.is_healthy(),
                "fits_healthy": fit_report.is_healthy(),
            })

        return pd.DataFrame(reports)


def check_data_gaps(
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[date]:
    """
    Find missing trading days in the data.

    Args:
        symbol: Ticker symbol
        start_date: Start of range
        end_date: End of range

    Returns:
        List of dates with missing data
    """
    conn = get_connection()

    # Get dates with data
    df = conn.execute(
        """
        SELECT DISTINCT quote_date
        FROM raw_options_chains
        WHERE symbol = ?
          AND quote_date >= ?
          AND quote_date <= ?
        """,
        [symbol, start_date, end_date],
    ).fetchdf()

    existing_dates = set()
    for _, row in df.iterrows():
        d = row["quote_date"]
        if hasattr(d, "date"):
            d = d.date()
        existing_dates.add(d)

    # Generate all weekdays in range
    missing = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5 and current not in existing_dates:
            missing.append(current)
        current += timedelta(days=1)

    return missing


def print_validation_summary(symbol: str, quote_date: date) -> None:
    """Print a formatted validation summary."""
    validator = DataValidator()

    data_report = validator.validate_date(symbol, quote_date)
    fit_report = validator.validate_fits(symbol, quote_date)

    print(f"\n{'='*50}")
    print(f"Validation Report: {symbol} on {quote_date}")
    print(f"{'='*50}")

    # Data quality
    print("\nData Quality:")
    print("-" * 40)
    status = "[OK]" if data_report.is_healthy() else "[WARNING]"
    print(f"  Status: {status}")
    print(f"  Total options: {data_report.total_options:,}")
    print(f"  Liquid options: {data_report.liquid_options:,} ({data_report.liquidity_ratio:.1%})")
    print(f"  Expirations: {data_report.num_expirations}")

    if data_report.invalid_spread > 0:
        print(f"  [!] Invalid spreads: {data_report.invalid_spread}")
    if data_report.missing_iv > 0:
        print(f"  [!] Missing IV: {data_report.missing_iv}")
    if data_report.extreme_iv > 0:
        print(f"  [!] Extreme IV: {data_report.extreme_iv}")

    # Fit quality
    print("\nFit Quality:")
    print("-" * 40)
    status = "[OK]" if fit_report.is_healthy() else "[WARNING]"
    print(f"  Status: {status}")
    print(f"  Surfaces fitted: {fit_report.num_surfaces}")

    if fit_report.avg_rmse:
        print(f"  Avg RMSE: {fit_report.avg_rmse:.4%}")
        print(f"  Max RMSE: {fit_report.max_rmse:.4%}")
    if fit_report.avg_points_per_fit:
        print(f"  Avg points/fit: {fit_report.avg_points_per_fit:.1f}")

    print(f"  Arb-free: {fit_report.arb_free_surfaces}/{fit_report.num_surfaces}")

    print("=" * 50)
