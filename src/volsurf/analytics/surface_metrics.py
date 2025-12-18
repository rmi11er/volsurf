"""Surface metrics and derived analytics from fitted surfaces."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection


@dataclass
class SurfaceSummary:
    """Summary metrics for fitted surfaces on a date."""

    symbol: str
    quote_date: date
    num_surfaces: int
    num_expirations: int

    # ATM vol range
    atm_vol_min: Optional[float] = None
    atm_vol_max: Optional[float] = None
    atm_vol_mean: Optional[float] = None

    # Skew metrics
    skew_min: Optional[float] = None
    skew_max: Optional[float] = None
    skew_mean: Optional[float] = None

    # Fit quality
    avg_rmse: Optional[float] = None
    max_rmse: Optional[float] = None
    total_points: int = 0

    # Arbitrage stats
    surfaces_passing_arb: int = 0
    total_butterfly_violations: int = 0
    total_calendar_violations: int = 0


def get_atm_vol_from_svi(
    a: float, b: float, rho: float, m: float, sigma: float, tte_years: float
) -> float:
    """
    Calculate ATM implied vol from SVI parameters.

    At ATM (k=0), total variance w(0) = a + b * (rho * (-m) + sqrt(m^2 + sigma^2))
    IV_ATM = sqrt(w(0) / T)

    Args:
        a, b, rho, m, sigma: SVI parameters
        tte_years: Time to expiration in years

    Returns:
        ATM implied volatility
    """
    # k = 0 at ATM
    total_variance = a + b * (rho * (-m) + np.sqrt(m**2 + sigma**2))
    if total_variance <= 0 or tte_years <= 0:
        return np.nan
    return np.sqrt(total_variance / tte_years)


def get_vol_at_moneyness(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    tte_years: float,
    log_moneyness: float,
) -> float:
    """
    Calculate implied vol at specific log-moneyness from SVI parameters.

    Args:
        a, b, rho, m, sigma: SVI parameters
        tte_years: Time to expiration in years
        log_moneyness: log(K/F)

    Returns:
        Implied volatility at that moneyness
    """
    k = log_moneyness
    total_variance = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    if total_variance <= 0 or tte_years <= 0:
        return np.nan
    return np.sqrt(total_variance / tte_years)


class SurfaceMetrics:
    """Calculator for surface-derived metrics."""

    def get_surface_summary(self, symbol: str, quote_date: date) -> SurfaceSummary:
        """
        Get summary metrics for all surfaces on a date.

        Args:
            symbol: Ticker symbol
            quote_date: Date to summarize

        Returns:
            SurfaceSummary with aggregated metrics
        """
        conn = get_connection()

        query = """
            SELECT
                COUNT(*) as num_surfaces,
                COUNT(DISTINCT expiration_date) as num_expirations,
                MIN(atm_vol) as atm_vol_min,
                MAX(atm_vol) as atm_vol_max,
                AVG(atm_vol) as atm_vol_mean,
                MIN(skew_25delta) as skew_min,
                MAX(skew_25delta) as skew_max,
                AVG(skew_25delta) as skew_mean,
                AVG(rmse) as avg_rmse,
                MAX(rmse) as max_rmse,
                SUM(num_points) as total_points,
                SUM(CASE WHEN passes_no_arbitrage THEN 1 ELSE 0 END) as surfaces_passing_arb,
                SUM(COALESCE(butterfly_arbitrage_violations, 0)) as total_butterfly_violations,
                SUM(COALESCE(calendar_arbitrage_violations, 0)) as total_calendar_violations
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date = ?
        """
        df = conn.execute(query, [symbol, quote_date]).fetchdf()

        if df.empty or df["num_surfaces"].iloc[0] == 0:
            return SurfaceSummary(
                symbol=symbol, quote_date=quote_date, num_surfaces=0, num_expirations=0
            )

        row = df.iloc[0]
        return SurfaceSummary(
            symbol=symbol,
            quote_date=quote_date,
            num_surfaces=int(row["num_surfaces"]),
            num_expirations=int(row["num_expirations"]),
            atm_vol_min=float(row["atm_vol_min"]) if row["atm_vol_min"] else None,
            atm_vol_max=float(row["atm_vol_max"]) if row["atm_vol_max"] else None,
            atm_vol_mean=float(row["atm_vol_mean"]) if row["atm_vol_mean"] else None,
            skew_min=float(row["skew_min"]) if row["skew_min"] else None,
            skew_max=float(row["skew_max"]) if row["skew_max"] else None,
            skew_mean=float(row["skew_mean"]) if row["skew_mean"] else None,
            avg_rmse=float(row["avg_rmse"]) if row["avg_rmse"] else None,
            max_rmse=float(row["max_rmse"]) if row["max_rmse"] else None,
            total_points=int(row["total_points"]) if row["total_points"] else 0,
            surfaces_passing_arb=int(row["surfaces_passing_arb"]) if row["surfaces_passing_arb"] else 0,
            total_butterfly_violations=int(row["total_butterfly_violations"]) if row["total_butterfly_violations"] else 0,
            total_calendar_violations=int(row["total_calendar_violations"]) if row["total_calendar_violations"] else 0,
        )

    def get_atm_vol_timeseries(
        self, symbol: str, start_date: date, end_date: date, tte_target_days: int = 30
    ) -> pd.DataFrame:
        """
        Get ATM volatility time series for a specific tenor.

        Uses interpolation if exact tenor not available.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range
            tte_target_days: Target tenor in trading days

        Returns:
            DataFrame with date and atm_vol columns
        """
        conn = get_connection()

        tte_target_years = tte_target_days / 252.0

        # Get all surfaces in range, then interpolate per date
        query = """
            SELECT quote_date, tte_years, atm_vol
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date >= ?
              AND quote_date <= ?
              AND atm_vol IS NOT NULL
            ORDER BY quote_date, tte_years
        """
        df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["date", "atm_vol"])

        # Group by quote_date and interpolate
        results = []
        for quote_date, group in df.groupby("quote_date"):
            tte_values = group["tte_years"].values
            atm_vols = group["atm_vol"].values

            # Interpolate to target tenor
            if tte_target_years <= tte_values[0]:
                interp_vol = atm_vols[0]
            elif tte_target_years >= tte_values[-1]:
                interp_vol = atm_vols[-1]
            else:
                interp_vol = np.interp(tte_target_years, tte_values, atm_vols)

            results.append({"date": quote_date, "atm_vol": float(interp_vol)})

        return pd.DataFrame(results)

    def get_skew_timeseries(
        self, symbol: str, start_date: date, end_date: date, tte_target_days: int = 30
    ) -> pd.DataFrame:
        """
        Get 25-delta skew time series for a specific tenor.

        Args:
            symbol: Ticker symbol
            start_date: Start of range
            end_date: End of range
            tte_target_days: Target tenor in trading days

        Returns:
            DataFrame with date and skew columns
        """
        conn = get_connection()

        tte_target_years = tte_target_days / 252.0

        query = """
            SELECT quote_date, tte_years, skew_25delta
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date >= ?
              AND quote_date <= ?
              AND skew_25delta IS NOT NULL
            ORDER BY quote_date, tte_years
        """
        df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["date", "skew"])

        results = []
        for quote_date, group in df.groupby("quote_date"):
            tte_values = group["tte_years"].values
            skews = group["skew_25delta"].values

            if tte_target_years <= tte_values[0]:
                interp_skew = skews[0]
            elif tte_target_years >= tte_values[-1]:
                interp_skew = skews[-1]
            else:
                interp_skew = np.interp(tte_target_years, tte_values, skews)

            results.append({"date": quote_date, "skew": float(interp_skew)})

        return pd.DataFrame(results)

    def get_smile_data(
        self, symbol: str, quote_date: date, expiration_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Get volatility smile data for a specific expiration.

        Args:
            symbol: Ticker symbol
            quote_date: Quote date
            expiration_date: Expiration date

        Returns:
            Dict with SVI params and computed smile, or None
        """
        conn = get_connection()

        query = """
            SELECT svi_a, svi_b, svi_rho, svi_m, svi_sigma,
                   tte_years, forward_price, atm_vol, skew_25delta
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date = ?
              AND expiration_date = ?
        """
        df = conn.execute(query, [symbol, quote_date, expiration_date]).fetchdf()

        if df.empty:
            return None

        row = df.iloc[0]

        # Generate smile across moneyness range
        k_range = np.linspace(-0.3, 0.3, 61)  # -30% to +30% log-moneyness
        vols = []
        for k in k_range:
            vol = get_vol_at_moneyness(
                row["svi_a"],
                row["svi_b"],
                row["svi_rho"],
                row["svi_m"],
                row["svi_sigma"],
                row["tte_years"],
                k,
            )
            vols.append(vol)

        return {
            "symbol": symbol,
            "quote_date": quote_date,
            "expiration_date": expiration_date,
            "tte_years": float(row["tte_years"]),
            "forward_price": float(row["forward_price"]) if row["forward_price"] else None,
            "atm_vol": float(row["atm_vol"]) if row["atm_vol"] else None,
            "skew_25delta": float(row["skew_25delta"]) if row["skew_25delta"] else None,
            "svi_params": {
                "a": float(row["svi_a"]),
                "b": float(row["svi_b"]),
                "rho": float(row["svi_rho"]),
                "m": float(row["svi_m"]),
                "sigma": float(row["svi_sigma"]),
            },
            "smile": {
                "log_moneyness": k_range.tolist(),
                "implied_vol": vols,
            },
        }

    def get_available_dates(self, symbol: str) -> List[date]:
        """Get all dates with fitted surfaces for a symbol."""
        conn = get_connection()
        query = """
            SELECT DISTINCT quote_date
            FROM fitted_surfaces
            WHERE symbol = ?
            ORDER BY quote_date
        """
        df = conn.execute(query, [symbol]).fetchdf()

        dates = []
        for _, row in df.iterrows():
            d = row["quote_date"]
            if hasattr(d, "date"):
                d = d.date()
            dates.append(d)
        return dates

    def get_latest_date(self, symbol: str) -> Optional[date]:
        """Get the most recent date with fitted surfaces."""
        conn = get_connection()
        query = """
            SELECT MAX(quote_date) as latest
            FROM fitted_surfaces
            WHERE symbol = ?
        """
        df = conn.execute(query, [symbol]).fetchdf()

        if df.empty or df["latest"].iloc[0] is None:
            return None

        latest = df["latest"].iloc[0]
        if hasattr(latest, "date"):
            latest = latest.date()
        return latest
