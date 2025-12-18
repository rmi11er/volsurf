"""Term structure analysis for volatility surfaces."""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import curve_fit

from volsurf.database.connection import get_connection


@dataclass
class TermStructureFit:
    """Result of term structure fitting."""

    # Power law parameters: σ(T) = a * T^b
    a: float  # Base level
    b: float  # Power exponent
    rmse: float  # Fit quality
    num_points: int


@dataclass
class TermStructureResult:
    """Combined term structure results."""

    symbol: str
    quote_date: date
    atm_fit: Optional[TermStructureFit] = None
    skew_fit: Optional[TermStructureFit] = None


def power_law(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power law model for term structure.

    σ(T) = a * T^b

    Args:
        t: Time to expiration in years
        a: Base volatility level
        b: Power exponent (typically negative for equity vol)

    Returns:
        Volatility values
    """
    return a * np.power(t, b)


def fit_power_law(
    tte_years: np.ndarray, values: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit power law to term structure data.

    Args:
        tte_years: Time to expiration in years
        values: ATM vol or skew values

    Returns:
        Tuple of (a, b, rmse)
    """
    if len(tte_years) < 2:
        raise ValueError("Need at least 2 points to fit term structure")

    # Initial guess: a = ATM level, b = -0.1 (slight decay)
    p0 = [values[0], -0.1]

    try:
        # Fit with bounds: a > 0, b can be any value
        popt, _ = curve_fit(
            power_law,
            tte_years,
            values,
            p0=p0,
            bounds=([0.001, -2.0], [2.0, 2.0]),
            maxfev=1000,
        )
        a, b = popt

        # Calculate RMSE
        fitted = power_law(tte_years, a, b)
        rmse = np.sqrt(np.mean((values - fitted) ** 2))

        return a, b, rmse

    except Exception as e:
        logger.warning(f"Power law fit failed: {e}")
        # Fallback: constant fit
        a = np.mean(values)
        rmse = np.sqrt(np.mean((values - a) ** 2))
        return a, 0.0, rmse


class TermStructureAnalyzer:
    """Analyzer for volatility term structures."""

    def get_surfaces_for_date(self, symbol: str, quote_date: date) -> pd.DataFrame:
        """
        Get all fitted surfaces for a specific quote date.

        Args:
            symbol: Ticker symbol
            quote_date: Date to get surfaces for

        Returns:
            DataFrame with surface data sorted by TTE
        """
        conn = get_connection()
        query = """
            SELECT expiration_date, tte_years, atm_vol, skew_25delta,
                   svi_a, svi_b, svi_rho, svi_m, svi_sigma,
                   rmse, num_points
            FROM fitted_surfaces
            WHERE symbol = ?
              AND quote_date = ?
              AND atm_vol IS NOT NULL
            ORDER BY tte_years
        """
        return conn.execute(query, [symbol, quote_date]).fetchdf()

    def fit_atm_term_structure(
        self, surfaces_df: pd.DataFrame
    ) -> Optional[TermStructureFit]:
        """
        Fit power law to ATM volatility term structure.

        Args:
            surfaces_df: DataFrame with tte_years and atm_vol columns

        Returns:
            TermStructureFit or None if insufficient data
        """
        if surfaces_df.empty or len(surfaces_df) < 2:
            return None

        tte_years = surfaces_df["tte_years"].values
        atm_vols = surfaces_df["atm_vol"].values

        # Filter out any NaN values
        valid_mask = ~np.isnan(atm_vols) & ~np.isnan(tte_years) & (tte_years > 0)
        tte_years = tte_years[valid_mask]
        atm_vols = atm_vols[valid_mask]

        if len(tte_years) < 2:
            return None

        try:
            a, b, rmse = fit_power_law(tte_years, atm_vols)
            return TermStructureFit(a=a, b=b, rmse=rmse, num_points=len(tte_years))
        except Exception as e:
            logger.warning(f"Failed to fit ATM term structure: {e}")
            return None

    def fit_skew_term_structure(
        self, surfaces_df: pd.DataFrame
    ) -> Optional[TermStructureFit]:
        """
        Fit power law to skew term structure.

        Args:
            surfaces_df: DataFrame with tte_years and skew_25delta columns

        Returns:
            TermStructureFit or None if insufficient data
        """
        if surfaces_df.empty or len(surfaces_df) < 2:
            return None

        if "skew_25delta" not in surfaces_df.columns:
            return None

        tte_years = surfaces_df["tte_years"].values
        skews = surfaces_df["skew_25delta"].values

        # Filter out any NaN values
        valid_mask = ~np.isnan(skews) & ~np.isnan(tte_years) & (tte_years > 0)
        tte_years = tte_years[valid_mask]
        skews = skews[valid_mask]

        if len(tte_years) < 2:
            return None

        try:
            # Skew is typically negative for equities, so use absolute value for fitting
            abs_skews = np.abs(skews)
            a, b, rmse = fit_power_law(tte_years, abs_skews)

            # Restore sign (use mean sign)
            mean_sign = np.sign(np.mean(skews))
            a = a * mean_sign

            return TermStructureFit(a=a, b=b, rmse=rmse, num_points=len(tte_years))
        except Exception as e:
            logger.warning(f"Failed to fit skew term structure: {e}")
            return None

    def analyze_date(self, symbol: str, quote_date: date) -> TermStructureResult:
        """
        Analyze term structure for a specific date.

        Args:
            symbol: Ticker symbol
            quote_date: Date to analyze

        Returns:
            TermStructureResult with ATM and skew fits
        """
        surfaces_df = self.get_surfaces_for_date(symbol, quote_date)

        result = TermStructureResult(symbol=symbol, quote_date=quote_date)

        if surfaces_df.empty:
            logger.warning(f"No surfaces found for {symbol} on {quote_date}")
            return result

        result.atm_fit = self.fit_atm_term_structure(surfaces_df)
        result.skew_fit = self.fit_skew_term_structure(surfaces_df)

        return result

    def store_result(self, result: TermStructureResult) -> None:
        """Store term structure result in database."""
        if result.atm_fit is None:
            return

        conn = get_connection()

        conn.execute(
            """
            INSERT INTO term_structure_params (
                ts_id, symbol, quote_date,
                atm_term_a, atm_term_b, atm_term_rmse,
                skew_term_a, skew_term_b, skew_term_rmse,
                num_expirations,
                fit_timestamp
            ) VALUES (
                nextval('seq_ts_id'), ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?,
                now()
            )
            ON CONFLICT (symbol, quote_date, model_type) DO UPDATE SET
                atm_term_a = excluded.atm_term_a,
                atm_term_b = excluded.atm_term_b,
                atm_term_rmse = excluded.atm_term_rmse,
                skew_term_a = excluded.skew_term_a,
                skew_term_b = excluded.skew_term_b,
                skew_term_rmse = excluded.skew_term_rmse,
                num_expirations = excluded.num_expirations,
                fit_timestamp = now()
            """,
            [
                result.symbol,
                result.quote_date,
                result.atm_fit.a if result.atm_fit else None,
                result.atm_fit.b if result.atm_fit else None,
                result.atm_fit.rmse if result.atm_fit else None,
                result.skew_fit.a if result.skew_fit else None,
                result.skew_fit.b if result.skew_fit else None,
                result.skew_fit.rmse if result.skew_fit else None,
                result.atm_fit.num_points if result.atm_fit else None,
            ],
        )

    def get_interpolated_atm_vol(
        self, symbol: str, quote_date: date, tte_years: float
    ) -> Optional[float]:
        """
        Get interpolated ATM vol using fitted term structure.

        Args:
            symbol: Ticker symbol
            quote_date: Quote date
            tte_years: Target time to expiration

        Returns:
            Interpolated ATM vol or None
        """
        conn = get_connection()
        query = """
            SELECT atm_term_a, atm_term_b
            FROM term_structure_params
            WHERE symbol = ?
              AND quote_date = ?
        """
        df = conn.execute(query, [symbol, quote_date]).fetchdf()

        if df.empty:
            return None

        a = df["atm_term_a"].iloc[0]
        b = df["atm_term_b"].iloc[0]

        if a is None or b is None:
            return None

        return float(power_law(np.array([tte_years]), a, b)[0])

    def get_term_structure_data(
        self, symbol: str, quote_date: date
    ) -> List[dict]:
        """
        Get term structure data for display/plotting.

        Args:
            symbol: Ticker symbol
            quote_date: Quote date

        Returns:
            List of dicts with expiration, tte, atm_vol, skew data
        """
        surfaces_df = self.get_surfaces_for_date(symbol, quote_date)

        if surfaces_df.empty:
            return []

        result = []
        for _, row in surfaces_df.iterrows():
            exp_date = row["expiration_date"]
            if hasattr(exp_date, "date"):
                exp_date = exp_date.date()

            result.append({
                "expiration": exp_date,
                "tte_years": float(row["tte_years"]),
                "tte_days": int(row["tte_years"] * 252),
                "atm_vol": float(row["atm_vol"]) if row["atm_vol"] is not None else None,
                "skew_25delta": float(row["skew_25delta"]) if row["skew_25delta"] is not None else None,
            })

        return result
