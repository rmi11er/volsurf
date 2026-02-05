"""Spot-vol dynamics analysis for intraday volatility surfaces."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy import stats


@dataclass
class SpotVolDynamicsResult:
    """Result of spot-vol dynamics analysis."""

    symbol: str
    start_date: date
    end_date: date
    sample_size: int

    # Leverage effect (spot-vol correlation)
    leverage_correlation: Optional[float] = None
    leverage_beta: Optional[float] = None
    leverage_r_squared: Optional[float] = None
    leverage_pvalue: Optional[float] = None

    # Skew dynamics
    skew_spot_correlation: Optional[float] = None
    skew_spot_beta: Optional[float] = None

    # Vol-of-vol metrics
    vol_of_vol: Optional[float] = None  # Annualized std of ATM vol changes
    vol_of_vol_daily: Optional[float] = None  # Average intraday vol-of-vol

    # Distributional properties
    vol_change_mean: Optional[float] = None
    vol_change_std: Optional[float] = None
    vol_change_skewness: Optional[float] = None
    vol_change_kurtosis: Optional[float] = None


def calculate_leverage_effect(
    spot_returns: np.ndarray,
    vol_changes: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Calculate leverage effect (negative spot-vol correlation).

    The leverage effect is the empirical observation that volatility tends
    to increase when prices fall (negative correlation between spot returns
    and vol changes).

    Args:
        spot_returns: Array of log returns of spot price
        vol_changes: Array of changes in ATM implied vol

    Returns:
        Tuple of (correlation, beta, r_squared, p_value)
    """
    if len(spot_returns) < 10 or len(vol_changes) < 10:
        return (np.nan, np.nan, np.nan, np.nan)

    # Remove NaN values
    mask = ~(np.isnan(spot_returns) | np.isnan(vol_changes))
    spot_returns = spot_returns[mask]
    vol_changes = vol_changes[mask]

    if len(spot_returns) < 10:
        return (np.nan, np.nan, np.nan, np.nan)

    # Correlation
    correlation = np.corrcoef(spot_returns, vol_changes)[0, 1]

    # Linear regression: vol_change = alpha + beta * spot_return
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        spot_returns, vol_changes
    )

    return (float(correlation), float(slope), float(r_value**2), float(p_value))


def calculate_skew_dynamics(
    spot_returns: np.ndarray,
    skew_changes: np.ndarray,
) -> Tuple[float, float]:
    """
    Analyze how 25-delta skew moves with spot.

    Args:
        spot_returns: Array of log returns of spot price
        skew_changes: Array of changes in 25-delta skew

    Returns:
        Tuple of (correlation, beta)
    """
    if len(spot_returns) < 10 or len(skew_changes) < 10:
        return (np.nan, np.nan)

    # Remove NaN values
    mask = ~(np.isnan(spot_returns) | np.isnan(skew_changes))
    spot_returns = spot_returns[mask]
    skew_changes = skew_changes[mask]

    if len(spot_returns) < 10:
        return (np.nan, np.nan)

    correlation = np.corrcoef(spot_returns, skew_changes)[0, 1]
    slope, _, _, _, _ = stats.linregress(spot_returns, skew_changes)

    return (float(correlation), float(slope))


def calculate_vol_of_vol(
    atm_vols: np.ndarray,
    annualization_factor: int = 252 * 390,  # Trading minutes per year
) -> float:
    """
    Calculate volatility of ATM vol (vol-of-vol).

    Args:
        atm_vols: Array of ATM implied volatilities
        annualization_factor: Factor to annualize (default: minutes in trading year)

    Returns:
        Annualized vol-of-vol
    """
    if len(atm_vols) < 2:
        return np.nan

    # Calculate vol changes (not log changes, since vol is already a rate)
    vol_changes = np.diff(atm_vols)

    # Remove NaN values
    vol_changes = vol_changes[~np.isnan(vol_changes)]

    if len(vol_changes) < 2:
        return np.nan

    # Standard deviation of changes, annualized
    std_changes = np.std(vol_changes, ddof=1)

    # Annualize
    return float(std_changes * np.sqrt(annualization_factor))


def calculate_intraday_vol_of_vol(
    daily_vols: List[np.ndarray],
) -> Tuple[float, float]:
    """
    Calculate average intraday vol-of-vol.

    Args:
        daily_vols: List of arrays, each containing intraday ATM vols for a day

    Returns:
        Tuple of (mean intraday vol-of-vol, std of intraday vol-of-vol)
    """
    if not daily_vols:
        return (np.nan, np.nan)

    intraday_vovs = []
    for day_vols in daily_vols:
        if len(day_vols) >= 10:
            vol_changes = np.diff(day_vols)
            vol_changes = vol_changes[~np.isnan(vol_changes)]
            if len(vol_changes) > 0:
                # Intraday vol-of-vol (not annualized)
                intraday_vovs.append(np.std(vol_changes, ddof=1))

    if not intraday_vovs:
        return (np.nan, np.nan)

    return (float(np.mean(intraday_vovs)), float(np.std(intraday_vovs)))


class SpotVolDynamicsAnalyzer:
    """Analyzer for spot-vol dynamics from intraday data."""

    def __init__(self, data_dir: Path):
        """
        Initialize analyzer.

        Args:
            data_dir: Directory containing intraday parquet files
        """
        self.data_dir = Path(data_dir)

    def get_available_dates(self, symbol: str = "SPY") -> List[date]:
        """Get list of dates with intraday data available."""
        if not self.data_dir.exists():
            return []

        dates = []
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir():
                # Check for either individual files or daily aggregate
                has_data = (
                    (d / f"{symbol}_daily.parquet").exists()
                    or len(list(d.glob(f"{symbol}_*.parquet"))) > 0
                )
                if has_data:
                    try:
                        dates.append(date.fromisoformat(d.name))
                    except ValueError:
                        continue
        return dates

    def load_intraday_timeseries(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        target_tte_days: int = 30,
    ) -> pd.DataFrame:
        """
        Load time series of intraday surface data.

        Interpolates ATM vol and skew to a target tenor.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period
            target_tte_days: Target days to expiry for interpolation

        Returns:
            DataFrame with columns: timestamp, underlying_price, atm_vol, skew_25delta
        """
        all_records = []

        current = start_date
        while current <= end_date:
            day_dir = self.data_dir / current.strftime("%Y-%m-%d")

            if day_dir.exists():
                # Try daily aggregate first
                daily_file = day_dir / f"{symbol}_daily.parquet"
                if daily_file.exists():
                    df = pl.read_parquet(str(daily_file))
                else:
                    # Load individual files
                    files = sorted(day_dir.glob(f"{symbol}_*.parquet"))
                    if files:
                        dfs = [pl.read_parquet(str(f)) for f in files]
                        df = pl.concat(dfs)
                    else:
                        df = None

                if df is not None and not df.is_empty():
                    # Group by timestamp and interpolate to target tenor
                    for ts in df["timestamp"].unique().sort():
                        ts_df = df.filter(pl.col("timestamp") == ts)

                        if ts_df.is_empty():
                            continue

                        # Get underlying price
                        if "underlying_price" in ts_df.columns:
                            underlying = ts_df["underlying_price"][0]
                        else:
                            underlying = ts_df["forward_price"][0]

                        # Interpolate ATM vol to target tenor
                        ttes = ts_df["tte_years"].to_numpy() * 365  # Convert to days
                        atm_vols = ts_df["atm_vol"].to_numpy()
                        skews = ts_df["skew_25delta"].to_numpy()

                        # Find bracketing tenors for interpolation
                        if len(ttes) < 2:
                            continue

                        sorted_idx = np.argsort(ttes)
                        ttes = ttes[sorted_idx]
                        atm_vols = atm_vols[sorted_idx]
                        skews = skews[sorted_idx]

                        # Linear interpolation
                        if target_tte_days <= ttes[0]:
                            interp_vol = atm_vols[0]
                            interp_skew = skews[0] if skews[0] is not None else np.nan
                        elif target_tte_days >= ttes[-1]:
                            interp_vol = atm_vols[-1]
                            interp_skew = skews[-1] if skews[-1] is not None else np.nan
                        else:
                            interp_vol = float(np.interp(target_tte_days, ttes, atm_vols))
                            # Handle None values in skews
                            valid_skew_mask = ~pd.isna(skews)
                            if valid_skew_mask.sum() >= 2:
                                interp_skew = float(np.interp(
                                    target_tte_days,
                                    ttes[valid_skew_mask],
                                    skews[valid_skew_mask]
                                ))
                            else:
                                interp_skew = np.nan

                        all_records.append({
                            "timestamp": ts,
                            "underlying_price": float(underlying),
                            "atm_vol": interp_vol,
                            "skew_25delta": interp_skew,
                        })

            current = current + pd.Timedelta(days=1)
            if hasattr(current, 'date'):
                current = current.date()

        if not all_records:
            return pd.DataFrame(columns=["timestamp", "underlying_price", "atm_vol", "skew_25delta"])

        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def analyze_period(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        target_tte_days: int = 30,
    ) -> SpotVolDynamicsResult:
        """
        Run full spot-vol dynamics analysis.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period
            target_tte_days: Target tenor for analysis

        Returns:
            SpotVolDynamicsResult with all metrics
        """
        logger.info(f"Analyzing spot-vol dynamics for {symbol} from {start_date} to {end_date}")

        # Load time series
        df = self.load_intraday_timeseries(symbol, start_date, end_date, target_tte_days)

        if df.empty or len(df) < 20:
            logger.warning(f"Insufficient data for analysis: {len(df)} observations")
            return SpotVolDynamicsResult(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                sample_size=len(df),
            )

        # Calculate returns and changes
        spot_prices = df["underlying_price"].values
        atm_vols = df["atm_vol"].values
        skews = df["skew_25delta"].values

        # Log returns for spot
        spot_returns = np.diff(np.log(spot_prices))

        # Changes for vol and skew
        vol_changes = np.diff(atm_vols)
        skew_changes = np.diff(skews)

        # Leverage effect
        leverage_corr, leverage_beta, leverage_r2, leverage_pval = calculate_leverage_effect(
            spot_returns, vol_changes
        )

        # Skew dynamics
        skew_corr, skew_beta = calculate_skew_dynamics(spot_returns, skew_changes)

        # Vol-of-vol
        vol_of_vol = calculate_vol_of_vol(atm_vols)

        # Intraday vol-of-vol (group by date)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_vols = [
            group["atm_vol"].values
            for _, group in df.groupby("date")
        ]
        mean_intraday_vov, _ = calculate_intraday_vol_of_vol(daily_vols)

        # Distributional properties of vol changes
        valid_vol_changes = vol_changes[~np.isnan(vol_changes)]

        result = SpotVolDynamicsResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            sample_size=len(df),
            leverage_correlation=leverage_corr,
            leverage_beta=leverage_beta,
            leverage_r_squared=leverage_r2,
            leverage_pvalue=leverage_pval,
            skew_spot_correlation=skew_corr,
            skew_spot_beta=skew_beta,
            vol_of_vol=vol_of_vol,
            vol_of_vol_daily=mean_intraday_vov,
            vol_change_mean=float(np.mean(valid_vol_changes)) if len(valid_vol_changes) > 0 else None,
            vol_change_std=float(np.std(valid_vol_changes)) if len(valid_vol_changes) > 0 else None,
            vol_change_skewness=float(stats.skew(valid_vol_changes)) if len(valid_vol_changes) > 2 else None,
            vol_change_kurtosis=float(stats.kurtosis(valid_vol_changes)) if len(valid_vol_changes) > 3 else None,
        )

        logger.info(
            f"Analysis complete: leverage_corr={leverage_corr:.3f}, "
            f"vol_of_vol={vol_of_vol:.4f}, n={len(df)}"
        )

        return result

    def get_rolling_leverage(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        window: int = 60,
        target_tte_days: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate rolling leverage correlation time series.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period
            window: Rolling window size (number of observations)
            target_tte_days: Target tenor for analysis

        Returns:
            DataFrame with timestamp and rolling_leverage_corr columns
        """
        df = self.load_intraday_timeseries(symbol, start_date, end_date, target_tte_days)

        if df.empty or len(df) < window + 10:
            return pd.DataFrame(columns=["timestamp", "rolling_leverage_corr"])

        # Calculate returns and changes
        spot_prices = df["underlying_price"].values
        atm_vols = df["atm_vol"].values

        spot_returns = np.diff(np.log(spot_prices))
        vol_changes = np.diff(atm_vols)

        # Calculate rolling correlation
        results = []
        for i in range(window, len(spot_returns)):
            window_returns = spot_returns[i - window:i]
            window_vol_changes = vol_changes[i - window:i]

            # Remove NaN values
            mask = ~(np.isnan(window_returns) | np.isnan(window_vol_changes))
            if mask.sum() >= 10:
                corr = np.corrcoef(
                    window_returns[mask],
                    window_vol_changes[mask]
                )[0, 1]
            else:
                corr = np.nan

            results.append({
                "timestamp": df.iloc[i + 1]["timestamp"],  # +1 because diff reduces length by 1
                "rolling_leverage_corr": corr,
            })

        return pd.DataFrame(results)

    def get_spot_vol_scatter_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        target_tte_days: int = 30,
    ) -> pd.DataFrame:
        """
        Get data for spot-vol scatter plot.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period
            target_tte_days: Target tenor

        Returns:
            DataFrame with spot_return and vol_change columns
        """
        df = self.load_intraday_timeseries(symbol, start_date, end_date, target_tte_days)

        if df.empty or len(df) < 2:
            return pd.DataFrame(columns=["spot_return", "vol_change", "timestamp"])

        spot_prices = df["underlying_price"].values
        atm_vols = df["atm_vol"].values

        spot_returns = np.diff(np.log(spot_prices))
        vol_changes = np.diff(atm_vols)

        result_df = pd.DataFrame({
            "spot_return": spot_returns * 100,  # Convert to percentage
            "vol_change": vol_changes * 100,    # Convert to vol points
            "timestamp": df.iloc[1:]["timestamp"].values,
        })

        # Remove NaN values
        result_df = result_df.dropna()

        return result_df
