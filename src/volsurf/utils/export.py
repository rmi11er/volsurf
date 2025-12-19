"""Data export utilities for volsurf."""

import json
from datetime import date
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection


def export_options_chain(
    symbol: str,
    quote_date: date,
    output_path: Union[str, Path],
    format: str = "csv",
    liquid_only: bool = True,
) -> Path:
    """
    Export raw options chain data.

    Args:
        symbol: Ticker symbol
        quote_date: Date to export
        output_path: Output file path (without extension)
        format: "csv" or "json"
        liquid_only: Only export liquid options

    Returns:
        Path to exported file
    """
    conn = get_connection()

    query = """
        SELECT
            symbol, quote_date, expiration_date, strike, option_type,
            bid, ask, mid, volume, open_interest,
            implied_volatility, delta, gamma, theta, vega,
            underlying_price, is_liquid
        FROM raw_options_chains
        WHERE symbol = ? AND quote_date = ?
    """

    if liquid_only:
        query += " AND is_liquid = TRUE"

    query += " ORDER BY expiration_date, strike, option_type"

    df = conn.execute(query, [symbol, quote_date]).fetchdf()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format == "json":
        file_path = output_path.with_suffix(".json")
        # Convert dates to strings for JSON
        df["quote_date"] = df["quote_date"].astype(str)
        df["expiration_date"] = df["expiration_date"].astype(str)
        df.to_json(file_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported {len(df)} options to {file_path}")
    return file_path


def export_fitted_surfaces(
    symbol: str,
    quote_date: date,
    output_path: Union[str, Path],
    format: str = "csv",
) -> Path:
    """
    Export fitted surface parameters.

    Args:
        symbol: Ticker symbol
        quote_date: Date to export
        output_path: Output file path (without extension)
        format: "csv" or "json"

    Returns:
        Path to exported file
    """
    conn = get_connection()

    query = """
        SELECT
            symbol, quote_date, expiration_date, tte_years,
            svi_a, svi_b, svi_rho, svi_m, svi_sigma,
            atm_vol, skew_25delta,
            rmse, mae, max_error, num_points,
            passes_no_arbitrage, butterfly_arbitrage_violations, calendar_arbitrage_violations
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ?
        ORDER BY expiration_date
    """

    df = conn.execute(query, [symbol, quote_date]).fetchdf()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format == "json":
        file_path = output_path.with_suffix(".json")
        df["quote_date"] = df["quote_date"].astype(str)
        df["expiration_date"] = df["expiration_date"].astype(str)
        df.to_json(file_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported {len(df)} surfaces to {file_path}")
    return file_path


def export_analytics(
    symbol: str,
    start_date: date,
    end_date: date,
    output_path: Union[str, Path],
    format: str = "csv",
) -> Path:
    """
    Export analytics data (realized vol, VRP) for a date range.

    Args:
        symbol: Ticker symbol
        start_date: Start of range
        end_date: End of range
        output_path: Output file path (without extension)
        format: "csv" or "json"

    Returns:
        Path to exported file
    """
    conn = get_connection()

    # Get realized vol data
    rv_query = """
        SELECT date, rv_10d, rv_21d, rv_63d, parkinson_21d, gk_21d
        FROM realized_volatility
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date
    """
    rv_df = conn.execute(rv_query, [symbol, start_date, end_date]).fetchdf()
    rv_df = rv_df.add_prefix("rv_").rename(columns={"rv_date": "date"})

    # Get VRP data
    vrp_query = """
        SELECT date, vrp_30d, vrp_60d, vrp_90d, implied_vol_30d, realized_vol_30d, vrp_zscore
        FROM vrp_metrics
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date
    """
    vrp_df = conn.execute(vrp_query, [symbol, start_date, end_date]).fetchdf()

    # Get ATM vol time series
    from volsurf.analytics import SurfaceMetrics
    metrics = SurfaceMetrics()
    atm_df = metrics.get_atm_vol_timeseries(symbol, start_date, end_date, tte_target_days=30)

    # Merge all data
    if not rv_df.empty:
        rv_df["date"] = pd.to_datetime(rv_df["date"]).dt.date

    if not vrp_df.empty:
        vrp_df["date"] = pd.to_datetime(vrp_df["date"]).dt.date

    if not atm_df.empty:
        atm_df["date"] = pd.to_datetime(atm_df["date"]).dt.date

    # Start with ATM vol as base
    if not atm_df.empty:
        merged = atm_df.copy()
        merged = merged.rename(columns={"atm_vol": "implied_vol_30d_interp"})
    else:
        # Create empty base
        merged = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="B").date})

    # Merge other data
    if not rv_df.empty:
        merged = merged.merge(rv_df, on="date", how="outer")
    if not vrp_df.empty:
        merged = merged.merge(vrp_df, on="date", how="outer")

    merged = merged.sort_values("date")
    merged["symbol"] = symbol

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        file_path = output_path.with_suffix(".csv")
        merged.to_csv(file_path, index=False)
    elif format == "json":
        file_path = output_path.with_suffix(".json")
        merged["date"] = merged["date"].astype(str)
        merged.to_json(file_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported analytics to {file_path}")
    return file_path


def export_smile_data(
    symbol: str,
    quote_date: date,
    expiration_date: date,
    output_path: Union[str, Path],
    format: str = "csv",
    num_points: int = 100,
) -> Path:
    """
    Export volatility smile data for a specific expiration.

    Args:
        symbol: Ticker symbol
        quote_date: Quote date
        expiration_date: Expiration to export
        output_path: Output file path (without extension)
        format: "csv" or "json"
        num_points: Number of points in the smile curve

    Returns:
        Path to exported file
    """
    import numpy as np
    from volsurf.analytics import get_vol_at_moneyness

    conn = get_connection()

    # Get surface parameters
    query = """
        SELECT svi_a, svi_b, svi_rho, svi_m, svi_sigma, tte_years, forward_price
        FROM fitted_surfaces
        WHERE symbol = ? AND quote_date = ? AND expiration_date = ?
    """
    df = conn.execute(query, [symbol, quote_date, expiration_date]).fetchdf()

    if df.empty:
        raise ValueError(f"No surface found for {symbol} {quote_date} {expiration_date}")

    row = df.iloc[0]

    # Generate smile
    k_range = np.linspace(-0.3, 0.3, num_points)
    vols = [
        get_vol_at_moneyness(
            row["svi_a"], row["svi_b"], row["svi_rho"],
            row["svi_m"], row["svi_sigma"], row["tte_years"], k
        )
        for k in k_range
    ]

    forward = float(row["forward_price"]) if row["forward_price"] else 590.0
    strikes = forward * np.exp(k_range)

    smile_df = pd.DataFrame({
        "symbol": symbol,
        "quote_date": str(quote_date),
        "expiration_date": str(expiration_date),
        "strike": strikes,
        "log_moneyness": k_range,
        "implied_vol": vols,
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        file_path = output_path.with_suffix(".csv")
        smile_df.to_csv(file_path, index=False)
    elif format == "json":
        file_path = output_path.with_suffix(".json")
        smile_df.to_json(file_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported smile data to {file_path}")
    return file_path


def export_full_snapshot(
    symbol: str,
    quote_date: date,
    output_dir: Union[str, Path],
    format: str = "csv",
) -> dict:
    """
    Export complete snapshot of all data for a date.

    Args:
        symbol: Ticker symbol
        quote_date: Date to export
        output_dir: Output directory
        format: "csv" or "json"

    Returns:
        Dict with paths to all exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = quote_date.strftime("%Y%m%d")

    paths = {}

    # Export options chain
    paths["options"] = export_options_chain(
        symbol, quote_date,
        output_dir / f"{symbol}_{date_str}_options",
        format=format,
    )

    # Export surfaces
    paths["surfaces"] = export_fitted_surfaces(
        symbol, quote_date,
        output_dir / f"{symbol}_{date_str}_surfaces",
        format=format,
    )

    logger.info(f"Exported full snapshot to {output_dir}")
    return paths
