"""Liquidity and data quality filters for options data."""

import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings


def apply_liquidity_filters(
    df: pl.DataFrame,
    underlying_price: float,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """
    Apply liquidity and data quality filters to options chain data.

    Filters applied:
    1. Bid-ask spread as percentage of mid
    2. Minimum open interest
    3. Days to expiration range
    4. Moneyness range
    5. Valid prices (bid > 0, ask > bid)

    Args:
        df: Options chain DataFrame with columns:
            [quote_date, expiration_date, strike, bid, ask, mid, open_interest, ...]
        underlying_price: Current underlying price for moneyness calculation
        settings: Optional settings object (uses global if not provided)

    Returns:
        DataFrame with 'is_liquid' column added
    """
    if settings is None:
        settings = get_settings()

    if df.is_empty():
        return df.with_columns(pl.lit(False).alias("is_liquid"))

    # Calculate derived columns for filtering
    df = df.with_columns([
        # Days to expiration
        (pl.col("expiration_date") - pl.col("quote_date")).dt.total_days().alias("dte"),
        # Moneyness (S/K)
        (pl.lit(underlying_price) / pl.col("strike")).alias("moneyness"),
        # Bid-ask spread as percentage of mid
        pl.when(pl.col("mid") > 0)
        .then((pl.col("ask") - pl.col("bid")) / pl.col("mid"))
        .otherwise(pl.lit(1.0))
        .alias("spread_pct"),
    ])

    # Apply filters
    is_liquid = (
        # Valid prices
        (pl.col("bid") > 0)
        & (pl.col("ask") > pl.col("bid"))
        & (pl.col("mid") > 0)
        # Bid-ask spread
        & (pl.col("spread_pct") <= settings.max_bid_ask_spread_pct)
        # Open interest
        & (pl.col("open_interest") >= settings.min_open_interest)
        # Days to expiration
        & (pl.col("dte") >= settings.min_dte)
        & (pl.col("dte") <= settings.max_dte)
        # Moneyness
        & (pl.col("moneyness") >= settings.min_moneyness)
        & (pl.col("moneyness") <= settings.max_moneyness)
    )

    df = df.with_columns(is_liquid.alias("is_liquid"))

    # Log filter statistics
    total = len(df)
    liquid = df.filter(pl.col("is_liquid")).height
    logger.debug(f"Liquidity filter: {liquid}/{total} options marked as liquid ({100*liquid/total:.1f}%)")

    return df


def filter_liquid_only(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter to only liquid options.

    Args:
        df: DataFrame with 'is_liquid' column

    Returns:
        Filtered DataFrame containing only liquid options
    """
    if "is_liquid" not in df.columns:
        raise ValueError("DataFrame must have 'is_liquid' column. Run apply_liquidity_filters first.")

    return df.filter(pl.col("is_liquid"))


def validate_data_quality(df: pl.DataFrame) -> dict[str, int]:
    """
    Validate data quality and return statistics.

    Checks for:
    - Missing implied volatility
    - Negative prices
    - Invalid spreads (bid > ask)
    - Missing Greeks
    - Extreme IVs (< 5% or > 200%)

    Args:
        df: Options chain DataFrame

    Returns:
        Dictionary with validation statistics
    """
    stats = {}

    if df.is_empty():
        return {"total_rows": 0}

    stats["total_rows"] = len(df)

    # Check for missing IV
    if "implied_volatility" in df.columns:
        stats["missing_iv"] = df.filter(pl.col("implied_volatility").is_null()).height

        # Check for extreme IVs
        stats["iv_too_low"] = df.filter(
            pl.col("implied_volatility").is_not_null() & (pl.col("implied_volatility") < 0.05)
        ).height
        stats["iv_too_high"] = df.filter(
            pl.col("implied_volatility").is_not_null() & (pl.col("implied_volatility") > 2.0)
        ).height

    # Check for negative prices
    if "bid" in df.columns:
        stats["negative_bid"] = df.filter(pl.col("bid") < 0).height

    if "ask" in df.columns:
        stats["negative_ask"] = df.filter(pl.col("ask") < 0).height

    # Check for invalid spreads
    if "bid" in df.columns and "ask" in df.columns:
        stats["invalid_spread"] = df.filter(pl.col("bid") > pl.col("ask")).height

    # Check for missing Greeks
    for greek in ["delta", "gamma", "theta", "vega"]:
        if greek in df.columns:
            stats[f"missing_{greek}"] = df.filter(pl.col(greek).is_null()).height

    return stats


def get_liquid_strikes_by_expiration(
    df: pl.DataFrame,
    min_strikes: int = 5,
) -> pl.DataFrame:
    """
    Get summary of liquid strikes per expiration.

    Useful for determining which expirations have enough data for fitting.

    Args:
        df: Options chain DataFrame with 'is_liquid' column
        min_strikes: Minimum strikes required for an expiration to be usable

    Returns:
        DataFrame with columns:
        [expiration_date, num_liquid_calls, num_liquid_puts, total_liquid, is_fittable]
    """
    if "is_liquid" not in df.columns:
        raise ValueError("DataFrame must have 'is_liquid' column")

    summary = (
        df.filter(pl.col("is_liquid"))
        .group_by("expiration_date")
        .agg([
            pl.col("option_type")
            .filter(pl.col("option_type") == "CALL")
            .count()
            .alias("num_liquid_calls"),
            pl.col("option_type")
            .filter(pl.col("option_type") == "PUT")
            .count()
            .alias("num_liquid_puts"),
            pl.col("strike").count().alias("total_liquid"),
            pl.col("strike").n_unique().alias("unique_strikes"),
        ])
        .with_columns(
            (pl.col("unique_strikes") >= min_strikes).alias("is_fittable")
        )
        .sort("expiration_date")
    )

    return summary
