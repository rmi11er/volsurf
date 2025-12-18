"""Reusable SQL queries for analytics."""

# =============================================================================
# Fitted Surfaces Queries
# =============================================================================

LATEST_SURFACES_BY_SYMBOL = """
SELECT *
FROM fitted_surfaces
WHERE symbol = ?
  AND quote_date = (
      SELECT MAX(quote_date)
      FROM fitted_surfaces
      WHERE symbol = ?
  )
ORDER BY tte_years
"""

SURFACES_FOR_DATE = """
SELECT *
FROM fitted_surfaces
WHERE symbol = ?
  AND quote_date = ?
ORDER BY tte_years
"""

ATM_VOL_TIMESERIES = """
SELECT quote_date, tte_years, atm_vol
FROM fitted_surfaces
WHERE symbol = ?
  AND quote_date >= ?
  AND quote_date <= ?
  AND atm_vol IS NOT NULL
ORDER BY quote_date, tte_years
"""

SKEW_TIMESERIES = """
SELECT quote_date, tte_years, skew_25delta
FROM fitted_surfaces
WHERE symbol = ?
  AND quote_date >= ?
  AND quote_date <= ?
  AND skew_25delta IS NOT NULL
ORDER BY quote_date, tte_years
"""

SURFACE_SUMMARY = """
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

# =============================================================================
# Realized Volatility Queries
# =============================================================================

REALIZED_VOL_FOR_DATE = """
SELECT *
FROM realized_volatility
WHERE symbol = ?
  AND date = ?
"""

REALIZED_VOL_TIMESERIES = """
SELECT date, rv_21d as realized_vol
FROM realized_volatility
WHERE symbol = ?
  AND date >= ?
  AND date <= ?
ORDER BY date
"""

REALIZED_VOL_LATEST = """
SELECT *
FROM realized_volatility
WHERE symbol = ?
ORDER BY date DESC
LIMIT 1
"""

# =============================================================================
# VRP Queries
# =============================================================================

VRP_FOR_DATE = """
SELECT *
FROM vrp_metrics
WHERE symbol = ?
  AND date = ?
"""

VRP_TIMESERIES = """
SELECT date, vrp_30d, vrp_60d, vrp_90d,
       implied_vol_30d, realized_vol_30d, vrp_zscore
FROM vrp_metrics
WHERE symbol = ?
  AND date >= ?
  AND date <= ?
ORDER BY date
"""

VRP_HISTORICAL_FOR_ZSCORE = """
SELECT vrp_30d
FROM vrp_metrics
WHERE symbol = ?
  AND date < ?
  AND vrp_30d IS NOT NULL
ORDER BY date DESC
LIMIT ?
"""

# =============================================================================
# Term Structure Queries
# =============================================================================

TERM_STRUCTURE_FOR_DATE = """
SELECT *
FROM term_structure_params
WHERE symbol = ?
  AND quote_date = ?
"""

TERM_STRUCTURE_TIMESERIES = """
SELECT quote_date, atm_term_a, atm_term_b, skew_term_a, skew_term_b
FROM term_structure_params
WHERE symbol = ?
  AND quote_date >= ?
  AND quote_date <= ?
ORDER BY quote_date
"""

# =============================================================================
# Underlying Price Queries
# =============================================================================

UNDERLYING_PRICES_FOR_RANGE = """
SELECT date, open, high, low, close, volume
FROM underlying_prices
WHERE symbol = ?
  AND date >= ?
  AND date <= ?
ORDER BY date
"""

UNDERLYING_LATEST = """
SELECT *
FROM underlying_prices
WHERE symbol = ?
ORDER BY date DESC
LIMIT 1
"""

# =============================================================================
# Date Range Queries
# =============================================================================

AVAILABLE_QUOTE_DATES = """
SELECT DISTINCT quote_date
FROM fitted_surfaces
WHERE symbol = ?
ORDER BY quote_date
"""

DATE_RANGE_FOR_SYMBOL = """
SELECT MIN(quote_date) as min_date, MAX(quote_date) as max_date
FROM fitted_surfaces
WHERE symbol = ?
"""

UNDERLYING_DATE_RANGE = """
SELECT MIN(date) as min_date, MAX(date) as max_date
FROM underlying_prices
WHERE symbol = ?
"""
