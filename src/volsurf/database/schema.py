"""DuckDB schema definitions and initialization."""

from typing import Optional

from duckdb import DuckDBPyConnection
from loguru import logger

from volsurf.database.connection import get_connection


RAW_OPTIONS_CHAINS_DDL = """
CREATE TABLE IF NOT EXISTS raw_options_chains (
    chain_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quote_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    strike DECIMAL(10, 2) NOT NULL,
    option_type VARCHAR(4) NOT NULL,  -- 'CALL' or 'PUT'

    -- Market data
    bid DECIMAL(10, 4),
    ask DECIMAL(10, 4),
    mid DECIMAL(10, 4),  -- computed as (bid + ask) / 2
    last DECIMAL(10, 4),
    volume INTEGER,
    open_interest INTEGER,

    -- Greeks (if provided by data source)
    delta DECIMAL(8, 6),
    gamma DECIMAL(8, 6),
    theta DECIMAL(8, 6),
    vega DECIMAL(8, 6),
    rho DECIMAL(8, 6),

    -- Implied volatility (if provided)
    implied_volatility DECIMAL(8, 6),

    -- Underlying data
    underlying_price DECIMAL(10, 4),

    -- Data quality flags
    is_liquid BOOLEAN,

    -- Metadata
    data_source VARCHAR DEFAULT 'theta_data',
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

RAW_OPTIONS_CHAINS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_options_symbol_date ON raw_options_chains(symbol, quote_date)",
    "CREATE INDEX IF NOT EXISTS idx_options_expiration ON raw_options_chains(expiration_date)",
    "CREATE INDEX IF NOT EXISTS idx_options_liquid ON raw_options_chains(is_liquid)",
]

UNDERLYING_PRICES_DDL = """
CREATE TABLE IF NOT EXISTS underlying_prices (
    price_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4),
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    volume BIGINT,

    UNIQUE(symbol, date)
);
"""

UNDERLYING_PRICES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_underlying_symbol_date ON underlying_prices(symbol, date)",
]

FITTED_SURFACES_DDL = """
CREATE TABLE IF NOT EXISTS fitted_surfaces (
    surface_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quote_date DATE NOT NULL,
    expiration_date DATE NOT NULL,

    -- Time to expiration (in trading days / 252)
    tte_years DECIMAL(10, 8),

    -- Forward price used for fitting
    forward_price DECIMAL(10, 4),

    -- SVI Parameters (Raw parameterization)
    -- Total variance: w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))
    svi_a DECIMAL(10, 8),      -- Vertical shift (ATM variance level)
    svi_b DECIMAL(10, 8),      -- Slope of the wings
    svi_rho DECIMAL(10, 8),    -- Skew parameter (-1 to 1)
    svi_m DECIMAL(10, 8),      -- Horizontal shift (ATM moneyness)
    svi_sigma DECIMAL(10, 8),  -- Smoothness of the smile

    -- Derived quantities for convenience
    atm_vol DECIMAL(8, 6),        -- ATM implied volatility
    skew_25delta DECIMAL(8, 6),  -- 25-delta skew

    -- Fit quality metrics
    rmse DECIMAL(10, 8),       -- Root mean squared error
    mae DECIMAL(10, 8),        -- Mean absolute error
    max_error DECIMAL(10, 8),  -- Maximum absolute error
    num_points INTEGER,        -- Number of data points used in fit

    -- Arbitrage checks
    passes_no_arbitrage BOOLEAN,
    butterfly_arbitrage_violations INTEGER,
    calendar_arbitrage_violations INTEGER,

    -- Model metadata
    model_type VARCHAR DEFAULT 'SVI',
    model_version VARCHAR,
    fit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, quote_date, expiration_date, model_type)
);
"""

FITTED_SURFACES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_surfaces_symbol_date ON fitted_surfaces(symbol, quote_date)",
    "CREATE INDEX IF NOT EXISTS idx_surfaces_expiration ON fitted_surfaces(expiration_date)",
]

TERM_STRUCTURE_PARAMS_DDL = """
CREATE TABLE IF NOT EXISTS term_structure_params (
    ts_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quote_date DATE NOT NULL,

    -- ATM term structure fit (power law: σ(T) = a * T^b)
    atm_term_a DECIMAL(10, 8),
    atm_term_b DECIMAL(10, 8),
    atm_term_rmse DECIMAL(10, 8),

    -- Skew term structure
    skew_term_a DECIMAL(10, 8),
    skew_term_b DECIMAL(10, 8),
    skew_term_rmse DECIMAL(10, 8),

    -- Number of expirations used
    num_expirations INTEGER,

    model_type VARCHAR DEFAULT 'power_law',
    fit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, quote_date, model_type)
);
"""

TERM_STRUCTURE_PARAMS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_term_structure_symbol_date ON term_structure_params(symbol, quote_date)",
]

REALIZED_VOLATILITY_DDL = """
CREATE TABLE IF NOT EXISTS realized_volatility (
    rv_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,

    -- Realized vol at different windows (annualized, using 252 trading days)
    rv_10d DECIMAL(8, 6),   -- 10-day realized vol
    rv_21d DECIMAL(8, 6),   -- 21-day (1 month)
    rv_63d DECIMAL(8, 6),   -- 63-day (3 month)
    rv_252d DECIMAL(8, 6),  -- 252-day (1 year)

    -- Parkinson estimator (high-low)
    parkinson_10d DECIMAL(8, 6),
    parkinson_21d DECIMAL(8, 6),

    -- Garman-Klass estimator
    gk_10d DECIMAL(8, 6),
    gk_21d DECIMAL(8, 6),

    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, date)
);
"""

REALIZED_VOLATILITY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_realized_vol_symbol_date ON realized_volatility(symbol, date)",
]

VRP_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS vrp_metrics (
    vrp_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,

    -- VRP at different horizons (ATM implied vol - realized vol)
    vrp_30d DECIMAL(8, 6),
    vrp_60d DECIMAL(8, 6),
    vrp_90d DECIMAL(8, 6),

    -- Corresponding implied and realized vols
    implied_vol_30d DECIMAL(8, 6),
    realized_vol_30d DECIMAL(8, 6),

    -- Statistical measures
    vrp_zscore DECIMAL(8, 6),  -- Z-score relative to historical VRP

    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, date)
);
"""

VRP_METRICS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_vrp_symbol_date ON vrp_metrics(symbol, date)",
]

# Interest rate term structure table for forward price calculation fallback
INTEREST_RATE_TERM_STRUCTURE_DDL = """
CREATE TABLE IF NOT EXISTS interest_rate_term_structure (
    rate_id BIGINT PRIMARY KEY,
    date DATE NOT NULL,
    tte_days INTEGER NOT NULL,  -- Time to expiration in trading days
    rate DECIMAL(10, 8) NOT NULL,  -- Annualized continuously compounded rate

    -- Source info
    source_symbol VARCHAR,  -- e.g., 'SPY' or 'QQQ' if backed out from options
    calculation_method VARCHAR,  -- 'synthetic_forward', 'treasury', etc.
    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date, tte_days)
);
"""

INTEREST_RATE_TERM_STRUCTURE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_rate_date ON interest_rate_term_structure(date)",
]

# Sequences for auto-incrementing IDs
SEQUENCES = [
    "CREATE SEQUENCE IF NOT EXISTS seq_chain_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_price_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_surface_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_ts_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_rv_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_vrp_id START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_rate_id START 1",
]


def init_schema(conn: Optional[DuckDBPyConnection] = None) -> None:
    """
    Initialize the database schema.

    Creates all tables, indexes, and sequences if they don't exist.

    Args:
        conn: Optional connection to use. If None, uses global connection.
    """
    if conn is None:
        conn = get_connection()

    logger.info("Initializing database schema...")

    # Create sequences
    for seq_ddl in SEQUENCES:
        conn.execute(seq_ddl)

    # Create tables
    tables = [
        ("raw_options_chains", RAW_OPTIONS_CHAINS_DDL, RAW_OPTIONS_CHAINS_INDEXES),
        ("underlying_prices", UNDERLYING_PRICES_DDL, UNDERLYING_PRICES_INDEXES),
        ("fitted_surfaces", FITTED_SURFACES_DDL, FITTED_SURFACES_INDEXES),
        ("term_structure_params", TERM_STRUCTURE_PARAMS_DDL, TERM_STRUCTURE_PARAMS_INDEXES),
        ("realized_volatility", REALIZED_VOLATILITY_DDL, REALIZED_VOLATILITY_INDEXES),
        ("vrp_metrics", VRP_METRICS_DDL, VRP_METRICS_INDEXES),
        ("interest_rate_term_structure", INTEREST_RATE_TERM_STRUCTURE_DDL, INTEREST_RATE_TERM_STRUCTURE_INDEXES),
    ]

    for table_name, ddl, indexes in tables:
        logger.debug(f"Creating table {table_name}")
        conn.execute(ddl)
        for idx_ddl in indexes:
            conn.execute(idx_ddl)

    logger.info("Database schema initialized successfully")


def drop_all_tables(conn: Optional[DuckDBPyConnection] = None) -> None:
    """
    Drop all tables (use with caution!).

    Args:
        conn: Optional connection to use. If None, uses global connection.
    """
    if conn is None:
        conn = get_connection()

    logger.warning("Dropping all tables...")

    tables = [
        "vrp_metrics",
        "realized_volatility",
        "term_structure_params",
        "fitted_surfaces",
        "underlying_prices",
        "raw_options_chains",
        "interest_rate_term_structure",
    ]

    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    sequences = [
        "seq_chain_id",
        "seq_price_id",
        "seq_surface_id",
        "seq_ts_id",
        "seq_rv_id",
        "seq_vrp_id",
        "seq_rate_id",
    ]

    for seq in sequences:
        conn.execute(f"DROP SEQUENCE IF EXISTS {seq}")

    logger.info("All tables dropped")
