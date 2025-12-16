# Volatility Surface Data Lake - Design Specification

## Project Overview

A data lake system for options chain data with fitted volatility surfaces, enabling sophisticated volatility trading analysis. The system will ingest historical and ongoing options data, fit multiple volatility surface models, and provide both programmatic and visual interfaces for analysis.

**MVP Scope:** SPY options only, end-of-day updates, SVI surface fitting
**Future Expansion:** Multi-asset universe, intraday/streaming updates, multiple surface models

---

## 1. System Architecture

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  (Theta Data API → Raw Options Chains → DuckDB Storage)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                      │
│  (Filtering, Liquidity Checks, Data Quality Validation)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Surface Fitting Engine                      │
│  (SVI/GVV Models, No-Arbitrage Constraints, Calibration)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Analytics & Metrics Layer                  │
│  (VRP, Realized Vol, Surface Parameters, Greeks)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  (CLI Dashboard, Jupyter Notebooks, Local Web Interface)    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Tech Stack

- **Language:** Python 3.12+
- **Package Manager:** uv
- **Data Processing:** Polars, NumPy
- **Database:** DuckDB (embedded, single-file mode)
- **Optimization:** SciPy, possibly cvxpy for constrained optimization
- **Visualization:** Plotly (interactive), matplotlib (notebooks)
- **CLI Framework:** Typer or Rich
- **Web Dashboard:** Streamlit or FastAPI + React (TBD based on complexity)
- **Notebooks:** Jupyter Lab
- **Testing:** pytest
- **Type Checking:** mypy, ruff for linting

---

## 2. Data Model & Schema

### 2.1 Database Structure (DuckDB)

#### Table: `raw_options_chains`
Stores complete daily options chain snapshots.

```sql
CREATE TABLE raw_options_chains (
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
    
    -- Greeks (if provided by Theta Data)
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
    is_liquid BOOLEAN,  -- based on bid-ask spread, volume, OI thresholds
    
    -- Metadata
    data_source VARCHAR DEFAULT 'theta_data',
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_symbol_date (symbol, quote_date),
    INDEX idx_expiration (expiration_date),
    INDEX idx_liquid (is_liquid)
);
```

#### Table: `underlying_prices`
Daily underlying asset prices for realized vol calculations.

```sql
CREATE TABLE underlying_prices (
    price_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4),
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    volume BIGINT,
    
    UNIQUE(symbol, date),
    INDEX idx_symbol_date (symbol, date)
);
```

#### Table: `fitted_surfaces`
Stores calibrated surface parameters for each date and expiration.

```sql
CREATE TABLE fitted_surfaces (
    surface_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quote_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    
    -- Time to expiration
    tte_years DECIMAL(10, 8),  -- Time to expiration in years
    
    -- SVI Parameters (Raw parameterization)
    -- Total variance: w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))
    svi_a DECIMAL(10, 8),      -- Vertical shift (ATM variance level)
    svi_b DECIMAL(10, 8),      -- Slope of the wings
    svi_rho DECIMAL(10, 8),    -- Skew parameter (-1 to 1)
    svi_m DECIMAL(10, 8),      -- Horizontal shift (ATM moneyness)
    svi_sigma DECIMAL(10, 8),  -- Smoothness of the smile
    
    -- Derived quantities for convenience
    atm_vol DECIMAL(8, 6),     -- ATM implied volatility
    skew_25delta DECIMAL(8, 6), -- 25-delta skew
    
    -- Fit quality metrics
    rmse DECIMAL(10, 8),        -- Root mean squared error
    mae DECIMAL(10, 8),         -- Mean absolute error
    max_error DECIMAL(10, 8),   -- Maximum absolute error
    num_points INTEGER,         -- Number of data points used in fit
    
    -- Arbitrage checks
    passes_no_arbitrage BOOLEAN,
    butterfly_arbitrage_violations INTEGER,
    calendar_arbitrage_violations INTEGER,
    
    -- Model metadata
    model_type VARCHAR DEFAULT 'SVI',
    model_version VARCHAR,
    fit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, quote_date, expiration_date, model_type),
    INDEX idx_symbol_date (symbol, quote_date),
    INDEX idx_expiration (expiration_date)
);
```

#### Table: `term_structure_params`
Aggregated term structure parameters across all expirations for a given date.

```sql
CREATE TABLE term_structure_params (
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
    
    UNIQUE(symbol, quote_date, model_type),
    INDEX idx_symbol_date (symbol, quote_date)
);
```

#### Table: `realized_volatility`
Historical realized volatility calculations at various windows.

```sql
CREATE TABLE realized_volatility (
    rv_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    
    -- Realized vol at different windows (annualized)
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
    
    UNIQUE(symbol, date),
    INDEX idx_symbol_date (symbol, date)
);
```

#### Table: `vrp_metrics`
Variance Risk Premium calculations (implied vol - realized vol).

```sql
CREATE TABLE vrp_metrics (
    vrp_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    
    -- VRP at different horizons (ATM vol - realized vol)
    vrp_30d DECIMAL(8, 6),   -- 30-day VRP
    vrp_60d DECIMAL(8, 6),   -- 60-day VRP
    vrp_90d DECIMAL(8, 6),   -- 90-day VRP
    
    -- Corresponding implied and realized vols
    implied_vol_30d DECIMAL(8, 6),
    realized_vol_30d DECIMAL(8, 6),
    
    -- Statistical measures
    vrp_zscore DECIMAL(8, 6),  -- Z-score relative to historical VRP
    
    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, date),
    INDEX idx_symbol_date (symbol, date)
);
```

### 2.2 Data Quality & Liquidity Filters

Options will be marked as liquid (`is_liquid = TRUE`) if they meet:
- Bid-ask spread < X% of mid (e.g., 20% for initial implementation)
- Open interest > Y contracts (e.g., 50)
- Volume > 0 (for the day)
- Days to expiration between 7 and 730 days (exclude very short and very long-dated)
- Moneyness within reasonable bounds (e.g., 0.7 < S/K < 1.3)

---

## 3. Data Ingestion Pipeline

### 3.1 Theta Data API Integration

**Module:** `src/ingestion/theta_client.py`

Key responsibilities:
- Authenticate with Theta Data API
- Fetch EOD options chain snapshots for specified symbols
- Fetch historical data with date range pagination
- Handle rate limiting and retries
- Transform API responses to Polars DataFrames

**API Endpoints to Use:**
- `/hist/option/quote` - Historical options quotes
- `/hist/stock/quote` - Historical underlying prices
- `/snapshot/option/quote` - Real-time snapshots (future enhancement)

**Implementation Notes:**
- Use `httpx` for async API calls
- Implement exponential backoff for rate limits
- Cache API responses locally (optional, for development)
- Log all API calls and responses for debugging

### 3.2 Data Pipeline Orchestration

**Module:** `src/ingestion/pipeline.py`

```python
class IngestionPipeline:
    """Orchestrates daily data ingestion and processing."""
    
    async def run_daily_ingestion(self, symbol: str, date: date) -> None:
        """
        1. Fetch options chain from Theta Data
        2. Fetch underlying price data
        3. Apply liquidity filters
        4. Insert into raw_options_chains and underlying_prices tables
        5. Trigger surface fitting pipeline
        """
        pass
    
    async def backfill_historical(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> None:
        """Backfill historical data in batches."""
        pass
```

**CLI Command:**
```bash
# Backfill all historical data for SPY
uv run python -m volsurf ingest backfill SPY --start 2020-01-01 --end 2024-12-16

# Run daily update
uv run python -m volsurf ingest daily SPY
```

---

## 4. Volatility Surface Fitting

### 4.1 SVI Model Implementation

**Module:** `src/models/svi.py`

**SVI Parameterization (Raw):**
Total implied variance as a function of log-moneyness:

```
w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))

where:
- k = log(K/F) = log-moneyness (F = forward price)
- a: vertical shift (level of ATM variance)
- b: slope of the wings
- ρ: correlation (skew), -1 < ρ < 1
- m: horizontal translation (ATM log-moneyness)
- σ: smoothness parameter, σ > 0
```

**Constraints for No-Arbitrage:**
1. `b >= 0` (no calendar arbitrage)
2. `a + b * σ * sqrt(1 - ρ^2) >= 0` (non-negative variance)
3. `|ρ| < 1` (valid correlation)

**Fitting Procedure:**
```python
def fit_svi_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward_price: float,
    time_to_expiry: float,
    weights: Optional[np.ndarray] = None
) -> SVIParams:
    """
    Fit SVI model to a single expiry slice.
    
    Args:
        strikes: Array of strike prices
        implied_vols: Array of implied volatilities
        forward_price: Forward price of underlying
        time_to_expiry: Time to expiration in years
        weights: Optional weights for each data point
    
    Returns:
        SVIParams object with fitted parameters and diagnostics
    """
    # 1. Convert strikes to log-moneyness
    k = np.log(strikes / forward_price)
    
    # 2. Convert implied vols to total variance
    w = implied_vols**2 * time_to_expiry
    
    # 3. Set up optimization problem with constraints
    # Use scipy.optimize.minimize with method='SLSQP'
    
    # 4. Compute fit metrics (RMSE, MAE, max error)
    
    # 5. Check no-arbitrage conditions
    
    return svi_params
```

### 4.2 Surface Fitting Pipeline

**Module:** `src/fitting/pipeline.py`

```python
class SurfaceFittingPipeline:
    """Fits volatility surfaces to daily options data."""
    
    def fit_all_expirations(
        self, 
        symbol: str, 
        quote_date: date
    ) -> List[FittedSurface]:
        """
        Fit SVI model to all liquid expirations for a given date.
        
        1. Query liquid options from raw_options_chains
        2. Group by expiration date
        3. For each expiration:
           - Filter to liquid strikes
           - Fit SVI model
           - Validate no-arbitrage
           - Compute fit metrics
        4. Insert into fitted_surfaces table
        5. Return list of fitted surfaces
        """
        pass
    
    def fit_term_structure(
        self,
        symbol: str,
        quote_date: date
    ) -> TermStructureParams:
        """
        Fit term structure model to surface parameters across expirations.
        
        1. Query fitted_surfaces for the given date
        2. Extract ATM vols and skews across TTEs
        3. Fit power law: σ(T) = a * T^b
        4. Insert into term_structure_params table
        """
        pass
```

**CLI Command:**
```bash
# Fit surfaces for a specific date
uv run python -m volsurf fit surfaces SPY --date 2024-12-16

# Fit term structure
uv run python -m volsurf fit term-structure SPY --date 2024-12-16

# Batch fit for date range
uv run python -m volsurf fit batch SPY --start 2024-01-01 --end 2024-12-16
```

---

## 5. Analytics & Metrics

### 5.1 Realized Volatility Calculations

**Module:** `src/analytics/realized_vol.py`

Implement multiple estimators:

1. **Close-to-Close:** Standard deviation of log returns
2. **Parkinson:** Uses high-low range (more efficient)
3. **Garman-Klass:** Uses OHLC data (even more efficient)

```python
def calculate_realized_vol(
    prices_df: pl.DataFrame,
    windows: List[int] = [10, 21, 63, 252]
) -> pl.DataFrame:
    """
    Calculate realized volatility at multiple windows.
    
    Args:
        prices_df: DataFrame with columns [date, open, high, low, close]
        windows: List of lookback windows in days
    
    Returns:
        DataFrame with realized vol estimates
    """
    pass
```

### 5.2 Variance Risk Premium (VRP)

**Module:** `src/analytics/vrp.py`

```python
def calculate_vrp(
    symbol: str,
    date: date,
    horizons: List[int] = [30, 60, 90]
) -> VRPMetrics:
    """
    Calculate VRP as implied vol minus realized vol.
    
    1. Get ATM implied vol for each horizon from fitted_surfaces
    2. Get corresponding realized vol from realized_volatility table
    3. Compute VRP = implied - realized
    4. Calculate z-score relative to historical VRP distribution
    """
    pass
```

### 5.3 Surface Comparison Metrics

**Module:** `src/analytics/surface_comparison.py`

For comparing surfaces across time or across models:

```python
def compare_surfaces(
    surface1: FittedSurface,
    surface2: FittedSurface,
    comparison_strikes: np.ndarray
) -> SurfaceComparison:
    """
    Compare two fitted surfaces.
    
    Metrics:
    - Parameter differences (Δa, Δb, Δρ, Δm, Δσ)
    - ATM vol change
    - Skew change
    - RMSE between interpolated vols
    """
    pass
```

---

## 6. Presentation Layer

### 6.1 CLI Dashboard

**Module:** `src/cli/dashboard.py`

Use `rich` library for beautiful terminal UI.

```bash
# Main dashboard command
uv run python -m volsurf dashboard SPY

# Display sections:
# 1. Latest surface snapshot (table of strikes, IVs, Greeks)
# 2. ATM term structure chart (ASCII art or redirect to web)
# 3. Recent VRP metrics
# 4. Surface parameter evolution (last 30 days)
```

**Features:**
- Interactive navigation with arrow keys
- Real-time refresh (for streaming data in future)
- Export views to CSV/JSON
- Quick surface visualizations in terminal (using `plotext`)

### 6.2 Web Dashboard

**Framework:** Streamlit (rapid prototyping) or FastAPI + React (production-grade)

**Initial Pages:**
1. **Surface Viewer**
   - 3D surface plot (plotly)
   - Slice view (volatility smile at selected expiry)
   - Time slider to animate historical surfaces
   
2. **Term Structure**
   - ATM vol term structure
   - Skew term structure
   - Parameter evolution over time
   
3. **VRP Analysis**
   - Time series of VRP
   - Implied vs realized vol scatter
   - VRP distribution and percentiles
   
4. **Model Diagnostics**
   - Fit quality metrics
   - Residual plots
   - Arbitrage violation alerts

**Running the Dashboard:**
```bash
uv run streamlit run src/web/app.py
# or
uv run python -m volsurf web --port 8501
```

### 6.3 Jupyter Notebooks

**Location:** `notebooks/`

Create template notebooks for common analyses:

1. `01_data_exploration.ipynb`
   - Basic data quality checks
   - Options chain visualization
   - Liquidity distribution analysis

2. `02_surface_fitting.ipynb`
   - Fit SVI model interactively
   - Visualize fit quality
   - Experiment with different parameterizations

3. `03_vrp_analysis.ipynb`
   - Calculate and visualize VRP
   - Statistical properties of VRP
   - Trading signal backtests

4. `04_model_comparison.ipynb`
   - Compare SVI vs other models
   - Parameter stability analysis
   - Out-of-sample fit quality

---

## 7. Project Structure

```
vol-surface-data-lake/
├── pyproject.toml              # uv project configuration
├── README.md                   # Project documentation
├── .env.example                # Environment variables template
├── .gitignore
│
├── data/
│   ├── raw/                    # Raw data cache (gitignored)
│   ├── processed/              # Processed data cache
│   └── volsurf.duckdb         # Main DuckDB database
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_surface_fitting.ipynb
│   ├── 03_vrp_analysis.ipynb
│   └── 04_model_comparison.ipynb
│
├── src/
│   ├── volsurf/
│   │   ├── __init__.py
│   │   ├── __main__.py         # CLI entry point
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py     # Configuration management
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py       # Table definitions
│   │   │   ├── connection.py   # DuckDB connection management
│   │   │   └── migrations.py   # Schema migrations
│   │   │
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── theta_client.py # Theta Data API client
│   │   │   ├── pipeline.py     # Ingestion orchestration
│   │   │   └── filters.py      # Liquidity filters
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Abstract base model
│   │   │   ├── svi.py          # SVI implementation
│   │   │   └── schemas.py      # Pydantic models for parameters
│   │   │
│   │   ├── fitting/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py     # Fitting orchestration
│   │   │   ├── optimizer.py    # Optimization routines
│   │   │   └── validation.py   # No-arbitrage checks
│   │   │
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   ├── realized_vol.py # Realized volatility
│   │   │   ├── vrp.py          # Variance risk premium
│   │   │   ├── greeks.py       # Greeks calculations
│   │   │   └── surface_comparison.py
│   │   │
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   ├── main.py         # Main CLI app (Typer)
│   │   │   ├── dashboard.py    # Terminal dashboard
│   │   │   └── commands/       # CLI command modules
│   │   │       ├── ingest.py
│   │   │       ├── fit.py
│   │   │       └── analyze.py
│   │   │
│   │   ├── web/
│   │   │   ├── __init__.py
│   │   │   ├── app.py          # Streamlit app
│   │   │   └── pages/          # Multi-page app structure
│   │   │       ├── surface_viewer.py
│   │   │       ├── term_structure.py
│   │   │       ├── vrp_analysis.py
│   │   │       └── diagnostics.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py      # Logging configuration
│   │       ├── dates.py        # Date utilities
│   │       └── math.py         # Mathematical utilities
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py         # pytest fixtures
│       ├── test_svi.py
│       ├── test_fitting.py
│       ├── test_analytics.py
│       └── test_integration.py
│
├── scripts/
│   ├── setup_database.py       # Initialize database schema
│   ├── backfill_data.sh        # Backfill historical data
│   └── daily_update.sh         # Daily cron job script
│
└── docs/
    ├── api_reference.md
    ├── model_details.md
    └── deployment.md
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal:** Basic data ingestion and storage

- [ ] Set up project structure with uv
- [ ] Implement DuckDB schema and connection management
- [ ] Build Theta Data API client
- [ ] Create basic ingestion pipeline for SPY
- [ ] Implement liquidity filters
- [ ] Write tests for data ingestion
- [ ] Run initial backfill for 1 year of SPY data

**Deliverable:** Populated DuckDB database with SPY options history

### Phase 2: Surface Fitting (Week 3-4)
**Goal:** Fit SVI surfaces to historical data

- [ ] Implement SVI model with no-arbitrage constraints
- [ ] Build surface fitting pipeline
- [ ] Add term structure fitting
- [ ] Create fit quality metrics and validation
- [ ] Batch fit all historical data
- [ ] Write tests for fitting algorithms

**Deliverable:** `fitted_surfaces` table populated with SVI parameters

### Phase 3: Analytics (Week 5)
**Goal:** Core volatility trading metrics

- [ ] Implement realized volatility calculations (multiple estimators)
- [ ] Build VRP calculation pipeline
- [ ] Add surface comparison utilities
- [ ] Create Greeks calculations (if not from Theta Data)
- [ ] Write tests for analytics

**Deliverable:** Complete analytics pipeline with VRP metrics

### Phase 4: Visualization (Week 6)
**Goal:** Interactive dashboards and notebooks

- [ ] Build CLI dashboard with rich
- [ ] Create Streamlit web dashboard
  - Surface viewer (3D + slices)
  - Term structure viewer
  - VRP analysis page
  - Model diagnostics page
- [ ] Develop Jupyter notebook templates
- [ ] Add export functionality (CSV, JSON, plots)

**Deliverable:** Fully functional visualization layer

### Phase 5: Refinement & Multi-Model (Week 7+)
**Goal:** Polish and expand capabilities

- [ ] Add alternative surface models (SABR, Heston, polynomial)
- [ ] Implement model comparison framework
- [ ] Optimize performance (vectorization, caching)
- [ ] Add data validation and monitoring
- [ ] Create comprehensive documentation
- [ ] Set up automated daily updates

**Deliverable:** Production-ready system with multiple models

### Phase 6: Future Enhancements
**Goals for later iterations:**

- [ ] Expand to multi-asset universe
- [ ] Implement intraday data ingestion
- [ ] Add real-time streaming data support
- [ ] Build advanced term structure models (SSVI)
- [ ] Create trading signal generation framework
- [ ] Add backtesting capabilities
- [ ] Implement risk management tools
- [ ] Add machine learning surface predictors

---

## 9. Configuration Management

### 9.1 Environment Variables (`.env`)

```bash
# Theta Data API
THETA_API_KEY=your_api_key_here
THETA_API_URL=https://api.thetadata.us/v2

# Database
DUCKDB_PATH=data/volsurf.duckdb

# Data Ingestion
DEFAULT_SYMBOL=SPY
INGESTION_START_DATE=2020-01-01

# Liquidity Filters
MIN_OPEN_INTEREST=50
MAX_BID_ASK_SPREAD_PCT=0.20
MIN_DTE=7
MAX_DTE=730
MIN_MONEYNESS=0.7
MAX_MONEYNESS=1.3

# Surface Fitting
SVI_MAX_ITERATIONS=1000
SVI_TOLERANCE=1e-8
MIN_STRIKES_PER_FIT=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/volsurf.log

# Web Dashboard
STREAMLIT_PORT=8501
```

### 9.2 Configuration Class

```python
# src/volsurf/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API
    theta_api_key: str
    theta_api_url: str = "https://api.thetadata.us/v2"
    
    # Database
    duckdb_path: Path = Path("data/volsurf.duckdb")
    
    # Ingestion
    default_symbol: str = "SPY"
    ingestion_start_date: str = "2020-01-01"
    
    # Liquidity
    min_open_interest: int = 50
    max_bid_ask_spread_pct: float = 0.20
    min_dte: int = 7
    max_dte: int = 730
    min_moneyness: float = 0.7
    max_moneyness: float = 1.3
    
    # Fitting
    svi_max_iterations: int = 1000
    svi_tolerance: float = 1e-8
    min_strikes_per_fit: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/volsurf.log")
    
    # Web
    streamlit_port: int = 8501
    
    class Config:
        env_file = ".env"
```

---

## 10. Key Dependencies (pyproject.toml)

```toml
[project]
name = "vol-surface-data-lake"
version = "0.1.0"
description = "Options data lake with volatility surface fitting"
requires-python = ">=3.12"

dependencies = [
    # Core
    "polars>=0.20.0",
    "duckdb>=0.10.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    
    # API & HTTP
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    
    # CLI
    "typer>=0.9.0",
    "rich>=13.7.0",
    "plotext>=5.2.8",
    
    # Visualization
    "plotly>=5.18.0",
    "matplotlib>=3.8.0",
    "streamlit>=1.30.0",
    
    # Notebooks
    "jupyter>=1.0.0",
    "ipykernel>=6.28.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "loguru>=0.7.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.8.0",
    "ruff>=0.1.11",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

---

## 11. API Client Design (Theta Data)

### 11.1 Core Client Methods

```python
# src/volsurf/ingestion/theta_client.py

class ThetaDataClient:
    """Client for Theta Data API."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_options_chain(
        self,
        symbol: str,
        date: datetime.date,
        exp_min: Optional[datetime.date] = None,
        exp_max: Optional[datetime.date] = None
    ) -> pl.DataFrame:
        """
        Fetch options chain for a specific date.
        
        Returns Polars DataFrame with columns:
        [strike, expiration, option_type, bid, ask, volume, oi, iv, delta, ...]
        """
        pass
    
    async def get_underlying_prices(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> pl.DataFrame:
        """
        Fetch underlying OHLCV data.
        
        Returns Polars DataFrame with columns:
        [date, open, high, low, close, volume]
        """
        pass
    
    async def get_historical_chain_batch(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        batch_size: int = 10
    ) -> AsyncIterator[pl.DataFrame]:
        """
        Fetch historical chains in batches (for backfilling).
        
        Yields DataFrames for each date or batch of dates.
        """
        pass
```

---

## 12. Performance Considerations

### 12.1 Database Optimization

- Use DuckDB's columnar storage efficiently
- Create appropriate indexes on frequently queried columns
- Partition data by date if dataset grows very large
- Use DuckDB's `COPY` command for bulk inserts (faster than row-by-row)

### 12.2 Fitting Optimization

- Vectorize SVI fitting where possible
- Use parallel processing for fitting multiple expirations
- Cache intermediate results (forward prices, moneyness calculations)
- Consider using JIT compilation (Numba) for hot paths

### 12.3 Data Loading

- Use Polars lazy evaluation for large datasets
- Stream data from API rather than loading all at once
- Implement data pagination for historical backfills

---

## 13. Error Handling & Monitoring

### 13.1 Data Quality Checks

- Validate option prices are positive
- Check bid <= ask
- Ensure strikes are monotonic
- Verify implied vols are within reasonable bounds (e.g., 5% - 200%)
- Flag and log data quality issues

### 13.2 Fitting Validation

- Check for arbitrage violations
- Validate parameter bounds
- Log fit failures with diagnostics
- Alert if fit quality degrades significantly

### 13.3 Logging Strategy

```python
# Use loguru for structured logging
from loguru import logger

logger.add(
    "logs/volsurf.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

# Example usage
logger.info(f"Fitting surface for {symbol} on {date}")
logger.warning(f"Low liquidity for {expiration}: only {n} strikes")
logger.error(f"Fit failed for {expiration}: {error}")
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

- Test SVI model with known parameter sets
- Test no-arbitrage constraint checks
- Test realized vol calculations with synthetic data
- Test liquidity filters

### 14.2 Integration Tests

- Test full ingestion pipeline with mock API
- Test fitting pipeline end-to-end
- Test database operations (insert, query, update)

### 14.3 Validation Tests

- Fit surfaces to synthetic data with known parameters
- Validate arbitrage-free surfaces
- Check numerical stability of optimization

---

## 15. Documentation Plan

### 15.1 Code Documentation

- Docstrings for all public functions (Google style)
- Type hints throughout codebase
- Inline comments for complex algorithms

### 15.2 User Documentation

- README with quick start guide
- API reference (auto-generated with Sphinx)
- Model details (mathematical formulations)
- CLI command reference
- Deployment guide

### 15.3 Developer Documentation

- Architecture overview
- Database schema documentation
- Contributing guide
- Release process

---

## 16. Deployment & Operations

### 16.1 Daily Operations

```bash
# Cron job for daily updates (runs at 6 PM after market close)
0 18 * * 1-5 /path/to/scripts/daily_update.sh

# daily_update.sh
#!/bin/bash
cd /path/to/vol-surface-data-lake
uv run python -m volsurf ingest daily SPY
uv run python -m volsurf fit surfaces SPY --date today
uv run python -m volsurf fit term-structure SPY --date today
uv run python -m volsurf analyze vrp SPY --date today
```

### 16.2 Monitoring

- Track API usage and rate limits
- Monitor fit quality metrics over time
- Alert on missing data or failed fits
- Log database size and query performance

### 16.3 Backup Strategy

- Daily backup of DuckDB database
- Version control configuration files
- Archive raw data periodically

---

## 17. Future Enhancements & Research Ideas

### 17.1 Model Extensions

1. **SSVI (Surface SVI):** Joint calibration across all expirations
2. **SABR Model:** Stochastic volatility model with analytic approximations
3. **Local Volatility:** Dupire's formula for model-free surfaces
4. **Jump Models:** Merton, Bates for handling tail risk
5. **Machine Learning:** Neural network-based surface interpolation

### 17.2 Advanced Analytics

1. **Skew Dynamics:** Model and predict skew evolution
2. **Vol Clustering:** Identify regimes in volatility behavior
3. **Cross-Asset Analysis:** Compare surfaces across SPX, NDX, RUT
4. **Options Flow Analysis:** Incorporate volume and OI trends
5. **Earnings Analysis:** Special handling for earnings events

### 17.3 Trading Applications

1. **Relative Value:** Identify mispriced options vs fitted surface
2. **Dispersion Trading:** Compare index vs single-stock vol
3. **Calendar Spreads:** Exploit term structure anomalies
4. **Skew Trading:** Trade realized skew vs implied skew
5. **Vol Arbitrage:** Statistical arbitrage based on VRP

---

## 18. Success Metrics

### MVP Success Criteria

- [ ] Successfully ingest 1+ year of SPY options data
- [ ] Fit SVI surfaces with RMSE < 1% on average
- [ ] Calculate VRP metrics for all historical dates
- [ ] Generate visualizations in both CLI and web interface
- [ ] Run daily updates without manual intervention

### Quality Metrics

- **Data Coverage:** >95% of trading days have complete data
- **Fit Quality:** Median RMSE < 0.5%, 95th percentile < 2%
- **No-Arbitrage:** <1% of fits violate arbitrage conditions
- **Performance:** Daily update completes in <10 minutes
- **Stability:** No failed fits due to numerical issues

---

## 19. Risk & Mitigations

### Risk 1: API Rate Limits
**Mitigation:** 
- Implement exponential backoff
- Cache aggressively during development
- Monitor usage and upgrade tier proactively

### Risk 2: Poor Fit Quality
**Mitigation:**
- Start with very liquid options only
- Implement robust initial parameter estimation
- Use multiple optimization algorithms as fallback

### Risk 3: Data Storage Growth
**Mitigation:**
- DuckDB handles multi-GB data efficiently
- Implement data retention policies
- Archive old raw data, keep derived metrics

### Risk 4: Computation Time
**Mitigation:**
- Parallelize fitting across expirations
- Use incremental updates rather than full refits
- Optimize hot paths with Numba/Cython if needed

---

## 20. Open Questions & Decisions Needed

1. **Forward Price Calculation:** Use spot + synthetic forward from put-call parity, or just spot? (Recommend: put-call parity for better accuracy)

2. **Weighting Scheme for Fitting:** Weight by vega, open interest, or equal weights? (Recommend: vega weighting to emphasize ATM)

3. **Handling Extreme Strikes:** Should we exclude very far OTM options even if liquid? (Recommend: cap at 2-3 standard deviations)

4. **Term Structure Model:** Power law sufficient or need more sophisticated model? (Recommend: start with power law, iterate based on fit quality)

5. **Greeks Source:** Calculate ourselves or use Theta Data's Greeks? (Recommend: use Theta Data initially, can compute independently later for validation)

6. **SVI Variant:** Raw SVI, Jump-Wing, or Natural parameterization? (Recommend: Raw for stability, can experiment with others)

7. **Time Convention:** Trading days (252) or calendar days (365)? (Recommend: calendar days for simplicity, ensure consistency)

---

## Conclusion

This specification provides a comprehensive roadmap for building a production-grade volatility surface data lake. The system is designed to be:

- **Modular:** Each component can be developed and tested independently
- **Scalable:** Easily extends to multiple assets and models
- **Maintainable:** Clean code structure with modern Python tooling
- **Performant:** Leverages Polars, DuckDB, and vectorization
- **Practical:** Focuses on trader-relevant metrics and visualizations

The phased implementation allows for rapid iteration while building toward a robust, feature-complete system. Starting with SPY and SVI provides a solid MVP that can be extended incrementally.

**Next Steps:**
1. Review and refine this specification
2. Set up development environment with uv
3. Initialize project structure
4. Begin Phase 1 implementation

Good luck with the build! 🚀
