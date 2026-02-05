## Overview

Personal vol training tool. Daily loop: scan watchlist → spot mispricings → log hypothetical trades → track accuracy. Three modules built in phases, with Phases 1-3 constituting the MVP.

**Watchlist**: SPY, QQQ, GLD, SLV, NVDA, GOOG, TSLA, PLTR, NFLX, AMD, GME

---

## Phase 1: Data Foundation

**Goal**: All 11 symbols ingested, fitted, and analyzed. Create denormalized `surface_metrics_history` table that underpins percentile/z-score calculations and the watchlist heat map.

### 1A. Multi-Symbol Batch CLI

**New file**: `src/volsurf/cli/commands/batch.py`

Orchestrator that iterates over the watchlist, calling existing single-symbol functions. Commands:

| Command | Description |
|---------|-------------|
| `volsurf batch ingest --start DATE --end DATE` | Backfill chains + underlying prices for all watchlist symbols |
| `volsurf batch fit --start DATE --end DATE` | Batch fit SVI surfaces for all symbols |
| `volsurf batch analyze --start DATE --end DATE` | Run RV, VRP, and surface metrics history for all symbols |
| `volsurf batch status` | Rich table showing data coverage across all symbols |

Each command wraps the existing `IngestionPipeline.backfill_historical()`, `FittingPipeline.batch_fit_surfaces()`, `RealizedVolCalculator.backfill()`, `VRPCalculator.backfill()`, and the new `SurfaceMetricsHistoryCalculator.backfill()` in a per-symbol loop with Rich progress bars and error handling.

**Modifications**:
- `src/volsurf/cli/main.py` — register `batch` sub-app
- `src/volsurf/config/settings.py` — add `watchlist: list[str]` field (default: the 11 symbols, configurable via `WATCHLIST` env var as comma-separated)

### 1B. Surface Metrics History Table

**Rationale**: `fitted_surfaces` has one row per symbol/date/expiration. The heat map and percentile calculations need one row per symbol/date with interpolated metrics at standard tenors. Pre-computing avoids expensive interpolation on every page load. Mirrors the pattern of `realized_volatility` and `vrp_metrics`.

**New DDL in `src/volsurf/database/schema.py`**:

```sql
CREATE TABLE IF NOT EXISTS surface_metrics_history (
    smh_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quote_date DATE NOT NULL,

    -- ATM vol at standard tenors (interpolated from fitted surfaces)
    atm_vol_30d DECIMAL(8, 6),
    atm_vol_60d DECIMAL(8, 6),
    atm_vol_90d DECIMAL(8, 6),

    -- 25-delta skew at standard tenors
    skew_25d_30d DECIMAL(8, 6),
    skew_25d_60d DECIMAL(8, 6),
    skew_25d_90d DECIMAL(8, 6),

    -- Term structure slope (90d ATM - 30d ATM)
    term_slope DECIMAL(8, 6),

    -- Butterfly (25d put IV + 25d call IV - 2*ATM IV, at 30d tenor)
    butterfly_30d DECIMAL(8, 6),

    -- Cross-reference (denormalized for fast heat map queries)
    rv_21d DECIMAL(8, 6),
    rv_63d DECIMAL(8, 6),
    vrp_30d DECIMAL(8, 6),
    underlying_close DECIMAL(10, 4),

    -- Day-over-day changes (computed during backfill)
    atm_vol_30d_1d_chg DECIMAL(8, 6),
    skew_25d_30d_1d_chg DECIMAL(8, 6),
    term_slope_1d_chg DECIMAL(8, 6),

    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, quote_date)
);
```

**New file**: `src/volsurf/analytics/surface_metrics_history.py`

Follows the exact Calculator pattern from `variance_risk_premium.py`:

```
@dataclass SurfaceMetricsSnapshot — one row of the table above

class SurfaceMetricsHistoryCalculator:
    calculate_for_date(symbol, date) → SurfaceMetricsSnapshot
        # Interpolates ATM vol and skew to 30d/60d/90d using existing
        # SurfaceMetrics.get_atm_vol_timeseries logic (single-date version)
        # Pulls rv_21d/rv_63d from realized_volatility table
        # Pulls vrp_30d from vrp_metrics table
        # Computes butterfly from SVI params at 30d tenor
        # Computes term_slope as atm_vol_90d - atm_vol_30d

    calculate_day_over_day(current, previous) → fills _1d_chg fields

    store_result(snapshot) → upsert into surface_metrics_history

    backfill(symbol, start, end) → iterate dates, calculate, store

    get_timeseries(symbol, start, end) → pd.DataFrame

    get_latest_snapshot(symbol) → Optional[SurfaceMetricsSnapshot]

    get_all_symbols_latest() → pd.DataFrame (one row per symbol, powers heat map)

    calculate_percentile_rank(symbol, metric, current_value, lookback_days=252) → float
        # Where current_value sits in trailing distribution (0-100)
        # Uses DuckDB PERCENT_RANK() window function for efficiency

    calculate_zscore(symbol, metric, current_value, lookback_days=252) → float
        # Same pattern as VRPCalculator.calculate_vrp_zscore

    get_enriched_snapshot(symbol, date, lookback_windows=[30,60,90,252]) → dict
        # Snapshot + percentile ranks + z-scores at multiple windows
        # Powers the single-symbol detail view

    get_watchlist_heatmap_data(symbols, date, lookback_days=252) → pd.DataFrame
        # For all symbols: latest snapshot + percentiles + z-scores
        # Single efficient DuckDB query with window functions
```

**Other modifications for Phase 1B**:
- `src/volsurf/database/schema.py` — add DDL, sequence, indexes, register in `init_schema()`
- `src/volsurf/analytics/__init__.py` — export new classes
- `src/volsurf/cli/commands/batch.py` — `batch analyze` calls the new calculator

---

## Phase 2: Watchlist Dashboard & Historical Context

**Goal**: Build the watchlist heat map page and enhance per-symbol views with percentile ranks and day-over-day change indicators.

### 2A. Watchlist Heat Map Page

**New file**: `src/volsurf/web/components/watchlist.py`

```
render_watchlist(symbols, target_date) → main component
    Tab 1: Heat Map Table
    Tab 2: (future) Sector/category groupings

render_heatmap_table(df) → core table using st.dataframe with conditional formatting
    Columns:
    - Symbol
    - Price
    - ATM Vol 30d (colored by percentile)
    - ATM Vol Pctile 252d (red <10, green >90)
    - Skew 30d
    - Skew Z-Score (colored by magnitude)
    - Term Slope
    - VRP 30d
    - 1d Δ ATM Vol (red/green)
    - 1d Δ Skew

render_symbol_detail_card(symbol, enriched_snapshot) → expandable detail
    Percentile bar charts for each metric across lookback windows
    Sparkline time series (last 30d)
```

**Data flow**: `SurfaceMetricsHistoryCalculator.get_watchlist_heatmap_data()` → single DuckDB query returning one row per symbol with all display columns pre-computed.

### 2B. Day-Over-Day Change Indicators

Enhance the existing Overview page in `src/volsurf/web/app.py`:
- Add `delta` parameter to `st.metric` calls using pre-computed `_1d_chg` fields
- Add percentile rank display next to each metric
- Example: `st.metric("ATM Vol (30d)", "18.5%", delta="+0.8%")` with "72nd pctile (252d)" caption

### 2C. Percentile Context on Existing Pages

- `src/volsurf/web/components/term_structure.py` — add 10th/90th percentile bands on ATM evolution chart
- `src/volsurf/web/components/vrp_analysis.py` — add percentile rank context to VRP display

### 2D. Navigation Changes

**Modify `src/volsurf/web/app.py`**:
- Add "Watchlist" as the first/default page in nav
- Add "Trade Journal" placeholder in nav (filled in Phase 3)
- When user selects a symbol from watchlist table → update `st.session_state` and navigate to Overview
- Navigation: `["Watchlist", "Overview", "Surface Viewer", "Term Structure", "VRP Analysis", "Intraday", "Diagnostics", "Trade Journal"]`

---

## Phase 3: Trade Journal (MVP Completion)

**Goal**: Close the feedback loop. Log hypothetical trades, track against market data, review performance.

### 3A. Trade Journal Schema

**New DDL in `src/volsurf/database/schema.py`**:

```sql
CREATE TABLE IF NOT EXISTS trade_journal (
    trade_id BIGINT PRIMARY KEY,

    -- Entry
    entry_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    thesis TEXT NOT NULL,
    category VARCHAR NOT NULL,
        -- 'surface_anomaly', 'event_mispricing', 'rv_vs_iv',
        -- 'skew_trade', 'term_structure', 'directional_vol'
    direction VARCHAR NOT NULL,
        -- 'long_vol', 'short_vol', 'long_skew', 'short_skew', 'calendar', 'other'
    structure_description TEXT,

    -- Entry levels
    entry_iv_level DECIMAL(8, 6),
    entry_skew_level DECIMAL(8, 6),
    entry_underlying_price DECIMAL(10, 4),
    entry_metric_value DECIMAL(10, 6),
    entry_metric_name VARCHAR,

    -- Target and stop
    target_description TEXT,
    stop_description TEXT,
    time_horizon_days INTEGER,

    -- Resolution
    exit_date DATE,
    exit_iv_level DECIMAL(8, 6),
    exit_skew_level DECIMAL(8, 6),
    exit_underlying_price DECIMAL(10, 4),
    exit_metric_value DECIMAL(10, 6),

    -- P&L
    pnl_vol_points DECIMAL(8, 6),
    pnl_direction_correct BOOLEAN,
    outcome VARCHAR DEFAULT 'open',
        -- 'win', 'loss', 'scratch', 'open'
    outcome_notes TEXT,

    -- Auto-populated context at entry (from surface_metrics_history)
    entry_atm_vol_pctile DECIMAL(5, 2),
    entry_skew_zscore DECIMAL(8, 6),
    entry_vrp DECIMAL(8, 6),

    -- Metadata
    tags VARCHAR,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, entry_date, structure_description)
);
```

### 3B. Trade Journal Analytics

**New file**: `src/volsurf/analytics/trade_journal.py`

```
@dataclass TradeEntry — input for creating a trade
@dataclass TradeResolution — input for resolving a trade

class TradeJournalManager:
    create_trade(entry) → int (trade_id)
        # Auto-populates context fields from surface_metrics_history
    resolve_trade(resolution) → None
    auto_resolve_expired() → int (count of auto-resolved)
        # Find open trades past time_horizon_days, pull current data, resolve
    get_open_trades() → pd.DataFrame
    get_trade_history(symbol?, category?, start?, end?) → pd.DataFrame
    get_performance_summary(category?, symbol?) → dict
        # Total trades, win rate, avg P&L, breakdown by category/direction/symbol
    get_category_performance() → pd.DataFrame
```

### 3C. Trade Journal CLI

**New file**: `src/volsurf/cli/commands/journal.py`

| Command | Description |
|---------|-------------|
| `volsurf journal new SYMBOL --thesis "..." --category skew_trade --direction long_skew` | Log a new trade |
| `volsurf journal resolve TRADE_ID --outcome win --notes "..."` | Manually resolve |
| `volsurf journal auto-resolve` | Auto-resolve expired trades |
| `volsurf journal open` | Show open trades |
| `volsurf journal history --last 20` | Show recent trades |
| `volsurf journal stats` | Performance summary |

### 3D. Trade Journal Web Component

**New file**: `src/volsurf/web/components/trade_journal.py`

```
render_trade_journal(symbols) → main component with tabs:
    Tab 1: New Trade Entry (form)
        - Symbol dropdown, category/direction selectboxes
        - Thesis text area, structure text input
        - Auto-fills current metrics from surface_metrics_history
    Tab 2: Open Trades (table with resolve buttons)
    Tab 3: Trade History (filterable table)
    Tab 4: Performance Analytics
        - Win rate by category (bar chart)
        - Cumulative P&L over time (line chart)
        - Monthly summary table
```

**Modify `src/volsurf/web/app.py`** — wire up the Trade Journal page.

---

## Phase 4: Event Vol Analytics (Post-MVP)

Explicitly out of MVP scope. Requires earnings calendar data source integration (Theta Data `hist_earnings` endpoint or supplemental source — unverified dependency).

### Planned deliverables (deferred):
- `earnings_calendar` and `event_vol_metrics` tables
- `src/volsurf/ingestion/earnings_client.py`
- `src/volsurf/analytics/event_vol.py` — implied move, straddle pricing, earnings vol premium, post-event crush
- `src/volsurf/web/components/event_vol.py`
- `src/volsurf/cli/commands/events.py`

### Phase 5: Cross-Module Integration (Post-MVP)

- Click surface anomaly in watchlist → pre-populate trade journal entry
- Trade resolution links back to surface state at entry time
- Weekly digest CLI command

---

## File Inventory

### New Files (6)
| File | Phase |
|------|-------|
| `src/volsurf/cli/commands/batch.py` | 1A |
| `src/volsurf/analytics/surface_metrics_history.py` | 1B |
| `src/volsurf/web/components/watchlist.py` | 2A |
| `src/volsurf/analytics/trade_journal.py` | 3B |
| `src/volsurf/cli/commands/journal.py` | 3C |
| `src/volsurf/web/components/trade_journal.py` | 3D |

### Modified Files (8)
| File | Phase | Change |
|------|-------|--------|
| `src/volsurf/database/schema.py` | 1B, 3A | Add `surface_metrics_history` + `trade_journal` DDL, sequences, indexes |
| `src/volsurf/config/settings.py` | 1A | Add `watchlist` field |
| `src/volsurf/cli/main.py` | 1A, 3C | Register `batch` and `journal` sub-apps |
| `src/volsurf/analytics/__init__.py` | 1B, 3B | Export new classes |
| `src/volsurf/web/app.py` | 2B, 2D | Add Watchlist + Trade Journal pages, day-over-day deltas, nav refactor |
| `src/volsurf/web/components/term_structure.py` | 2C | Add percentile bands |
| `src/volsurf/web/components/vrp_analysis.py` | 2C | Add percentile context |
| `scripts/daily_update.sh` | 1A | Update to use `batch` commands |

### Deleted Files (1)
| File | Reason |
|------|--------|
| `brief.txt` | Superseded by PROJECT_SPEC.md |

---

## Design Rationale

**Why a denormalized `surface_metrics_history` table?**
The heat map needs one row per symbol with interpolated metrics at standard tenors. Computing this from `fitted_surfaces` (one row per expiration) requires grouping + interpolation per symbol on every page load. Pre-computing makes the heat map query a trivial `SELECT WHERE quote_date = ? AND symbol IN (...)`.

**Why compute percentiles in SQL?**
DuckDB `PERCENT_RANK() OVER` processes everything in a single pass. Pulling all history into Python to compute with NumPy would be slower and use more memory for 11 symbols x 252 days.

**Why pre-compute `_1d_chg` columns?**
Avoids self-join or `LAG()` window function on the read path. Trivial to compute during backfill, makes heat map rendering fast.

**Why include trade journal in MVP?**
Without the journal, it's a read-only dashboard. The journal is what makes it a training tool with a feedback loop. Implementation is straightforward CRUD — no new data sources.

**Why exclude event vol from MVP?**
Requires earnings calendar data — an unverified integration point. The surface percentile system already flags pre-earnings vol elevation as an extreme reading, providing partial coverage.

**Why keep Streamlit?**
Already in place with 5 working pages. Sufficient for single-user personal tool. Fastest path to adding new features. Migration to Dash/React would delay the MVP for minimal benefit at this stage.
