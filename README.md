# volsurf

Crypto options data lake with volatility surface fitting.

## Overview

A data lake system for options chain data with fitted volatility surfaces, enabling sophisticated volatility trading analysis. The system ingests historical and ongoing crypto options data, fits SVI volatility surface models, and provides both programmatic and visual interfaces for analysis.

**Current First Pass:** BTC on Deribit, end-of-day snapshots, SVI surface fitting

## Quick Start

```bash
# Install dependencies
uv sync

# Initialize the database
uv run volsurf init-db

# Run daily ingestion for BTC
uv run volsurf ingest daily BTC

# Fit surfaces
uv run volsurf fit surfaces BTC

# Run analytics
uv run volsurf analyze realized-vol BTC
uv run volsurf analyze vrp BTC

# Launch the dashboard
uv run volsurf web
```

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Key settings:
- `DATA_PROVIDER`: `deribit` for crypto or `theta` for the old equities path
- `DEFAULT_SYMBOL`: Default underlying symbol, now `BTC`
- `WATCHLIST`: Comma-separated crypto watchlist, e.g. `BTC,ETH,SOL`
- `DUCKDB_PATH`: Path to the database file

## Project Structure

```
volsurf/
├── src/volsurf/          # Main package
│   ├── config/           # Configuration management
│   ├── database/         # DuckDB schema and connection
│   ├── ingestion/        # Data ingestion pipeline
│   ├── models/           # Volatility models (SVI)
│   ├── fitting/          # Surface fitting pipeline
│   ├── analytics/        # Metrics (realized vol, VRP)
│   ├── cli/              # Command-line interface
│   └── web/              # Web dashboard
├── data/                 # Database and data files
├── notebooks/            # Jupyter analysis notebooks
└── tests/                # Test suite
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src

# Linting
uv run ruff check src
```

## License

MIT
