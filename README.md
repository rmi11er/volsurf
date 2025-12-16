# volsurf

Options data lake with volatility surface fitting.

## Overview

A data lake system for options chain data with fitted volatility surfaces, enabling sophisticated volatility trading analysis. The system ingests historical and ongoing options data, fits SVI volatility surface models, and provides both programmatic and visual interfaces for analysis.

**MVP Scope:** SPY options only, end-of-day updates, SVI surface fitting

## Quick Start

```bash
# Install dependencies
uv sync

# Initialize the database
uv run volsurf init-db

# Run daily ingestion (uses mock data if no API key configured)
uv run volsurf ingest daily SPY

# Backfill historical data
uv run volsurf ingest backfill SPY --start 2024-01-01 --end 2024-12-16

# Check data coverage
uv run volsurf ingest status
```

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Key settings:
- `THETA_API_KEY`: Your Theta Data API key (leave blank to use mock data)
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
