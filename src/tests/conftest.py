"""Pytest fixtures for volsurf tests."""

from datetime import date
from pathlib import Path
from typing import Generator

import duckdb
import pytest

from volsurf.config.settings import Settings
from volsurf.database.schema import init_schema


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with in-memory database."""
    return Settings(
        theta_api_key=None,  # Use mock data
        duckdb_path=Path(":memory:"),
        min_open_interest=10,  # Lower threshold for tests
        max_bid_ask_spread_pct=0.30,  # Higher for tests
    )


@pytest.fixture
def memory_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Create an in-memory DuckDB connection with schema initialized."""
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_date() -> date:
    """Sample date for testing."""
    return date(2024, 12, 13)


@pytest.fixture
def sample_symbol() -> str:
    """Sample symbol for testing."""
    return "SPY"
