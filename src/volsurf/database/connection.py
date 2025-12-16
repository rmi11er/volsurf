"""DuckDB connection management."""

from pathlib import Path
from typing import Optional

import duckdb
from duckdb import DuckDBPyConnection

from volsurf.config.settings import get_settings

# Global connection instance
_connection: Optional[DuckDBPyConnection] = None


def get_connection(path: Optional[Path] = None) -> DuckDBPyConnection:
    """
    Get or create the global DuckDB connection.

    Args:
        path: Optional path to database file. If None, uses settings.

    Returns:
        DuckDB connection instance.
    """
    global _connection

    if _connection is not None:
        return _connection

    settings = get_settings()
    db_path = path or settings.duckdb_path

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _connection = duckdb.connect(str(db_path))
    return _connection


def close_connection() -> None:
    """Close the global connection if open."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def get_memory_connection() -> DuckDBPyConnection:
    """Get an in-memory connection (useful for testing)."""
    return duckdb.connect(":memory:")
