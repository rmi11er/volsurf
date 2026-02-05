"""Macro data ingestion via yfinance → underlying_prices table."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from loguru import logger

from volsurf.config.settings import get_settings
from volsurf.database.connection import get_connection


class MacroClient:
    """Fetches macro symbol OHLCV data from yfinance and stores in underlying_prices."""

    def fetch_symbol(
        self,
        display_name: str,
        yf_ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch OHLCV for a single symbol from yfinance.

        Returns a Polars DataFrame matching the underlying_prices schema.
        """
        import yfinance as yf

        # yfinance end date is exclusive, add 1 day
        ticker = yf.Ticker(yf_ticker)
        df_pd = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )

        if df_pd.empty:
            logger.warning(f"No data returned for {display_name} ({yf_ticker})")
            return pl.DataFrame(
                schema={"symbol": pl.Utf8, "date": pl.Date, "open": pl.Float64,
                        "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                        "volume": pl.Int64}
            )

        # Convert to Polars
        df_pd = df_pd.reset_index()
        df = pl.from_pandas(df_pd)

        # Normalize column names (yfinance uses "Date", "Open", etc.)
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(col_map)

        # Ensure date column is Date type (yfinance returns datetime with tz)
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        # Select and add symbol
        available_cols = set(df.columns)
        select_exprs = [pl.lit(display_name).alias("symbol")]
        for col in ["date", "open", "high", "low", "close"]:
            if col in available_cols:
                select_exprs.append(pl.col(col).cast(pl.Float64) if col != "date" else pl.col(col))
        if "volume" in available_cols:
            select_exprs.append(pl.col("volume").cast(pl.Int64))
        else:
            select_exprs.append(pl.lit(0).alias("volume"))

        df = df.select(select_exprs)

        logger.info(f"Fetched {len(df)} rows for {display_name} ({yf_ticker})")
        return df

    def _upsert_prices(self, df: pl.DataFrame) -> int:
        """Upsert a Polars DataFrame into underlying_prices. Returns row count."""
        if df.is_empty():
            return 0

        conn = get_connection()
        # Use DuckDB's INSERT OR REPLACE via temp table approach
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_macro AS SELECT * FROM underlying_prices LIMIT 0")
        conn.execute("DELETE FROM _tmp_macro")

        # Get next price_id
        max_id = conn.execute("SELECT COALESCE(MAX(price_id), 0) FROM underlying_prices").fetchone()[0]

        # Add price_id column
        df_with_id = df.with_row_index("price_id", offset=max_id + 1).with_columns(
            pl.col("price_id").cast(pl.Int64)
        )

        # Insert into temp table then upsert
        conn.register("_macro_upload", df_with_id.to_arrow())
        conn.execute("INSERT INTO _tmp_macro SELECT * FROM _macro_upload")
        conn.unregister("_macro_upload")

        # Upsert: delete existing rows for same symbol+date, then insert
        conn.execute("""
            DELETE FROM underlying_prices
            WHERE (symbol, date) IN (SELECT symbol, date FROM _tmp_macro)
        """)
        inserted = conn.execute("""
            INSERT INTO underlying_prices
            SELECT * FROM _tmp_macro
        """).fetchone()

        conn.execute("DROP TABLE IF EXISTS _tmp_macro")

        row_count = len(df)
        logger.info(f"Upserted {row_count} rows into underlying_prices")
        return row_count

    def backfill_symbol(
        self,
        display_name: str,
        yf_ticker: str,
        start: date,
        end: date,
    ) -> int:
        """Fetch and upsert a single macro symbol. Returns row count."""
        df = self.fetch_symbol(display_name, yf_ticker, start, end)
        return self._upsert_prices(df)

    def backfill_all(
        self,
        start: date,
        end: date,
        symbols: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """Backfill all macro symbols. Returns {display_name: row_count}."""
        if symbols is None:
            symbols = get_settings().macro_symbols

        results: dict[str, int] = {}
        for display_name, yf_ticker in symbols.items():
            try:
                count = self.backfill_symbol(display_name, yf_ticker, start, end)
                results[display_name] = count
            except Exception as e:
                logger.error(f"Failed to backfill {display_name}: {e}")
                results[display_name] = -1

        return results
