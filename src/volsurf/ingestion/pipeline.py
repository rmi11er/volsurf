"""Data ingestion pipeline orchestration."""

from datetime import date
from typing import Optional

import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings
from volsurf.database.connection import get_connection
from volsurf.database.schema import init_schema
from volsurf.ingestion.filters import apply_liquidity_filters, validate_data_quality
from volsurf.ingestion.theta_client import ThetaDataClient


class IngestionPipeline:
    """
    Orchestrates daily data ingestion and processing.

    Handles:
    - Fetching options chain data from API (or mock)
    - Applying liquidity filters
    - Storing in DuckDB database
    - Basic data validation
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = ThetaDataClient(self.settings)

    async def run_daily_ingestion(
        self,
        symbol: str,
        target_date: date,
        skip_existing: bool = True,
    ) -> int:
        """
        Run daily data ingestion for a symbol.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            target_date: Date to ingest
            skip_existing: If True, skip if data already exists for this date

        Returns:
            Number of records inserted
        """
        conn = get_connection()

        # Ensure schema exists
        init_schema(conn)

        # Check if data already exists
        if skip_existing:
            existing = conn.execute(
                "SELECT COUNT(*) FROM raw_options_chains WHERE symbol = ? AND quote_date = ?",
                [symbol, target_date],
            ).fetchone()
            if existing and existing[0] > 0:
                logger.info(f"Data already exists for {symbol} on {target_date}, skipping")
                return 0

        logger.info(f"Ingesting data for {symbol} on {target_date}")

        # Fetch options chain
        chain_df = await self.client.get_options_chain(symbol, target_date)

        if chain_df.is_empty():
            logger.warning(f"No options data returned for {symbol} on {target_date}")
            return 0

        # Get underlying price for liquidity calculations
        underlying_price = chain_df.select("underlying_price").to_series()[0]

        # Apply liquidity filters
        chain_df = apply_liquidity_filters(chain_df, underlying_price, self.settings)

        # Validate data quality
        quality_stats = validate_data_quality(chain_df)
        logger.debug(f"Data quality: {quality_stats}")

        # Insert options chain data
        records_inserted = await self._insert_options_chain(conn, chain_df)

        # Fetch and insert underlying prices
        prices_df = await self.client.get_underlying_prices(symbol, target_date, target_date)
        if not prices_df.is_empty():
            await self._insert_underlying_prices(conn, prices_df)

        logger.info(f"Inserted {records_inserted} options records for {symbol} on {target_date}")

        return records_inserted

    async def backfill_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """
        Backfill historical data for a symbol.

        Args:
            symbol: Underlying symbol
            start_date: Start of date range
            end_date: End of date range
            skip_existing: If True, skip dates with existing data

        Returns:
            Dictionary with statistics:
            {days_processed, records_inserted, days_skipped, days_failed}
        """
        conn = get_connection()
        init_schema(conn)

        stats = {
            "days_processed": 0,
            "records_inserted": 0,
            "days_skipped": 0,
            "days_failed": 0,
        }

        logger.info(f"Starting backfill for {symbol} from {start_date} to {end_date}")

        # Get existing dates to potentially skip
        existing_dates = set()
        if skip_existing:
            result = conn.execute(
                "SELECT DISTINCT quote_date FROM raw_options_chains WHERE symbol = ?",
                [symbol],
            ).fetchall()
            existing_dates = {row[0] for row in result}

        async for quote_date, chain_df in self.client.get_historical_chain_batch(
            symbol, start_date, end_date
        ):
            if quote_date in existing_dates:
                logger.debug(f"Skipping {quote_date} - data exists")
                stats["days_skipped"] += 1
                continue

            try:
                if chain_df.is_empty():
                    logger.warning(f"No data for {quote_date}")
                    stats["days_failed"] += 1
                    continue

                # Get underlying price
                underlying_price = chain_df.select("underlying_price").to_series()[0]

                # Apply liquidity filters
                chain_df = apply_liquidity_filters(chain_df, underlying_price, self.settings)

                # Insert data
                records = await self._insert_options_chain(conn, chain_df)
                stats["records_inserted"] += records
                stats["days_processed"] += 1

                if stats["days_processed"] % 10 == 0:
                    logger.info(f"Processed {stats['days_processed']} days, {stats['records_inserted']} records")

            except Exception as e:
                logger.error(f"Failed to process {quote_date}: {e}")
                stats["days_failed"] += 1

        # Backfill underlying prices
        logger.info("Backfilling underlying prices...")
        prices_df = await self.client.get_underlying_prices(symbol, start_date, end_date)
        if not prices_df.is_empty():
            await self._insert_underlying_prices(conn, prices_df)

        logger.info(
            f"Backfill complete: {stats['days_processed']} days, "
            f"{stats['records_inserted']} records, "
            f"{stats['days_skipped']} skipped, "
            f"{stats['days_failed']} failed"
        )

        return stats

    async def _insert_options_chain(
        self,
        conn,
        df: pl.DataFrame,
    ) -> int:
        """Insert options chain data into database."""
        if df.is_empty():
            return 0

        # Prepare data for insertion
        # Generate chain_ids using sequence
        num_rows = len(df)
        start_id = conn.execute(
            f"SELECT nextval('seq_chain_id') FROM generate_series(1, {num_rows})"
        ).fetchall()
        chain_ids = [row[0] for row in start_id]

        # Add chain_id to dataframe
        df = df.with_columns(pl.Series("chain_id", chain_ids))

        # Select columns in correct order for insertion
        columns = [
            "chain_id", "symbol", "quote_date", "expiration_date", "strike", "option_type",
            "bid", "ask", "mid", "last", "volume", "open_interest",
            "delta", "gamma", "theta", "vega", "rho",
            "implied_volatility", "underlying_price", "is_liquid",
        ]

        # Filter to existing columns
        available_cols = [c for c in columns if c in df.columns]
        insert_df = df.select(available_cols)

        # Use DuckDB's DataFrame insertion
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.register("temp_options", insert_df.to_arrow())
            conn.execute(f"""
                INSERT INTO raw_options_chains ({', '.join(available_cols)})
                SELECT * FROM temp_options
            """)
            conn.execute("COMMIT")
            conn.unregister("temp_options")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to insert options data: {e}")
            raise

        return num_rows

    async def _insert_underlying_prices(
        self,
        conn,
        df: pl.DataFrame,
    ) -> int:
        """Insert underlying price data into database."""
        if df.is_empty():
            return 0

        # Generate price_ids
        num_rows = len(df)
        start_id = conn.execute(
            f"SELECT nextval('seq_price_id') FROM generate_series(1, {num_rows})"
        ).fetchall()
        price_ids = [row[0] for row in start_id]

        df = df.with_columns(pl.Series("price_id", price_ids))

        columns = ["price_id", "symbol", "date", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in columns if c in df.columns]
        insert_df = df.select(available_cols)

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.register("temp_prices", insert_df.to_arrow())
            # Delete existing records for these dates, then insert
            # This is simpler than ON CONFLICT with multiple constraints
            dates_list = df.select("date").to_series().to_list()
            symbols_list = df.select("symbol").to_series().to_list()
            for symbol, dt in zip(symbols_list, dates_list):
                conn.execute(
                    "DELETE FROM underlying_prices WHERE symbol = ? AND date = ?",
                    [symbol, dt]
                )
            conn.execute(f"""
                INSERT INTO underlying_prices ({', '.join(available_cols)})
                SELECT * FROM temp_prices
            """)
            conn.execute("COMMIT")
            conn.unregister("temp_prices")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to insert price data: {e}")
            raise

        return num_rows
