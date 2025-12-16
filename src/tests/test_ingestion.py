"""Tests for data ingestion pipeline."""

from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from volsurf.config.settings import Settings
from volsurf.database.schema import init_schema
from volsurf.ingestion.theta_client import ThetaDataClient
from volsurf.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def mock_settings() -> Settings:
    """Create settings that use mock data."""
    return Settings(
        theta_api_key=None,  # Force mock data
        min_open_interest=10,
        max_bid_ask_spread_pct=0.30,
    )


@pytest.fixture
def memory_db() -> duckdb.DuckDBPyConnection:
    """Create an in-memory database with schema initialized."""
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


class TestThetaDataClient:
    """Tests for Theta Data API client."""

    @pytest.mark.asyncio
    async def test_uses_mock_when_no_api_key(self, mock_settings):
        """Client should use mock data when no API key configured."""
        client = ThetaDataClient(mock_settings)
        assert client.use_mock is True
        await client.close()

    @pytest.mark.asyncio
    async def test_get_options_chain_returns_dataframe(self, mock_settings):
        """get_options_chain should return a Polars DataFrame."""
        client = ThetaDataClient(mock_settings)
        try:
            df = await client.get_options_chain("SPY", date(2024, 12, 13))

            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0
            assert "symbol" in df.columns
            assert "strike" in df.columns
            assert "option_type" in df.columns
            assert "implied_volatility" in df.columns
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_options_chain_has_correct_columns(self, mock_settings):
        """Options chain should have all expected columns."""
        client = ThetaDataClient(mock_settings)
        try:
            df = await client.get_options_chain("SPY", date(2024, 12, 13))

            expected_columns = {
                "symbol", "quote_date", "expiration_date", "strike", "option_type",
                "bid", "ask", "mid", "last", "volume", "open_interest",
                "delta", "gamma", "theta", "vega", "rho",
                "implied_volatility", "underlying_price",
            }

            assert expected_columns.issubset(set(df.columns))
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_underlying_prices(self, mock_settings):
        """get_underlying_prices should return OHLCV data."""
        client = ThetaDataClient(mock_settings)
        try:
            start = date(2024, 12, 9)
            end = date(2024, 12, 13)
            df = await client.get_underlying_prices("SPY", start, end)

            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_underlying_prices_skips_weekends(self, mock_settings):
        """Underlying prices should not include weekend dates."""
        client = ThetaDataClient(mock_settings)
        try:
            # Dec 14-15, 2024 are Saturday-Sunday
            start = date(2024, 12, 13)  # Friday
            end = date(2024, 12, 16)  # Monday
            df = await client.get_underlying_prices("SPY", start, end)

            dates = df["date"].to_list()
            for d in dates:
                assert d.weekday() < 5, f"Weekend date found: {d}"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_expiration_filter(self, mock_settings):
        """Options chain should respect expiration filters."""
        client = ThetaDataClient(mock_settings)
        try:
            quote_date = date(2024, 12, 13)
            exp_min = quote_date + timedelta(days=30)
            exp_max = quote_date + timedelta(days=90)

            df = await client.get_options_chain(
                "SPY", quote_date, exp_min=exp_min, exp_max=exp_max
            )

            if len(df) > 0:
                expirations = df["expiration_date"].unique().to_list()
                for exp in expirations:
                    assert exp >= exp_min, f"Expiration {exp} before min {exp_min}"
                    assert exp <= exp_max, f"Expiration {exp} after max {exp_max}"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_historical_batch_yields_dates(self, mock_settings):
        """Historical batch should yield data for each trading day."""
        client = ThetaDataClient(mock_settings)
        try:
            start = date(2024, 12, 9)
            end = date(2024, 12, 13)

            dates_seen = []
            async for quote_date, df in client.get_historical_chain_batch("SPY", start, end):
                dates_seen.append(quote_date)
                assert isinstance(df, pl.DataFrame)

            # Should have 5 trading days (Mon-Fri)
            assert len(dates_seen) == 5
        finally:
            await client.close()


class TestIngestionPipeline:
    """Tests for the ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_daily_ingestion_inserts_data(self, mock_settings, memory_db, monkeypatch):
        """Daily ingestion should insert data into database."""
        # Monkeypatch to use our in-memory db
        import volsurf.ingestion.pipeline as pipeline_module
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        records = await pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        assert records > 0

        # Verify data in database
        result = memory_db.execute(
            "SELECT COUNT(*) FROM raw_options_chains WHERE symbol = 'SPY'"
        ).fetchone()
        assert result[0] > 0

    @pytest.mark.asyncio
    async def test_daily_ingestion_skips_existing(self, mock_settings, memory_db, monkeypatch):
        """Should skip ingestion if data already exists."""
        import volsurf.ingestion.pipeline as pipeline_module
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)

        # First ingestion
        records1 = await pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))
        assert records1 > 0

        # Second ingestion should skip
        records2 = await pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))
        assert records2 == 0

    @pytest.mark.asyncio
    async def test_daily_ingestion_applies_filters(self, mock_settings, memory_db, monkeypatch):
        """Ingested data should have is_liquid column."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        await pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        # Check that is_liquid column exists and has values
        result = memory_db.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_liquid THEN 1 ELSE 0 END) as liquid
            FROM raw_options_chains
        """).fetchone()

        assert result[0] > 0  # Has data
        assert result[1] > 0  # Some are liquid

    @pytest.mark.asyncio
    async def test_backfill_processes_multiple_days(self, mock_settings, memory_db, monkeypatch):
        """Backfill should process multiple trading days."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        stats = await pipeline.backfill_historical(
            "SPY",
            date(2024, 12, 9),
            date(2024, 12, 13),
        )

        assert stats["days_processed"] == 5
        assert stats["records_inserted"] > 0

        # Verify multiple dates in database
        result = memory_db.execute(
            "SELECT COUNT(DISTINCT quote_date) FROM raw_options_chains"
        ).fetchone()
        assert result[0] == 5

    @pytest.mark.asyncio
    async def test_inserts_underlying_prices(self, mock_settings, memory_db, monkeypatch):
        """Should insert underlying price data."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        await pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        result = memory_db.execute(
            "SELECT COUNT(*) FROM underlying_prices WHERE symbol = 'SPY'"
        ).fetchone()
        assert result[0] > 0
