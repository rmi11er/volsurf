"""Tests for data ingestion pipeline."""

from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from volsurf.config.settings import Settings
from volsurf.database.schema import init_schema
from volsurf.ingestion.deribit_client import DeribitClient
from volsurf.ingestion.theta_client import ThetaDataClient
from volsurf.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def mock_settings() -> Settings:
    """Create settings that use mock data."""
    # Explicitly set theta_username=None to override .env and force mock mode
    return Settings(
        data_provider="theta",
        theta_username=None,
        min_open_interest=10,
        max_bid_ask_spread_pct=0.30,
    )


@pytest.fixture
def deribit_settings() -> Settings:
    """Create settings that use the Deribit public API."""
    return Settings(
        data_provider="deribit",
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

    def test_uses_mock_when_no_api_key(self, mock_settings):
        """Client should use mock data when no API key configured."""
        client = ThetaDataClient(mock_settings)
        assert client.use_mock is True
        client.close()

    def test_get_options_chain_returns_dataframe(self, mock_settings):
        """get_options_chain should return a Polars DataFrame."""
        client = ThetaDataClient(mock_settings)
        try:
            df = client.get_options_chain("SPY", date(2024, 12, 13))

            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0
            assert "symbol" in df.columns
            assert "strike" in df.columns
            assert "option_type" in df.columns
        finally:
            client.close()

    def test_options_chain_has_core_columns(self, mock_settings):
        """Options chain should have core required columns."""
        client = ThetaDataClient(mock_settings)
        try:
            df = client.get_options_chain("SPY", date(2024, 12, 13))

            # Core columns that should always be present
            core_columns = {
                "symbol", "quote_date", "expiration_date", "strike", "option_type",
                "bid", "ask", "mid", "volume", "open_interest",
            }

            assert core_columns.issubset(set(df.columns))
        finally:
            client.close()

    def test_get_underlying_prices(self, mock_settings):
        """get_underlying_eod should return OHLCV data."""
        client = ThetaDataClient(mock_settings)
        try:
            start = date(2024, 12, 9)
            end = date(2024, 12, 13)
            df = client.get_underlying_eod("SPY", start, end)

            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
        finally:
            client.close()

    def test_underlying_prices_skips_weekends(self, mock_settings):
        """Underlying prices should not include weekend dates."""
        client = ThetaDataClient(mock_settings)
        try:
            # Dec 14-15, 2024 are Saturday-Sunday
            start = date(2024, 12, 13)  # Friday
            end = date(2024, 12, 16)  # Monday
            df = client.get_underlying_eod("SPY", start, end)

            dates = df["date"].to_list()
            for d in dates:
                assert d.weekday() < 5, f"Weekend date found: {d}"
        finally:
            client.close()

    def test_get_expirations_returns_dates(self, mock_settings):
        """get_expirations should return a list of dates."""
        client = ThetaDataClient(mock_settings)
        try:
            expirations = client.get_expirations("SPY")

            assert isinstance(expirations, list)
            assert len(expirations) > 0
            assert all(isinstance(d, date) for d in expirations)
        finally:
            client.close()

    def test_get_strikes_returns_floats(self, mock_settings):
        """get_strikes should return a list of strike prices."""
        client = ThetaDataClient(mock_settings)
        try:
            # First get an expiration
            expirations = client.get_expirations("SPY")
            if expirations:
                strikes = client.get_strikes("SPY", expirations[0])

                assert isinstance(strikes, list)
                assert len(strikes) > 0
                assert all(isinstance(s, (int, float)) for s in strikes)
        finally:
            client.close()


class TestIngestionPipeline:
    """Tests for the ingestion pipeline."""

    def test_daily_ingestion_inserts_data(self, mock_settings, memory_db, monkeypatch):
        """Daily ingestion should insert data into database."""
        # Monkeypatch to use our in-memory db
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        records = pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        assert records > 0

        # Verify data in database
        result = memory_db.execute(
            "SELECT COUNT(*) FROM raw_options_chains WHERE symbol = 'SPY'"
        ).fetchone()
        assert result[0] > 0

    def test_daily_ingestion_skips_existing(self, mock_settings, memory_db, monkeypatch):
        """Should skip ingestion if data already exists."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)

        # First ingestion
        records1 = pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))
        assert records1 > 0

        # Second ingestion should skip
        records2 = pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))
        assert records2 == 0

    def test_daily_ingestion_applies_filters(self, mock_settings, memory_db, monkeypatch):
        """Ingested data should have is_liquid column."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        # Check that is_liquid column exists and has values
        result = memory_db.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_liquid THEN 1 ELSE 0 END) as liquid
            FROM raw_options_chains
        """).fetchone()

        assert result[0] > 0  # Has data
        assert result[1] > 0  # Some are liquid

    def test_backfill_processes_multiple_days(self, mock_settings, memory_db, monkeypatch):
        """Backfill should process multiple trading days."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        stats = pipeline.backfill_historical(
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

    def test_inserts_underlying_prices(self, mock_settings, memory_db, monkeypatch):
        """Should insert underlying price data."""
        import volsurf.database.connection as conn_module

        monkeypatch.setattr(conn_module, "_connection", memory_db)

        pipeline = IngestionPipeline(mock_settings)
        pipeline.run_daily_ingestion("SPY", date(2024, 12, 13))

        result = memory_db.execute(
            "SELECT COUNT(*) FROM underlying_prices WHERE symbol = 'SPY'"
        ).fetchone()
        assert result[0] > 0


class TestDeribitClient:
    """Tests for the Deribit crypto client."""

    def test_non_today_snapshot_returns_empty(self, deribit_settings):
        """Historical option snapshots should return empty until backfill support exists."""
        client = DeribitClient(deribit_settings)
        try:
            df = client.get_options_chain("BTC", date(2024, 12, 13))
            assert isinstance(df, pl.DataFrame)
            assert df.is_empty()
        finally:
            client.close()

    def test_options_chain_maps_deribit_snapshot(self, deribit_settings, monkeypatch):
        """Current Deribit snapshot should map into the generic options schema."""
        client = DeribitClient(deribit_settings)
        today = client._today_utc()

        def fake_get_instruments(symbol: str):
            assert symbol == "BTC"
            return [
                {
                    "instrument_name": "BTC-30DEC26-100000-C",
                    "expiration_timestamp": 1798588800000,
                    "strike": 100000,
                    "option_type": "call",
                    "kind": "option",
                }
            ]

        def fake_request(method: str, params: dict[str, object]):
            if method == "public/get_book_summary_by_currency":
                return [
                    {
                        "instrument_name": "BTC-30DEC26-100000-C",
                        "underlying_price": 100000.0,
                        "bid_price": 0.05,
                        "ask_price": 0.06,
                        "last": 0.055,
                        "mark_price": 0.056,
                        "mark_iv": 65.0,
                        "volume": 4.2,
                    }
                ]
            raise AssertionError(f"Unexpected method: {method}")

        monkeypatch.setattr(client, "get_instruments", fake_get_instruments)
        monkeypatch.setattr(client, "_request", fake_request)

        try:
            df = client.get_options_chain("BTC", today)
            assert len(df) == 1
            row = df.to_dicts()[0]
            assert row["symbol"] == "BTC"
            assert row["option_type"] == "CALL"
            assert row["strike"] == 100000.0
            assert row["bid"] == 5000.0
            assert row["ask"] == 6000.0
            assert row["mid"] == 5500.0
            assert row["implied_volatility"] == 0.65
        finally:
            client.close()

    def test_pipeline_uses_deribit_client(self, deribit_settings):
        """Ingestion pipeline should select the Deribit client when configured."""
        pipeline = IngestionPipeline(deribit_settings)
        try:
            assert isinstance(pipeline.client, DeribitClient)
        finally:
            pipeline.client.close()
