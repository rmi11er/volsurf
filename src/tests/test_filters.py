"""Tests for liquidity filters."""

from datetime import date, timedelta

import polars as pl
import pytest

from volsurf.config.settings import Settings
from volsurf.ingestion.filters import (
    apply_liquidity_filters,
    filter_liquid_only,
    get_liquid_strikes_by_expiration,
    validate_data_quality,
)


@pytest.fixture
def sample_options_df() -> pl.DataFrame:
    """Create a sample options DataFrame for testing."""
    today = date(2024, 12, 13)
    exp_near = today + timedelta(days=30)
    exp_far = today + timedelta(days=180)

    return pl.DataFrame({
        "quote_date": [today] * 8,
        "expiration_date": [exp_near, exp_near, exp_near, exp_near, exp_far, exp_far, exp_far, exp_far],
        "strike": [580.0, 590.0, 600.0, 610.0, 550.0, 590.0, 630.0, 700.0],
        "option_type": ["PUT", "CALL", "CALL", "CALL", "PUT", "CALL", "CALL", "CALL"],
        "bid": [5.0, 8.0, 3.0, 0.5, 10.0, 15.0, 5.0, 0.01],
        "ask": [5.5, 8.5, 3.5, 1.0, 11.0, 16.0, 6.0, 0.50],  # Last one has wide spread
        "mid": [5.25, 8.25, 3.25, 0.75, 10.5, 15.5, 5.5, 0.255],
        "volume": [100, 500, 200, 50, 80, 300, 100, 5],
        "open_interest": [1000, 5000, 2000, 100, 800, 3000, 1000, 10],  # Last one low OI
        "implied_volatility": [0.20, 0.18, 0.19, 0.22, 0.25, 0.20, 0.21, 0.30],
        "delta": [-0.3, 0.5, 0.3, 0.1, -0.4, 0.5, 0.2, 0.01],
        "gamma": [0.02, 0.03, 0.02, 0.01, 0.015, 0.025, 0.015, 0.001],
        "theta": [-0.05, -0.06, -0.04, -0.02, -0.03, -0.04, -0.03, -0.01],
        "vega": [0.3, 0.4, 0.3, 0.1, 0.5, 0.6, 0.4, 0.05],
    })


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with specific filter thresholds."""
    return Settings(
        theta_api_key=None,
        min_open_interest=50,
        max_bid_ask_spread_pct=0.20,
        min_dte=7,
        max_dte=365,
        min_moneyness=0.8,
        max_moneyness=1.2,
    )


class TestApplyLiquidityFilters:
    """Tests for apply_liquidity_filters function."""

    def test_adds_is_liquid_column(self, sample_options_df, test_settings):
        """Should add is_liquid column to DataFrame."""
        underlying_price = 590.0
        result = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)

        assert "is_liquid" in result.columns
        assert result.schema["is_liquid"] == pl.Boolean

    def test_filters_by_open_interest(self, sample_options_df, test_settings):
        """Options with low OI should be marked illiquid."""
        underlying_price = 590.0
        result = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)

        # Last option has OI=10, below threshold of 50
        low_oi_row = result.filter(pl.col("strike") == 700.0)
        assert not low_oi_row["is_liquid"][0]

    def test_filters_by_spread(self, sample_options_df, test_settings):
        """Options with wide spread should be marked illiquid."""
        underlying_price = 590.0
        result = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)

        # The 700 strike has spread of (0.50 - 0.01) / 0.255 ≈ 192%
        wide_spread_row = result.filter(pl.col("strike") == 700.0)
        assert not wide_spread_row["is_liquid"][0]

    def test_filters_by_moneyness(self, sample_options_df, test_settings):
        """Options outside moneyness range should be marked illiquid."""
        underlying_price = 590.0
        result = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)

        # 700 strike: moneyness = 590/700 = 0.84, within 0.8-1.2
        # 550 strike: moneyness = 590/550 = 1.07, within 0.8-1.2
        # Both should pass moneyness but may fail other filters

    def test_liquid_options_pass_all_filters(self, sample_options_df, test_settings):
        """ATM options with good liquidity should pass."""
        underlying_price = 590.0
        result = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)

        # 590 strike should be liquid
        atm_row = result.filter(
            (pl.col("strike") == 590.0) & (pl.col("expiration_date") == result["expiration_date"][0])
        )
        assert atm_row["is_liquid"][0]

    def test_empty_dataframe(self, test_settings):
        """Empty DataFrame should return empty with is_liquid column."""
        empty_df = pl.DataFrame(schema={
            "quote_date": pl.Date,
            "expiration_date": pl.Date,
            "strike": pl.Float64,
            "bid": pl.Float64,
            "ask": pl.Float64,
            "mid": pl.Float64,
            "open_interest": pl.Int64,
        })

        result = apply_liquidity_filters(empty_df, 590.0, test_settings)
        assert "is_liquid" in result.columns
        assert len(result) == 0


class TestFilterLiquidOnly:
    """Tests for filter_liquid_only function."""

    def test_filters_to_liquid_only(self, sample_options_df, test_settings):
        """Should return only liquid options."""
        underlying_price = 590.0
        filtered = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)
        liquid_only = filter_liquid_only(filtered)

        assert all(liquid_only["is_liquid"])
        assert len(liquid_only) < len(filtered)

    def test_raises_without_is_liquid(self, sample_options_df):
        """Should raise if is_liquid column missing."""
        with pytest.raises(ValueError, match="is_liquid"):
            filter_liquid_only(sample_options_df)


class TestValidateDataQuality:
    """Tests for data quality validation."""

    def test_counts_total_rows(self, sample_options_df):
        """Should count total rows."""
        stats = validate_data_quality(sample_options_df)
        assert stats["total_rows"] == 8

    def test_detects_missing_iv(self):
        """Should detect missing implied volatility."""
        df = pl.DataFrame({
            "implied_volatility": [0.2, None, 0.18, None],
        })
        stats = validate_data_quality(df)
        assert stats["missing_iv"] == 2

    def test_detects_extreme_iv(self):
        """Should detect extreme IVs."""
        df = pl.DataFrame({
            "implied_volatility": [0.02, 0.20, 2.5, 0.18],  # First too low, third too high
        })
        stats = validate_data_quality(df)
        assert stats["iv_too_low"] == 1
        assert stats["iv_too_high"] == 1

    def test_detects_invalid_spread(self):
        """Should detect bid > ask."""
        df = pl.DataFrame({
            "bid": [5.0, 8.0, 6.0],
            "ask": [5.5, 7.0, 6.5],  # Second has bid > ask
        })
        stats = validate_data_quality(df)
        assert stats["invalid_spread"] == 1

    def test_empty_dataframe(self):
        """Empty DataFrame should return total_rows=0."""
        df = pl.DataFrame()
        stats = validate_data_quality(df)
        assert stats["total_rows"] == 0


class TestGetLiquidStrikesByExpiration:
    """Tests for liquid strikes summary by expiration."""

    def test_summarizes_by_expiration(self, sample_options_df, test_settings):
        """Should summarize liquid strikes per expiration."""
        underlying_price = 590.0
        filtered = apply_liquidity_filters(sample_options_df, underlying_price, test_settings)
        summary = get_liquid_strikes_by_expiration(filtered)

        assert "expiration_date" in summary.columns
        assert "total_liquid" in summary.columns
        assert "is_fittable" in summary.columns

    def test_raises_without_is_liquid(self, sample_options_df):
        """Should raise if is_liquid column missing."""
        with pytest.raises(ValueError, match="is_liquid"):
            get_liquid_strikes_by_expiration(sample_options_df)
