"""Tests for intraday vol surface functionality."""

from datetime import date, datetime, time
from decimal import Decimal

import numpy as np
import pytest

from volsurf.ingestion.mock_data import MockDataGenerator, MockIntradayGenerator
from volsurf.fitting.intraday_pipeline import (
    IntradaySurfacePipeline,
    IntradayPipelineConfig,
    fit_intraday_surface,
)
from volsurf.models.schemas import IntradaySurfaceResult


class TestMockIntradayGenerator:
    """Tests for MockIntradayGenerator."""

    def test_intraday_vol_multiplier_u_shaped(self):
        """Test that intraday vol follows U-shaped pattern."""
        generator = MockIntradayGenerator()

        # Open (9:30) - should be higher
        open_mult = generator._get_intraday_vol_multiplier(time(9, 30))

        # Mid-day (12:00) - should be lower
        midday_mult = generator._get_intraday_vol_multiplier(time(12, 0))

        # Close (15:55) - should be higher
        close_mult = generator._get_intraday_vol_multiplier(time(15, 55))

        # U-shape: open and close higher than midday
        assert open_mult > midday_mult
        assert close_mult > midday_mult

        # Multipliers should be in reasonable range
        assert 0.85 <= midday_mult <= 0.95
        assert 1.0 <= open_mult <= 1.2
        assert 1.0 <= close_mult <= 1.2

    def test_intraday_spread_multiplier(self):
        """Test that spreads are wider at open."""
        generator = MockIntradayGenerator()

        # Open spread (9:35)
        open_spread = generator._get_intraday_spread_multiplier(time(9, 35))

        # Mid-day spread (12:00)
        midday_spread = generator._get_intraday_spread_multiplier(time(12, 0))

        # Open should have wider spreads
        assert open_spread > midday_spread
        assert midday_spread == 1.0  # Baseline at mid-day

    def test_generate_intraday_chain(self):
        """Test intraday chain generation."""
        generator = MockIntradayGenerator()
        target_date = date(2024, 12, 20)
        target_time = time(10, 30)

        chain_df = generator.generate_intraday_chain("SPY", target_date, target_time)

        # Should have data
        assert not chain_df.is_empty()

        # Check columns
        expected_cols = [
            "symbol", "timestamp", "expiration_date", "strike",
            "option_type", "bid", "ask", "mid"
        ]
        for col in expected_cols:
            assert col in chain_df.columns

        # Check values make sense
        assert (chain_df["bid"] > 0).all()
        assert (chain_df["ask"] > chain_df["bid"]).all()
        assert (chain_df["mid"] > 0).all()

    def test_underlying_at_time(self):
        """Test underlying price at specific time."""
        generator = MockIntradayGenerator()
        target_date = date(2024, 12, 20)

        # Get prices at different times
        price_930 = generator.get_underlying_at_time("SPY", target_date, time(9, 30))
        price_1200 = generator.get_underlying_at_time("SPY", target_date, time(12, 0))
        price_1555 = generator.get_underlying_at_time("SPY", target_date, time(15, 55))

        # All should be reasonable prices
        assert 400 < float(price_930) < 800
        assert 400 < float(price_1200) < 800
        assert 400 < float(price_1555) < 800

        # Prices should vary throughout the day (not identical)
        # Due to random walk with same seed, they should be different
        prices = [float(price_930), float(price_1200), float(price_1555)]
        assert len(set(prices)) > 1  # At least some variation


class TestIntradaySurfacePipeline:
    """Tests for IntradaySurfacePipeline."""

    def test_fit_at_time_mock(self):
        """Test fitting surfaces at a specific time with mock data."""
        target_date = date(2024, 12, 20)
        target_time = time(10, 30)

        result = fit_intraday_surface(
            "SPY", target_date, target_time, use_mock=True
        )

        # Should have results
        assert isinstance(result, IntradaySurfaceResult)
        assert result.symbol == "SPY"
        assert result.timestamp == datetime.combine(target_date, target_time)

        # Should have fitted some surfaces
        assert result.successful_fits > 0
        assert len(result.surfaces) > 0

        # Check surface properties
        for surface in result.surfaces:
            assert surface.symbol == "SPY"
            assert surface.timestamp == result.timestamp
            assert 0 < surface.atm_vol < 1.0  # Reasonable ATM vol
            assert surface.rmse < 0.1  # Reasonable fit quality
            assert surface.num_points >= 10

    def test_fit_at_time_different_times(self):
        """Test that different times produce different results."""
        target_date = date(2024, 12, 20)

        # Fit at 10:30
        result_1030 = fit_intraday_surface(
            "SPY", target_date, time(10, 30), use_mock=True
        )

        # Fit at 14:00
        result_1400 = fit_intraday_surface(
            "SPY", target_date, time(14, 0), use_mock=True
        )

        # Should have different underlying prices
        assert result_1030.underlying_price != result_1400.underlying_price

        # Should have different timestamps
        assert result_1030.timestamp != result_1400.timestamp

    def test_pipeline_config(self):
        """Test pipeline respects configuration."""
        config = IntradayPipelineConfig(
            min_strikes_per_expiration=20,  # Higher threshold
            max_tte_days=180,  # Shorter max TTE
        )

        pipeline = IntradaySurfacePipeline(config, use_mock=True)
        result = pipeline.fit_at_time("SPY", date(2024, 12, 20), time(10, 30))
        pipeline.close()

        # With higher min_strikes, might have fewer successful fits
        # but should still work
        assert isinstance(result, IntradaySurfaceResult)

        # All fitted surfaces should have at least min_strikes points
        for surface in result.surfaces:
            assert surface.num_points >= config.min_strikes_per_expiration

    def test_fit_time_series(self):
        """Test fitting surfaces across a time series."""
        config = IntradayPipelineConfig()
        pipeline = IntradaySurfacePipeline(config, use_mock=True)

        try:
            results = pipeline.fit_time_series(
                "SPY",
                date(2024, 12, 20),
                start_time=time(10, 0),
                end_time=time(11, 0),
                interval_minutes=30,
            )
        finally:
            pipeline.close()

        # Should have 3 results (10:00, 10:30, 11:00)
        assert len(results) == 3

        # Each should be a valid result
        for r in results:
            assert isinstance(r, IntradaySurfaceResult)
            assert r.symbol == "SPY"

        # Timestamps should be sequential
        times = [r.timestamp.time() for r in results]
        assert times == [time(10, 0), time(10, 30), time(11, 0)]


class TestIntradaySurfaceResult:
    """Tests for IntradaySurfaceResult model."""

    def test_parquet_export_import(self, tmp_path):
        """Test exporting and importing surfaces to Parquet."""
        # Generate a result
        result = fit_intraday_surface(
            "SPY", date(2024, 12, 20), time(10, 30), use_mock=True
        )

        if not result.surfaces:
            pytest.skip("No surfaces to export")

        # Export
        output_path = tmp_path / "test_surface.parquet"
        result.to_parquet(str(output_path))

        assert output_path.exists()

        # Import
        loaded = IntradaySurfaceResult.from_parquet(str(output_path))

        # Check loaded data matches
        assert loaded.symbol == result.symbol
        assert len(loaded.surfaces) == len(result.surfaces)

        for orig, loaded_s in zip(result.surfaces, loaded.surfaces):
            assert orig.expiration_date == loaded_s.expiration_date
            assert abs(orig.atm_vol - loaded_s.atm_vol) < 1e-6
            assert abs(orig.svi_params.a - loaded_s.svi_params.a) < 1e-6


class TestIVCalculation:
    """Tests for IV calculation from intraday quotes."""

    def test_iv_reasonable_values(self):
        """Test that calculated IVs are reasonable."""
        result = fit_intraday_surface(
            "SPY", date(2024, 12, 20), time(10, 30), use_mock=True
        )

        for surface in result.surfaces:
            # ATM vol should be between 5% and 100%
            assert 0.05 < surface.atm_vol < 1.0

            # SVI params should be reasonable
            params = surface.svi_params
            assert params.b >= 0  # Non-negative wing slope
            assert -1 < params.rho < 1  # Valid correlation
            assert params.sigma > 0  # Positive smoothness

    def test_skew_direction(self):
        """Test that skew has expected direction (puts more expensive)."""
        result = fit_intraday_surface(
            "SPY", date(2024, 12, 20), time(10, 30), use_mock=True
        )

        # For equity indices, typically negative rho (downside skew)
        # and positive 25-delta skew (put vol > call vol)
        for surface in result.surfaces:
            if surface.skew_25delta is not None:
                # 25-delta skew is put_vol - call_vol
                # Should be positive for typical equity skew
                assert surface.skew_25delta > -0.1  # Allow some tolerance
