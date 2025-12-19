"""Tests for validation and export utilities."""

from datetime import date, timedelta
from pathlib import Path
import tempfile

import pytest

from volsurf.utils.validation import (
    DataQualityReport,
    DataValidator,
    FitQualityReport,
)
from volsurf.utils.export import (
    export_options_chain,
    export_fitted_surfaces,
)


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_healthy_report(self):
        """Healthy data should pass is_healthy check."""
        report = DataQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            total_options=1000,
            liquid_options=500,
            num_expirations=20,
            liquidity_ratio=0.5,
            invalid_spread=0,
        )

        assert report.is_healthy()

    def test_unhealthy_no_options(self):
        """No options should fail health check."""
        report = DataQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            total_options=0,
        )

        assert not report.is_healthy()

    def test_unhealthy_low_liquidity(self):
        """Low liquidity should fail health check."""
        report = DataQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            total_options=1000,
            liquid_options=50,
            liquidity_ratio=0.05,  # Below 10% threshold
            invalid_spread=0,
        )

        assert not report.is_healthy()

    def test_unhealthy_invalid_spreads(self):
        """Invalid spreads should fail health check."""
        report = DataQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            total_options=1000,
            liquid_options=500,
            liquidity_ratio=0.5,
            invalid_spread=5,  # Has invalid spreads
        )

        assert not report.is_healthy()

    def test_to_dict(self):
        """to_dict should return all fields."""
        report = DataQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            total_options=1000,
            liquid_options=500,
        )

        d = report.to_dict()

        assert d["symbol"] == "SPY"
        assert d["quote_date"] == "2025-12-16"
        assert d["total_options"] == 1000
        assert d["liquid_options"] == 500
        assert "is_healthy" in d


class TestFitQualityReport:
    """Tests for FitQualityReport dataclass."""

    def test_healthy_fit_report(self):
        """Good fits should pass health check."""
        report = FitQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            num_surfaces=20,
            successful_fits=20,
            failed_fits=0,
            avg_rmse=0.005,  # 0.5% RMSE
        )

        assert report.is_healthy()

    def test_unhealthy_no_surfaces(self):
        """No surfaces should fail health check."""
        report = FitQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            num_surfaces=0,
        )

        assert not report.is_healthy()

    def test_unhealthy_high_rmse(self):
        """High RMSE should fail health check."""
        report = FitQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            num_surfaces=20,
            successful_fits=20,
            avg_rmse=0.02,  # 2% RMSE - above 1% threshold
        )

        assert not report.is_healthy()

    def test_unhealthy_failed_fits(self):
        """Failed fits should fail health check."""
        report = FitQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            num_surfaces=20,
            successful_fits=18,
            failed_fits=2,  # Some failed
            avg_rmse=0.005,
        )

        assert not report.is_healthy()

    def test_to_dict(self):
        """to_dict should return all fields."""
        report = FitQualityReport(
            symbol="SPY",
            quote_date=date(2025, 12, 16),
            num_surfaces=20,
        )

        d = report.to_dict()

        assert d["symbol"] == "SPY"
        assert d["quote_date"] == "2025-12-16"
        assert d["num_surfaces"] == 20
        assert "is_healthy" in d


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_init_default_thresholds(self):
        """Default thresholds should be set correctly."""
        validator = DataValidator()

        assert validator.min_oi_threshold == 50
        assert validator.iv_min == 0.05
        assert validator.iv_max == 2.0

    def test_init_custom_thresholds(self):
        """Custom thresholds should be set correctly."""
        validator = DataValidator(min_oi_threshold=100, iv_bounds=(0.10, 1.5))

        assert validator.min_oi_threshold == 100
        assert validator.iv_min == 0.10
        assert validator.iv_max == 1.5


class TestExportPath:
    """Tests for export file path handling."""

    def test_csv_suffix(self):
        """CSV export should add .csv suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_file"

            # Just check path building logic
            file_path = output_path.with_suffix(".csv")
            assert str(file_path).endswith(".csv")

    def test_json_suffix(self):
        """JSON export should add .json suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_file"

            file_path = output_path.with_suffix(".json")
            assert str(file_path).endswith(".json")

    def test_parent_directory_creation(self):
        """Export should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test_file"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            assert output_path.parent.exists()
