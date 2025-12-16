"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Theta Data API
    theta_api_key: Optional[str] = Field(default=None, description="Theta Data API key")
    theta_api_url: str = Field(
        default="https://api.thetadata.us/v2", description="Theta Data API base URL"
    )

    # Database
    duckdb_path: Path = Field(
        default=Path("data/volsurf.duckdb"), description="Path to DuckDB database file"
    )

    # Data Ingestion
    default_symbol: str = Field(default="SPY", description="Default symbol for ingestion")
    ingestion_start_date: str = Field(
        default="2020-01-01", description="Default start date for backfill"
    )

    # Liquidity Filters
    min_open_interest: int = Field(default=50, description="Minimum open interest threshold")
    max_bid_ask_spread_pct: float = Field(
        default=0.20, description="Maximum bid-ask spread as fraction of mid"
    )
    min_dte: int = Field(default=7, description="Minimum days to expiration")
    max_dte: int = Field(default=730, description="Maximum days to expiration")
    min_moneyness: float = Field(default=0.7, description="Minimum moneyness (S/K)")
    max_moneyness: float = Field(default=1.3, description="Maximum moneyness (S/K)")

    # Surface Fitting
    svi_max_iterations: int = Field(default=1000, description="Maximum SVI optimization iterations")
    svi_tolerance: float = Field(default=1e-8, description="SVI optimization tolerance")
    min_strikes_per_fit: int = Field(
        default=10, description="Minimum strikes required for surface fit"
    )
    trading_days_per_year: int = Field(
        default=252, description="Trading days per year for annualization"
    )
    vega_weight_reference_dte: int = Field(
        default=100, description="Reference DTE for vega weighting (sqrt(TTE/reference))"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(default=Path("logs/volsurf.log"), description="Log file path")

    # Web Dashboard
    streamlit_port: int = Field(default=8501, description="Streamlit dashboard port")

    @property
    def use_mock_data(self) -> bool:
        """Return True if no API key is configured (use mock data)."""
        return self.theta_api_key is None or self.theta_api_key == "your_api_key_here"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
