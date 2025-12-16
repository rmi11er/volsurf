"""Theta Data API client with mock data fallback."""

from datetime import date, timedelta
from typing import AsyncIterator, Optional

import httpx
import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings
from volsurf.ingestion.mock_data import MockDataGenerator
from volsurf.models.schemas import OptionsChain, UnderlyingPrice


class ThetaDataClient:
    """
    Client for Theta Data API with mock data fallback.

    If no API key is configured, generates realistic mock data for testing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.theta_api_url

        self._client: Optional[httpx.AsyncClient] = None
        self._mock_generator: Optional[MockDataGenerator] = None

    @property
    def use_mock(self) -> bool:
        """Check if we should use mock data."""
        return self.settings.use_mock_data

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Authorization": f"Bearer {self.settings.theta_api_key}"},
            )
        return self._client

    def _get_mock_generator(self) -> MockDataGenerator:
        """Get or create the mock data generator."""
        if self._mock_generator is None:
            self._mock_generator = MockDataGenerator()
        return self._mock_generator

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_options_chain(
        self,
        symbol: str,
        quote_date: date,
        exp_min: Optional[date] = None,
        exp_max: Optional[date] = None,
    ) -> pl.DataFrame:
        """
        Fetch options chain for a specific date.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            quote_date: Date to fetch data for
            exp_min: Optional minimum expiration date filter
            exp_max: Optional maximum expiration date filter

        Returns:
            Polars DataFrame with columns:
            [symbol, quote_date, expiration_date, strike, option_type,
             bid, ask, mid, last, volume, open_interest,
             delta, gamma, theta, vega, rho, implied_volatility, underlying_price]
        """
        if self.use_mock:
            return await self._get_mock_options_chain(symbol, quote_date, exp_min, exp_max)

        return await self._get_api_options_chain(symbol, quote_date, exp_min, exp_max)

    async def _get_mock_options_chain(
        self,
        symbol: str,
        quote_date: date,
        exp_min: Optional[date],
        exp_max: Optional[date],
    ) -> pl.DataFrame:
        """Generate mock options chain data."""
        generator = self._get_mock_generator()
        chain = generator.generate_options_chain(symbol, quote_date)

        # Convert to DataFrame
        data = []
        for quote in chain.quotes:
            # Apply expiration filters
            if exp_min and quote.expiration_date < exp_min:
                continue
            if exp_max and quote.expiration_date > exp_max:
                continue

            data.append({
                "symbol": quote.symbol,
                "quote_date": quote.quote_date,
                "expiration_date": quote.expiration_date,
                "strike": float(quote.strike),
                "option_type": quote.option_type.value,
                "bid": float(quote.bid) if quote.bid else None,
                "ask": float(quote.ask) if quote.ask else None,
                "mid": float(quote.mid) if quote.mid else None,
                "last": float(quote.last) if quote.last else None,
                "volume": quote.volume,
                "open_interest": quote.open_interest,
                "delta": float(quote.delta) if quote.delta else None,
                "gamma": float(quote.gamma) if quote.gamma else None,
                "theta": float(quote.theta) if quote.theta else None,
                "vega": float(quote.vega) if quote.vega else None,
                "rho": float(quote.rho) if quote.rho else None,
                "implied_volatility": float(quote.implied_volatility) if quote.implied_volatility else None,
                "underlying_price": float(quote.underlying_price),
            })

        if not data:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "quote_date": pl.Date,
                "expiration_date": pl.Date,
                "strike": pl.Float64,
                "option_type": pl.Utf8,
                "bid": pl.Float64,
                "ask": pl.Float64,
                "mid": pl.Float64,
                "last": pl.Float64,
                "volume": pl.Int64,
                "open_interest": pl.Int64,
                "delta": pl.Float64,
                "gamma": pl.Float64,
                "theta": pl.Float64,
                "vega": pl.Float64,
                "rho": pl.Float64,
                "implied_volatility": pl.Float64,
                "underlying_price": pl.Float64,
            })

        return pl.DataFrame(data)

    async def _get_api_options_chain(
        self,
        symbol: str,
        quote_date: date,
        exp_min: Optional[date],
        exp_max: Optional[date],
    ) -> pl.DataFrame:
        """Fetch options chain from Theta Data API."""
        client = await self._get_client()

        # Build request parameters
        params = {
            "root": symbol,
            "start_date": quote_date.isoformat(),
            "end_date": quote_date.isoformat(),
        }

        if exp_min:
            params["exp_start_date"] = exp_min.isoformat()
        if exp_max:
            params["exp_end_date"] = exp_max.isoformat()

        logger.debug(f"Fetching options chain: {params}")

        try:
            # Note: Actual Theta Data API endpoints may differ
            # This is a placeholder implementation
            response = await client.get(
                f"{self.base_url}/hist/option/quote",
                params=params,
            )
            response.raise_for_status()

            data = response.json()

            # Transform API response to DataFrame
            # Actual transformation depends on Theta Data response format
            # This is a placeholder
            logger.warning("Real API implementation pending - using mock data")
            return await self._get_mock_options_chain(symbol, quote_date, exp_min, exp_max)

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            raise

    async def get_underlying_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """
        Fetch underlying OHLCV data for a date range.

        Args:
            symbol: Underlying symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Polars DataFrame with columns:
            [symbol, date, open, high, low, close, volume]
        """
        if self.use_mock:
            return await self._get_mock_underlying_prices(symbol, start_date, end_date)

        return await self._get_api_underlying_prices(symbol, start_date, end_date)

    async def _get_mock_underlying_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Generate mock underlying price data."""
        generator = self._get_mock_generator()

        data = []
        current_date = start_date
        previous_close: Optional[float] = None

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                price = generator.generate_underlying_price(symbol, current_date, previous_close)
                data.append({
                    "symbol": price.symbol,
                    "date": price.date,
                    "open": float(price.open),
                    "high": float(price.high),
                    "low": float(price.low),
                    "close": float(price.close),
                    "volume": price.volume,
                })
                previous_close = float(price.close)

            current_date += timedelta(days=1)

        if not data:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            })

        return pl.DataFrame(data)

    async def _get_api_underlying_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch underlying prices from Theta Data API."""
        client = await self._get_client()

        params = {
            "root": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        logger.debug(f"Fetching underlying prices: {params}")

        try:
            response = await client.get(
                f"{self.base_url}/hist/stock/quote",
                params=params,
            )
            response.raise_for_status()

            # Placeholder - actual implementation depends on API response format
            logger.warning("Real API implementation pending - using mock data")
            return await self._get_mock_underlying_prices(symbol, start_date, end_date)

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            raise

    async def get_historical_chain_batch(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        batch_size: int = 5,
    ) -> AsyncIterator[tuple[date, pl.DataFrame]]:
        """
        Fetch historical options chains in batches.

        Yields (date, DataFrame) tuples for each trading day.

        Args:
            symbol: Underlying symbol
            start_date: Start of date range
            end_date: End of date range
            batch_size: Number of days per batch (for rate limiting)

        Yields:
            Tuple of (date, options_chain_df)
        """
        current_date = start_date
        batch_count = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                try:
                    chain_df = await self.get_options_chain(symbol, current_date)
                    yield current_date, chain_df

                    batch_count += 1
                    if batch_count >= batch_size and not self.use_mock:
                        # Rate limiting for real API
                        import asyncio
                        await asyncio.sleep(1.0)
                        batch_count = 0

                except Exception as e:
                    logger.error(f"Failed to fetch data for {current_date}: {e}")
                    # Continue to next date

            current_date += timedelta(days=1)
