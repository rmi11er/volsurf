"""Theta Data Terminal client for local v3 API."""

from datetime import date, timedelta
from typing import Iterator, Optional

import httpx
import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings
from volsurf.ingestion.mock_data import MockDataGenerator


class ThetaTerminalClient:
    """
    Client for Theta Data Terminal v3 local API.

    Connects to the locally running Theta Terminal (Java application)
    which exposes a REST API on localhost.

    If no username is configured, falls back to mock data for testing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.theta_terminal_url

        self._client: Optional[httpx.Client] = None
        self._mock_generator: Optional[MockDataGenerator] = None

    @property
    def use_mock(self) -> bool:
        """Check if we should use mock data."""
        return self.settings.use_mock_data

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def _get_mock_generator(self) -> MockDataGenerator:
        """Get or create the mock data generator."""
        if self._mock_generator is None:
            self._mock_generator = MockDataGenerator()
        return self._mock_generator

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def check_terminal_status(self) -> bool:
        """Check if the Theta Terminal is running and accessible."""
        if self.use_mock:
            return True

        try:
            client = self._get_client()
            # Try to list SPY expirations as a health check
            response = client.get(
                f"{self.base_url}/option/list/expirations",
                params={"symbol": "SPY", "format": "json"},
            )
            return response.status_code == 200
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Theta Terminal at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error checking terminal status: {e}")
            return False

    def get_expirations(self, symbol: str) -> list[date]:
        """
        Get available expiration dates for a symbol.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')

        Returns:
            List of expiration dates
        """
        if self.use_mock:
            # Generate mock expirations (weekly + monthly for next 2 years)
            today = date.today()
            expirations = []
            current = today + timedelta(days=7)
            while current < today + timedelta(days=730):
                # Add Fridays as expirations
                if current.weekday() == 4:
                    expirations.append(current)
                current += timedelta(days=1)
            return expirations

        client = self._get_client()
        response = client.get(
            f"{self.base_url}/option/list/expirations",
            params={"symbol": symbol, "format": "json"},
        )
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON: {"symbol": [...], "expiration": [...]}
        expirations = []
        for exp_str in data.get("expiration", []):
            if exp_str:
                expirations.append(date.fromisoformat(exp_str))

        return sorted(set(expirations))  # dedupe and sort

    def get_strikes(self, symbol: str, expiration: date) -> list[float]:
        """
        Get available strikes for a symbol and expiration.

        Args:
            symbol: Underlying symbol
            expiration: Expiration date

        Returns:
            List of strike prices
        """
        if self.use_mock:
            # Generate mock strikes around $500 (SPY-like)
            return [float(x) for x in range(400, 601, 5)]

        client = self._get_client()
        response = client.get(
            f"{self.base_url}/option/list/strikes",
            params={
                "symbol": symbol,
                "expiration": expiration.strftime("%Y%m%d"),
                "format": "json",
            },
        )
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON: {"symbol": [...], "strike": [...]}
        strikes = [float(s) for s in data.get("strike", []) if s is not None]
        return sorted(set(strikes))  # dedupe and sort

    def get_options_eod(
        self,
        symbol: str,
        quote_date: date,
        expiration: Optional[date] = None,
    ) -> pl.DataFrame:
        """
        Fetch end-of-day options data for a specific date.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            quote_date: Date to fetch data for
            expiration: Optional specific expiration (default: all expirations)

        Returns:
            Polars DataFrame with columns:
            [symbol, quote_date, expiration_date, strike, option_type,
             open, high, low, close, bid, ask, volume, count]
        """
        if self.use_mock:
            return self._get_mock_options_eod(symbol, quote_date)

        client = self._get_client()
        params = {
            "symbol": symbol,
            "start_date": quote_date.strftime("%Y%m%d"),
            "end_date": quote_date.strftime("%Y%m%d"),
            "expiration": expiration.strftime("%Y%m%d") if expiration else "*",
            "format": "json",
        }

        logger.debug(f"Fetching options EOD: {params}")

        response = client.get(f"{self.base_url}/option/history/eod", params=params)
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON with parallel arrays
        if not data or not data.get("expiration"):
            return self._empty_options_df()

        num_rows = len(data["expiration"])
        records = []
        for i in range(num_rows):
            exp_str = data["expiration"][i]
            records.append({
                "symbol": symbol,
                "quote_date": quote_date,
                "expiration_date": date.fromisoformat(exp_str) if exp_str else None,
                "strike": float(data["strike"][i]) if data.get("strike") and data["strike"][i] is not None else None,
                "option_type": data["right"][i].upper() if data.get("right") and data["right"][i] else None,
                "open": float(data["open"][i]) if data.get("open") and data["open"][i] is not None else None,
                "high": float(data["high"][i]) if data.get("high") and data["high"][i] is not None else None,
                "low": float(data["low"][i]) if data.get("low") and data["low"][i] is not None else None,
                "close": float(data["close"][i]) if data.get("close") and data["close"][i] is not None else None,
                "bid": float(data["bid"][i]) if data.get("bid") and data["bid"][i] is not None else None,
                "ask": float(data["ask"][i]) if data.get("ask") and data["ask"][i] is not None else None,
                "volume": int(data["volume"][i]) if data.get("volume") and data["volume"][i] is not None else 0,
                "count": int(data["count"][i]) if data.get("count") and data["count"][i] is not None else 0,
            })

        return pl.DataFrame(records)

    def get_options_open_interest(
        self,
        symbol: str,
        quote_date: date,
        expiration: Optional[date] = None,
    ) -> pl.DataFrame:
        """
        Fetch historical open interest data.

        Args:
            symbol: Underlying symbol
            quote_date: Date to fetch data for
            expiration: Optional specific expiration

        Returns:
            DataFrame with [expiration_date, strike, option_type, open_interest]
        """
        if self.use_mock:
            return self._get_mock_open_interest(symbol, quote_date)

        client = self._get_client()
        params = {
            "symbol": symbol,
            "date": quote_date.strftime("%Y%m%d"),
            "expiration": expiration.strftime("%Y%m%d") if expiration else "*",
            "format": "json",
        }

        response = client.get(f"{self.base_url}/option/history/open_interest", params=params)
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON with parallel arrays
        if not data or not data.get("expiration"):
            return pl.DataFrame(schema={
                "expiration_date": pl.Date,
                "strike": pl.Float64,
                "option_type": pl.Utf8,
                "open_interest": pl.Int64,
            })

        num_rows = len(data["expiration"])
        records = []
        for i in range(num_rows):
            records.append({
                "expiration_date": date.fromisoformat(data["expiration"][i]) if data["expiration"][i] else None,
                "strike": float(data["strike"][i]) if data.get("strike") and data["strike"][i] is not None else None,
                "option_type": data["right"][i].upper() if data.get("right") and data["right"][i] else None,
                "open_interest": int(data["open_interest"][i]) if data.get("open_interest") and data["open_interest"][i] is not None else 0,
            })

        return pl.DataFrame(records)

    def get_options_chain(
        self,
        symbol: str,
        quote_date: date,
    ) -> pl.DataFrame:
        """
        Fetch complete options chain for a date (EOD + Open Interest combined).

        This is the main method for ingestion - combines EOD prices with OI data.

        Args:
            symbol: Underlying symbol
            quote_date: Date to fetch data for

        Returns:
            Polars DataFrame with full options chain data
        """
        # Get EOD data
        eod_df = self.get_options_eod(symbol, quote_date)
        if eod_df.is_empty():
            return eod_df

        # Get open interest
        oi_df = self.get_options_open_interest(symbol, quote_date)

        # Join EOD with OI
        if not oi_df.is_empty():
            eod_df = eod_df.join(
                oi_df,
                on=["expiration_date", "strike", "option_type"],
                how="left",
            )
        else:
            eod_df = eod_df.with_columns(pl.lit(None).alias("open_interest").cast(pl.Int64))

        # Calculate mid price
        eod_df = eod_df.with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid")
        )

        return eod_df

    def get_underlying_eod(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """
        Fetch underlying stock EOD data for a date range.

        Args:
            symbol: Stock symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with [symbol, date, open, high, low, close, volume]
        """
        if self.use_mock:
            return self._get_mock_underlying_prices(symbol, start_date, end_date)

        client = self._get_client()
        params = {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "format": "json",
        }

        logger.debug(f"Fetching underlying EOD: {params}")

        response = client.get(f"{self.base_url}/stock/history/eod", params=params)
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON with parallel arrays
        if not data or not data.get("created"):
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            })

        num_rows = len(data["created"])
        records = []
        for i in range(num_rows):
            # Parse the created timestamp to get the date
            created = data["created"][i]
            if not created:
                continue
            row_date = date.fromisoformat(created[:10])

            records.append({
                "symbol": symbol,
                "date": row_date,
                "open": float(data["open"][i]) if data.get("open") and data["open"][i] is not None else None,
                "high": float(data["high"][i]) if data.get("high") and data["high"][i] is not None else None,
                "low": float(data["low"][i]) if data.get("low") and data["low"][i] is not None else None,
                "close": float(data["close"][i]) if data.get("close") and data["close"][i] is not None else None,
                "volume": int(data["volume"][i]) if data.get("volume") and data["volume"][i] is not None else 0,
            })

        return pl.DataFrame(records)

    def get_realtime_greeks(
        self,
        symbol: str,
        expiration: Optional[date] = None,
    ) -> pl.DataFrame:
        """
        Fetch real-time Greeks snapshot (only works during market hours).

        Args:
            symbol: Underlying symbol
            expiration: Optional specific expiration

        Returns:
            DataFrame with Greeks data including implied volatility
        """
        if self.use_mock:
            logger.warning("Real-time Greeks not available in mock mode")
            return pl.DataFrame()

        client = self._get_client()
        params = {
            "symbol": symbol,
            "expiration": expiration.strftime("%Y%m%d") if expiration else "*",
            "format": "json",
        }

        response = client.get(f"{self.base_url}/option/snapshot/greeks/all", params=params)
        response.raise_for_status()

        data = response.json()
        # v3 API returns columnar JSON with parallel arrays
        if not data or not data.get("expiration"):
            return pl.DataFrame()

        num_rows = len(data["expiration"])
        records = []
        for i in range(num_rows):
            records.append({
                "symbol": symbol,
                "expiration_date": date.fromisoformat(data["expiration"][i]) if data["expiration"][i] else None,
                "strike": float(data["strike"][i]) if data.get("strike") and data["strike"][i] is not None else None,
                "option_type": data["right"][i].upper() if data.get("right") and data["right"][i] else None,
                "bid": float(data["bid"][i]) if data.get("bid") and data["bid"][i] is not None else None,
                "ask": float(data["ask"][i]) if data.get("ask") and data["ask"][i] is not None else None,
                "implied_volatility": float(data["implied_volatility"][i]) if data.get("implied_volatility") and data["implied_volatility"][i] is not None else None,
                "delta": float(data["delta"][i]) if data.get("delta") and data["delta"][i] is not None else None,
                "gamma": float(data["gamma"][i]) if data.get("gamma") and data["gamma"][i] is not None else None,
                "theta": float(data["theta"][i]) if data.get("theta") and data["theta"][i] is not None else None,
                "vega": float(data["vega"][i]) / 100 if data.get("vega") and data["vega"][i] is not None else None,
                "rho": float(data["rho"][i]) / 100 if data.get("rho") and data["rho"][i] is not None else None,
                "underlying_price": float(data["underlying_price"][i]) if data.get("underlying_price") and data["underlying_price"][i] is not None else None,
            })

        return pl.DataFrame(records)

    def iter_historical_chains(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Iterator[tuple[date, pl.DataFrame]]:
        """
        Iterate over historical options chains for a date range.

        Yields (date, DataFrame) tuples for each trading day.

        Args:
            symbol: Underlying symbol
            start_date: Start of date range
            end_date: End of date range

        Yields:
            Tuple of (quote_date, options_chain_df)
        """
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                try:
                    chain_df = self.get_options_chain(symbol, current_date)
                    if not chain_df.is_empty():
                        yield current_date, chain_df
                    else:
                        logger.debug(f"No data for {symbol} on {current_date}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        logger.debug(f"No data for {symbol} on {current_date}")
                    else:
                        logger.error(f"Failed to fetch {symbol} on {current_date}: {e}")
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} on {current_date}: {e}")

            current_date += timedelta(days=1)

    # --- Mock data methods ---

    def _empty_options_df(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame(schema={
            "symbol": pl.Utf8,
            "quote_date": pl.Date,
            "expiration_date": pl.Date,
            "strike": pl.Float64,
            "option_type": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "bid": pl.Float64,
            "ask": pl.Float64,
            "volume": pl.Int64,
            "count": pl.Int64,
        })

    def _get_mock_options_eod(self, symbol: str, quote_date: date) -> pl.DataFrame:
        """Generate mock EOD options data."""
        generator = self._get_mock_generator()
        chain = generator.generate_options_chain(symbol, quote_date)

        records = []
        for quote in chain.quotes:
            records.append({
                "symbol": symbol,
                "quote_date": quote_date,
                "expiration_date": quote.expiration_date,
                "strike": float(quote.strike),
                "option_type": quote.option_type.value,
                "open": float(quote.bid) if quote.bid else None,
                "high": float(quote.ask) if quote.ask else None,
                "low": float(quote.bid) if quote.bid else None,
                "close": float(quote.mid) if quote.mid else None,
                "bid": float(quote.bid) if quote.bid else None,
                "ask": float(quote.ask) if quote.ask else None,
                "volume": quote.volume or 0,
                "count": (quote.volume or 0) // 10,
            })

        if not records:
            return self._empty_options_df()

        return pl.DataFrame(records)

    def _get_mock_open_interest(self, symbol: str, quote_date: date) -> pl.DataFrame:
        """Generate mock open interest data."""
        generator = self._get_mock_generator()
        chain = generator.generate_options_chain(symbol, quote_date)

        records = []
        for quote in chain.quotes:
            records.append({
                "expiration_date": quote.expiration_date,
                "strike": float(quote.strike),
                "option_type": quote.option_type.value,
                "open_interest": quote.open_interest or 0,
            })

        if not records:
            return pl.DataFrame(schema={
                "expiration_date": pl.Date,
                "strike": pl.Float64,
                "option_type": pl.Utf8,
                "open_interest": pl.Int64,
            })

        return pl.DataFrame(records)

    def _get_mock_underlying_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Generate mock underlying price data."""
        generator = self._get_mock_generator()

        records = []
        current_date = start_date
        previous_close: Optional[float] = None

        while current_date <= end_date:
            if current_date.weekday() < 5:
                price = generator.generate_underlying_price(symbol, current_date, previous_close)
                records.append({
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

        if not records:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            })

        return pl.DataFrame(records)


# Backwards compatibility alias
ThetaDataClient = ThetaTerminalClient
