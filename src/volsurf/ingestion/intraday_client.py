"""Theta Data Terminal client for intraday v3 API endpoints."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Optional

import httpx
import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings


class ThetaIntradayClient:
    """
    Client for Theta Data Terminal v3 intraday API endpoints.

    Uses the option/at_time/quote endpoint to fetch NBBO quotes
    at a specific timestamp with wildcard support for all strikes
    and expirations.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.theta_terminal_url
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=120.0)
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _format_time(self, t: time) -> str:
        """Format time for Theta API (HH:MM:SS.mmm format)."""
        return t.strftime("%H:%M:%S.000")

    def _format_date(self, d: date) -> str:
        """Format date for Theta API (YYYYMMDD format)."""
        return d.strftime("%Y%m%d")

    def get_options_at_time(
        self,
        symbol: str,
        target_date: date,
        time_of_day: time,
        expiration: Optional[date] = None,
        timeout: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Fetch NBBO quotes for options chain at a specific time.

        Uses wildcards to get all strikes and optionally all expirations
        in a single request.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            target_date: Date to fetch data for
            time_of_day: Time of day in America/New_York timezone
            expiration: Specific expiration date, or None for all (*)
            timeout: Optional request timeout in seconds

        Returns:
            Polars DataFrame with columns:
            [symbol, timestamp, expiration_date, strike, option_type,
             bid, ask, bid_size, ask_size, mid]
        """
        client = self._get_client()
        params = {
            "symbol": symbol,
            "start_date": self._format_date(target_date),
            "end_date": self._format_date(target_date),
            "time_of_day": self._format_time(time_of_day),
            "expiration": self._format_date(expiration) if expiration else "*",
            "strike": "*",  # Always use wildcard for all strikes
            "format": "json",
        }

        logger.debug(f"Fetching options at time: {params}")

        response = client.get(
            f"{self.base_url}/option/at_time/quote",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()

        # v3 API returns columnar JSON with parallel arrays
        if not data or not data.get("expiration"):
            return self._empty_intraday_df()

        num_rows = len(data["expiration"])
        records = []

        for i in range(num_rows):
            exp_str = data["expiration"][i]
            if not exp_str:
                continue

            # Parse timestamp from response
            ts_str = data.get("timestamp", [None] * num_rows)[i]
            if ts_str:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                # Construct timestamp from date and time_of_day
                timestamp = datetime.combine(target_date, time_of_day)

            bid = data.get("bid", [None] * num_rows)[i]
            ask = data.get("ask", [None] * num_rows)[i]

            # Calculate mid price
            mid = None
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2

            records.append({
                "symbol": symbol,
                "timestamp": timestamp,
                "expiration_date": date.fromisoformat(exp_str),
                "strike": float(data["strike"][i]) if data.get("strike") and data["strike"][i] is not None else None,
                "option_type": data["right"][i].upper() if data.get("right") and data["right"][i] else None,
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "bid_size": int(data["bid_size"][i]) if data.get("bid_size") and data["bid_size"][i] is not None else None,
                "ask_size": int(data["ask_size"][i]) if data.get("ask_size") and data["ask_size"][i] is not None else None,
                "mid": mid,
            })

        if not records:
            return self._empty_intraday_df()

        return pl.DataFrame(records)

    def get_underlying_at_time(
        self,
        symbol: str,
        target_date: date,
        time_of_day: time,
        timeout: Optional[float] = None,
    ) -> Optional[Decimal]:
        """
        Fetch underlying price at a specific time.

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            target_date: Date to fetch data for
            time_of_day: Time of day in America/New_York timezone
            timeout: Optional request timeout in seconds

        Returns:
            Underlying price at the specified time, or None if not available
        """
        client = self._get_client()
        params = {
            "symbol": symbol,
            "start_date": self._format_date(target_date),
            "end_date": self._format_date(target_date),
            "time_of_day": self._format_time(time_of_day),
            "format": "json",
        }

        logger.debug(f"Fetching underlying at time: {params}")

        # Try trade endpoint first
        try:
            response = client.get(
                f"{self.base_url}/stock/at_time/trade",
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data and data.get("price") and len(data["price"]) > 0:
                price = data["price"][0]
                if price is not None:
                    return Decimal(str(price))
        except Exception as e:
            logger.warning(f"Failed to get underlying trade: {e}")

        # Fall back to quote endpoint and use mid
        try:
            response = client.get(
                f"{self.base_url}/stock/at_time/quote",
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data and data.get("bid") and data.get("ask"):
                bid = data["bid"][0] if data["bid"] else None
                ask = data["ask"][0] if data["ask"] else None
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2
                    return Decimal(str(mid))
        except Exception as e:
            logger.warning(f"Failed to get underlying quote: {e}")

        return None

    def _empty_intraday_df(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame(schema={
            "symbol": pl.Utf8,
            "timestamp": pl.Datetime,
            "expiration_date": pl.Date,
            "strike": pl.Float64,
            "option_type": pl.Utf8,
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_size": pl.Int64,
            "ask_size": pl.Int64,
            "mid": pl.Float64,
        })

    def check_terminal_status(self) -> bool:
        """Check if the Theta Terminal is running and accessible."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self.base_url}/option/list/expirations",
                params={"symbol": "SPY", "format": "json"},
                timeout=10.0,
            )
            return response.status_code == 200
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Theta Terminal at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error checking terminal status: {e}")
            return False
