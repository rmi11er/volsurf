"""Deribit public API client for crypto options and underlying data."""

from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterator, Optional, Set

import httpx
import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings


class DeribitClient:
    """Client for Deribit's public HTTP API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.deribit_base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        """Call a Deribit public endpoint and return the result payload."""
        client = self._get_client()
        response = client.get(f"{self.base_url}/{method}", params=params)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Deribit API error for {method}: {payload['error']}")
        return payload.get("result")

    def check_terminal_status(self) -> bool:
        """Check whether Deribit is reachable."""
        try:
            self._request("public/get_time", {})
            return True
        except Exception as exc:
            logger.warning(f"Cannot connect to Deribit at {self.base_url}: {exc}")
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize a crypto symbol to Deribit currency format."""
        return symbol.strip().upper()

    def _today_utc(self) -> date:
        """Current UTC date."""
        return datetime.now(UTC).date()

    def _parse_expiration_date(self, timestamp_ms: int) -> date:
        """Convert millisecond timestamp to a UTC date."""
        return datetime.fromtimestamp(timestamp_ms / 1000, UTC).date()

    def _empty_options_df(self) -> pl.DataFrame:
        """Return empty DataFrame with expected options schema."""
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
            "implied_volatility": pl.Float64,
        })

    def _empty_underlying_df(self) -> pl.DataFrame:
        """Return empty DataFrame with expected OHLCV schema."""
        return pl.DataFrame(schema={
            "symbol": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        })

    def _option_price_to_usd(self, price: Optional[float], underlying_price: Optional[float]) -> Optional[float]:
        """Convert Deribit option premiums into USD-equivalent prices."""
        if price is None:
            return None
        if not self.settings.deribit_quote_in_usd:
            return float(price)
        if underlying_price is None:
            return None
        return float(price) * float(underlying_price)

    def get_instruments(self, symbol: str) -> list[dict[str, Any]]:
        """Get active option instruments for a crypto symbol."""
        currency = self._normalize_symbol(symbol)
        result = self._request(
            "public/get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
        )
        if not result:
            return []
        return [item for item in result if item.get("kind") == "option"]

    def get_expirations(self, symbol: str) -> list[date]:
        """Get available expiration dates."""
        expirations = {
            self._parse_expiration_date(int(item["expiration_timestamp"]))
            for item in self.get_instruments(symbol)
            if item.get("expiration_timestamp") is not None
        }
        return sorted(expirations)

    def get_strikes(self, symbol: str, expiration: date) -> list[float]:
        """Get available strikes for an expiration."""
        strikes = []
        for item in self.get_instruments(symbol):
            exp_ts = item.get("expiration_timestamp")
            if exp_ts is None:
                continue
            if self._parse_expiration_date(int(exp_ts)) != expiration:
                continue
            strike = item.get("strike")
            if strike is not None:
                strikes.append(float(strike))
        return sorted(set(strikes))

    def get_options_chain(
        self,
        symbol: str,
        quote_date: date,
        timeout: Optional[float] = None,
    ) -> pl.DataFrame:
        """Fetch a current options snapshot for a crypto symbol."""
        del timeout

        today = self._today_utc()
        if quote_date != today:
            logger.warning(
                f"Deribit client currently supports snapshot ingestion for today's UTC date only. "
                f"Requested {quote_date}, current UTC date is {today}."
            )
            return self._empty_options_df()

        currency = self._normalize_symbol(symbol)
        instruments = self.get_instruments(currency)
        if not instruments:
            return self._empty_options_df()

        instrument_map = {
            item["instrument_name"]: item
            for item in instruments
            if item.get("instrument_name")
        }

        summaries = self._request(
            "public/get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
        )
        if not summaries:
            return self._empty_options_df()

        records: list[dict[str, Any]] = []
        for summary in summaries:
            instrument_name = summary.get("instrument_name")
            instrument = instrument_map.get(instrument_name)
            if instrument is None:
                continue

            underlying_price = summary.get("underlying_price")
            bid = self._option_price_to_usd(summary.get("bid_price"), underlying_price)
            ask = self._option_price_to_usd(summary.get("ask_price"), underlying_price)
            last = self._option_price_to_usd(summary.get("last"), underlying_price)
            mark = self._option_price_to_usd(summary.get("mark_price"), underlying_price)
            mid = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2
            elif mark is not None:
                mid = mark

            expiration_timestamp = instrument.get("expiration_timestamp")
            if expiration_timestamp is None:
                continue

            option_type = str(instrument.get("option_type", "")).upper()
            if option_type == "CALL":
                normalized_type = "CALL"
            elif option_type == "PUT":
                normalized_type = "PUT"
            else:
                continue

            records.append({
                "symbol": currency,
                "quote_date": quote_date,
                "expiration_date": self._parse_expiration_date(int(expiration_timestamp)),
                "strike": float(instrument["strike"]),
                "option_type": normalized_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": last,
                "volume": int(round(float(summary["volume"]))) if summary.get("volume") is not None else 0,
                "open_interest": None,
                "implied_volatility": (
                    float(summary["mark_iv"]) / 100.0
                    if summary.get("mark_iv") is not None
                    else None
                ),
                "underlying_price": float(underlying_price) if underlying_price is not None else None,
            })

        if not records:
            return self._empty_options_df()

        return pl.DataFrame(records)

    def get_underlying_eod(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV from the perpetual contract chart endpoint."""
        instrument_name = f"{self._normalize_symbol(symbol)}-PERPETUAL"
        start_ts = int(datetime.combine(start_date, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        end_ts = int(end_dt.timestamp() * 1000) - 1

        result = self._request(
            "public/get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "resolution": "1D",
            },
        )

        if not result or not result.get("ticks"):
            return self._empty_underlying_df()

        ticks = result.get("ticks", [])
        opens = result.get("open", [])
        highs = result.get("high", [])
        lows = result.get("low", [])
        closes = result.get("close", [])
        volumes = result.get("volume", [])

        records = []
        for i, tick in enumerate(ticks):
            row_date = datetime.fromtimestamp(tick / 1000, UTC).date()
            if row_date < start_date or row_date > end_date:
                continue
            records.append({
                "symbol": self._normalize_symbol(symbol),
                "date": row_date,
                "open": float(opens[i]) if i < len(opens) and opens[i] is not None else None,
                "high": float(highs[i]) if i < len(highs) and highs[i] is not None else None,
                "low": float(lows[i]) if i < len(lows) and lows[i] is not None else None,
                "close": float(closes[i]) if i < len(closes) and closes[i] is not None else None,
                "volume": int(round(float(volumes[i]))) if i < len(volumes) and volumes[i] is not None else 0,
            })

        if not records:
            return self._empty_underlying_df()

        return pl.DataFrame(records)

    def iter_historical_chains(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        skip_dates: Optional[Set[date]] = None,
    ) -> Iterator[tuple[date, pl.DataFrame]]:
        """Iterate available option snapshots.

        Deribit public HTTP access gives us a live chain snapshot, not a historical daily chain archive.
        For now we only ingest the current UTC day's snapshot.
        """
        skip_dates = skip_dates or set()
        today = self._today_utc()
        if not (start_date <= today <= end_date):
            logger.warning(
                "Deribit historical options-chain backfill is not implemented with the public snapshot API. "
                f"Only today's UTC snapshot ({today}) can be ingested right now."
            )
            return

        if today in skip_dates:
            logger.debug(f"Skipping {today} - already exists")
            return

        chain_df = self.get_options_chain(symbol, today)
        if not chain_df.is_empty():
            yield today, chain_df
