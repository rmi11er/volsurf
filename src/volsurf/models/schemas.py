"""Pydantic models for options data and surface parameters."""

import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OptionType(str, Enum):
    """Option type enum."""

    CALL = "CALL"
    PUT = "PUT"


class OptionQuote(BaseModel):
    """Single option quote data."""

    symbol: str
    quote_date: date
    expiration_date: date
    strike: Decimal
    option_type: OptionType

    # Market data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    mid: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    # Greeks
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    rho: Optional[Decimal] = None

    # Implied volatility
    implied_volatility: Optional[Decimal] = None

    # Underlying
    underlying_price: Decimal


class OptionsChain(BaseModel):
    """Complete options chain for a symbol on a date."""

    symbol: str
    quote_date: date
    underlying_price: Decimal
    quotes: list[OptionQuote]


class UnderlyingPrice(BaseModel):
    """Underlying OHLCV data."""

    symbol: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class SVIParams(BaseModel):
    """SVI model parameters (raw parameterization)."""

    # w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))
    a: float = Field(description="Vertical shift (ATM variance level)")
    b: float = Field(ge=0, description="Slope of the wings (non-negative)")
    rho: float = Field(ge=-1, le=1, description="Skew parameter")
    m: float = Field(description="Horizontal shift (ATM log-moneyness)")
    sigma: float = Field(gt=0, description="Smoothness parameter")

    def total_variance(self, log_moneyness: float) -> float:
        """
        Calculate total implied variance for given log-moneyness.

        Args:
            log_moneyness: k = log(K/F)

        Returns:
            Total variance w(k)
        """
        import math

        k = log_moneyness
        return self.a + self.b * (
            self.rho * (k - self.m) + math.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, log_moneyness: float, tte_years: float) -> float:
        """
        Calculate implied volatility for given log-moneyness and time to expiry.

        Args:
            log_moneyness: k = log(K/F)
            tte_years: Time to expiration in years

        Returns:
            Implied volatility (annualized)
        """
        import math

        w = self.total_variance(log_moneyness)
        return math.sqrt(w / tte_years) if tte_years > 0 else 0.0


class FittedSurface(BaseModel):
    """Fitted volatility surface for a single expiration."""

    symbol: str
    quote_date: date
    expiration_date: date
    tte_years: float
    forward_price: Decimal

    # SVI parameters
    svi_params: SVIParams

    # Derived quantities
    atm_vol: float
    skew_25delta: Optional[float] = None

    # Fit quality
    rmse: float
    mae: float
    max_error: float
    num_points: int

    # Arbitrage checks
    passes_no_arbitrage: bool
    butterfly_violations: int = 0
    calendar_violations: int = 0


class SABRParams(BaseModel):
    """SABR model parameters."""

    # SABR stochastic volatility model
    alpha: float = Field(gt=0, description="Initial volatility level")
    beta: float = Field(ge=0, le=1, description="CEV exponent (0=normal, 1=lognormal)")
    rho: float = Field(ge=-1, le=1, description="Correlation between forward and vol")
    nu: float = Field(ge=0, description="Vol-of-vol")

    def implied_vol(
        self, forward: float, strike: float, tte_years: float
    ) -> float:
        """Calculate implied volatility using Hagan's approximation."""
        if tte_years <= 0:
            return 0.0

        # ATM case
        if abs(forward - strike) < 1e-10:
            fk = forward
            fk_beta = fk ** (1 - self.beta)
            term1 = self.alpha / fk_beta
            term2 = 1 + (
                ((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / fk ** (2 - 2 * self.beta))
                + (self.rho * self.beta * self.nu * self.alpha) / (4 * fk_beta)
                + ((2 - 3 * self.rho ** 2) * self.nu ** 2) / 24
            ) * tte_years
            return term1 * term2

        # General case
        fk = forward * strike
        fk_beta = fk ** ((1 - self.beta) / 2)
        log_fk = math.log(forward / strike)

        z = (self.nu / self.alpha) * fk_beta * log_fk

        if abs(z) < 1e-10:
            x_z = 1.0
        else:
            sqrt_term = math.sqrt(1 - 2 * self.rho * z + z ** 2)
            x_z = z / math.log((sqrt_term + z - self.rho) / (1 - self.rho))

        one_minus_beta = 1 - self.beta
        fk_pow = fk ** (one_minus_beta / 2)

        denom = fk_pow * (
            1
            + (one_minus_beta ** 2 / 24) * log_fk ** 2
            + (one_minus_beta ** 4 / 1920) * log_fk ** 4
        )

        numer_corr = 1 + (
            (one_minus_beta ** 2 / 24) * (self.alpha ** 2 / fk ** one_minus_beta)
            + (self.rho * self.beta * self.nu * self.alpha) / (4 * fk_pow)
            + ((2 - 3 * self.rho ** 2) * self.nu ** 2) / 24
        ) * tte_years

        return max((self.alpha / denom) * x_z * numer_corr, 1e-10)


class PolynomialParams(BaseModel):
    """Polynomial smile model parameters."""

    # Polynomial fit: IV(k) = a0 + a1*k + a2*k^2 + a3*k^3 + ...
    coefficients: list[float] = Field(description="Polynomial coefficients [a0, a1, a2, ...]")
    degree: int = Field(ge=1, le=6, description="Polynomial degree")

    def implied_vol(self, log_moneyness: float) -> float:
        """Calculate implied volatility from polynomial."""
        k = log_moneyness
        result = 0.0
        for i, coef in enumerate(self.coefficients):
            result += coef * (k ** i)
        return max(result, 1e-10)


class TermStructureParams(BaseModel):
    """Term structure model parameters."""

    symbol: str
    quote_date: date

    # ATM term structure: σ(T) = a * T^b
    atm_a: float
    atm_b: float
    atm_rmse: float

    # Skew term structure
    skew_a: Optional[float] = None
    skew_b: Optional[float] = None
    skew_rmse: Optional[float] = None

    num_expirations: int


# --- Intraday Models ---


class IntradayOptionQuote(BaseModel):
    """Single option quote at a specific intraday timestamp."""

    symbol: str
    timestamp: datetime
    expiration_date: date
    strike: Decimal
    option_type: OptionType

    # Market data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    @property
    def mid(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


class IntradayOptionsChain(BaseModel):
    """Complete options chain at a specific intraday timestamp."""

    symbol: str
    timestamp: datetime
    underlying_price: Decimal
    quotes: list[IntradayOptionQuote]

    def to_dataframe(self) -> "pl.DataFrame":
        """Convert to Polars DataFrame for analysis."""
        import polars as pl

        records = []
        for q in self.quotes:
            records.append({
                "symbol": q.symbol,
                "timestamp": q.timestamp,
                "expiration_date": q.expiration_date,
                "strike": float(q.strike),
                "option_type": q.option_type.value,
                "bid": float(q.bid) if q.bid else None,
                "ask": float(q.ask) if q.ask else None,
                "mid": float(q.mid) if q.mid else None,
                "bid_size": q.bid_size,
                "ask_size": q.ask_size,
                "underlying_price": float(self.underlying_price),
            })
        return pl.DataFrame(records)


class IntradayFittedSurface(BaseModel):
    """Fitted volatility surface at a specific intraday timestamp."""

    symbol: str
    timestamp: datetime  # Intraday timestamp instead of date
    expiration_date: date
    tte_years: float
    forward_price: Decimal

    # SVI parameters
    svi_params: SVIParams

    # Derived quantities
    atm_vol: float
    skew_25delta: Optional[float] = None

    # Fit quality
    rmse: float
    mae: float
    max_error: float
    num_points: int

    # Arbitrage checks
    passes_no_arbitrage: bool
    butterfly_violations: int = 0
    calendar_violations: int = 0


class IntradaySurfaceResult(BaseModel):
    """Result of fitting surfaces at an intraday timestamp."""

    symbol: str
    timestamp: datetime
    underlying_price: Decimal
    surfaces: list[IntradayFittedSurface]

    # Aggregate statistics
    total_expirations: int = 0
    successful_fits: int = 0
    failed_fits: int = 0
    avg_rmse: Optional[float] = None

    def to_parquet(self, path: str) -> None:
        """Export surfaces to Parquet file."""
        import polars as pl

        records = []
        for s in self.surfaces:
            records.append({
                "symbol": s.symbol,
                "timestamp": s.timestamp,
                "expiration_date": s.expiration_date,
                "tte_years": s.tte_years,
                "forward_price": float(s.forward_price),
                "underlying_price": float(self.underlying_price),
                "svi_a": s.svi_params.a,
                "svi_b": s.svi_params.b,
                "svi_rho": s.svi_params.rho,
                "svi_m": s.svi_params.m,
                "svi_sigma": s.svi_params.sigma,
                "atm_vol": s.atm_vol,
                "skew_25delta": s.skew_25delta,
                "rmse": s.rmse,
                "mae": s.mae,
                "max_error": s.max_error,
                "num_points": s.num_points,
                "passes_no_arbitrage": s.passes_no_arbitrage,
            })
        if records:
            pl.DataFrame(records).write_parquet(path)

    @classmethod
    def from_parquet(cls, path: str) -> "IntradaySurfaceResult":
        """Load surfaces from Parquet file."""
        import polars as pl

        df = pl.read_parquet(path)
        if df.is_empty():
            raise ValueError(f"Empty Parquet file: {path}")

        first_row = df.row(0, named=True)
        surfaces = []

        for row in df.iter_rows(named=True):
            surfaces.append(IntradayFittedSurface(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                expiration_date=row["expiration_date"],
                tte_years=row["tte_years"],
                forward_price=Decimal(str(row["forward_price"])),
                svi_params=SVIParams(
                    a=row["svi_a"],
                    b=row["svi_b"],
                    rho=row["svi_rho"],
                    m=row["svi_m"],
                    sigma=row["svi_sigma"],
                ),
                atm_vol=row["atm_vol"],
                skew_25delta=row["skew_25delta"],
                rmse=row["rmse"],
                mae=row["mae"],
                max_error=row["max_error"],
                num_points=row["num_points"],
                passes_no_arbitrage=row["passes_no_arbitrage"],
            ))

        # Use actual underlying_price if available, else fall back to forward_price
        if "underlying_price" in df.columns and first_row.get("underlying_price") is not None:
            underlying = Decimal(str(first_row["underlying_price"]))
        else:
            underlying = Decimal(str(first_row["forward_price"]))

        return cls(
            symbol=first_row["symbol"],
            timestamp=first_row["timestamp"],
            underlying_price=underlying,
            surfaces=surfaces,
            total_expirations=len(surfaces),
            successful_fits=len(surfaces),
        )
