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
