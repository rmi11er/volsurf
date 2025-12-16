"""Mock data generator for testing without API access."""

import math
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
from loguru import logger

from volsurf.config.settings import get_settings
from volsurf.models.schemas import (
    OptionQuote,
    OptionsChain,
    OptionType,
    SVIParams,
    UnderlyingPrice,
)


class MockDataGenerator:
    """
    Generate realistic mock options chain data.

    Uses SVI model to create realistic volatility smiles and generates
    consistent Greeks and prices.
    """

    # Typical SVI parameters for SPY-like volatility surface
    # These create a realistic skew with higher put volatility
    DEFAULT_SVI_BASE = SVIParams(
        a=0.04,  # Base variance level (~20% vol)
        b=0.15,  # Wing slope
        rho=-0.4,  # Negative skew (puts more expensive)
        m=0.0,  # Centered at ATM
        sigma=0.1,  # Smoothness
    )

    def __init__(
        self,
        base_price: float = 590.0,  # Approximate current SPY price
        base_volatility: float = 0.18,  # Base ATM volatility
        risk_free_rate: float = 0.045,  # Approximate current rate
        dividend_yield: float = 0.013,  # SPY dividend yield
    ):
        self.base_price = base_price
        self.base_volatility = base_volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.settings = get_settings()

        # Random state for reproducibility per date
        self._rng = np.random.default_rng(42)

    def _get_date_rng(self, quote_date: date) -> np.random.Generator:
        """Get a reproducible random generator for a specific date."""
        seed = int(quote_date.toordinal())
        return np.random.default_rng(seed)

    def _generate_svi_params(self, tte_years: float, quote_date: date) -> SVIParams:
        """
        Generate SVI parameters with term structure effects.

        ATM vol typically decreases with maturity (power law).
        Skew also decreases with maturity.
        """
        rng = self._get_date_rng(quote_date)

        # Add some day-to-day variation
        vol_noise = rng.normal(0, 0.01)
        skew_noise = rng.normal(0, 0.05)

        # Term structure: vol decreases with sqrt(T) approximately
        # σ(T) ≈ base_vol * T^(-0.1) with floor
        term_adjustment = max(0.8, min(1.3, tte_years ** (-0.1)))
        atm_vol = self.base_volatility * term_adjustment + vol_noise
        atm_vol = max(0.10, min(0.50, atm_vol))  # Clamp to reasonable range

        # Skew decreases with maturity
        skew_adjustment = max(0.5, min(1.5, tte_years ** (-0.25)))
        rho = -0.35 * skew_adjustment + skew_noise * 0.1
        rho = max(-0.9, min(0.3, rho))

        # Total variance at ATM
        a = atm_vol**2 * tte_years

        return SVIParams(
            a=a,
            b=0.12 + rng.uniform(-0.02, 0.02),
            rho=rho,
            m=rng.uniform(-0.01, 0.01),  # Small horizontal shift
            sigma=0.08 + rng.uniform(-0.02, 0.02),
        )

    def _black_scholes_greeks(
        self,
        spot: float,
        strike: float,
        tte_years: float,
        vol: float,
        rate: float,
        div_yield: float,
        is_call: bool,
    ) -> dict[str, float]:
        """Calculate Black-Scholes Greeks."""
        from scipy.stats import norm

        if tte_years <= 0 or vol <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        forward = spot * math.exp((rate - div_yield) * tte_years)
        d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * tte_years) / (
            vol * math.sqrt(tte_years)
        )
        d2 = d1 - vol * math.sqrt(tte_years)

        # Standard normal CDF and PDF
        n_d1 = norm.cdf(d1)
        n_d2 = norm.cdf(d2)
        n_pdf_d1 = norm.pdf(d1)

        discount = math.exp(-rate * tte_years)
        div_discount = math.exp(-div_yield * tte_years)

        if is_call:
            delta = div_discount * n_d1
            theta = (
                -spot * div_discount * n_pdf_d1 * vol / (2 * math.sqrt(tte_years))
                - rate * strike * discount * n_d2
                + div_yield * spot * div_discount * n_d1
            )
            rho_greek = strike * tte_years * discount * n_d2
        else:
            delta = div_discount * (n_d1 - 1)
            theta = (
                -spot * div_discount * n_pdf_d1 * vol / (2 * math.sqrt(tte_years))
                + rate * strike * discount * (1 - n_d2)
                - div_yield * spot * div_discount * (1 - n_d1)
            )
            rho_greek = -strike * tte_years * discount * (1 - n_d2)

        gamma = div_discount * n_pdf_d1 / (spot * vol * math.sqrt(tte_years))
        vega = spot * div_discount * n_pdf_d1 * math.sqrt(tte_years)

        # Scale theta to per-day (divide by 365)
        theta = theta / 365

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega / 100,  # Per 1% vol move
            "rho": rho_greek / 100,  # Per 1% rate move
        }

    def _black_scholes_price(
        self,
        spot: float,
        strike: float,
        tte_years: float,
        vol: float,
        rate: float,
        div_yield: float,
        is_call: bool,
    ) -> float:
        """Calculate Black-Scholes option price."""
        from scipy.stats import norm

        if tte_years <= 0:
            # Intrinsic value
            if is_call:
                return max(0, spot - strike)
            return max(0, strike - spot)

        d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * tte_years) / (
            vol * math.sqrt(tte_years)
        )
        d2 = d1 - vol * math.sqrt(tte_years)

        discount = math.exp(-rate * tte_years)
        div_discount = math.exp(-div_yield * tte_years)

        if is_call:
            return spot * div_discount * norm.cdf(d1) - strike * discount * norm.cdf(d2)
        return strike * discount * norm.cdf(-d2) - spot * div_discount * norm.cdf(-d1)

    def generate_underlying_price(
        self,
        symbol: str,
        target_date: date,
        previous_close: Optional[float] = None,
    ) -> UnderlyingPrice:
        """Generate underlying OHLCV data for a date."""
        rng = self._get_date_rng(target_date)

        if previous_close is None:
            previous_close = self.base_price

        # Generate daily return with volatility
        daily_vol = self.base_volatility / math.sqrt(252)
        daily_return = rng.normal(0.0003, daily_vol)  # Slight positive drift

        close = previous_close * (1 + daily_return)

        # Generate OHLC with realistic intraday range
        intraday_vol = daily_vol * 1.5
        high = close * (1 + abs(rng.normal(0, intraday_vol)))
        low = close * (1 - abs(rng.normal(0, intraday_vol)))
        open_price = previous_close * (1 + rng.normal(0, daily_vol * 0.3))

        # Ensure consistency: low <= open, close <= high
        low = min(low, open_price, close)
        high = max(high, open_price, close)

        # Volume in millions of shares
        volume = int(rng.lognormal(17.5, 0.3))  # ~40-80M shares typical for SPY

        return UnderlyingPrice(
            symbol=symbol,
            date=target_date,
            open=Decimal(str(round(open_price, 2))),
            high=Decimal(str(round(high, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(close, 2))),
            volume=volume,
        )

    def generate_options_chain(
        self,
        symbol: str,
        quote_date: date,
        underlying_price: Optional[float] = None,
    ) -> OptionsChain:
        """
        Generate a complete options chain for a date.

        Creates realistic expirations, strikes, and prices using SVI model.
        """
        rng = self._get_date_rng(quote_date)

        if underlying_price is None:
            underlying_price = self.base_price

        spot = underlying_price
        quotes: list[OptionQuote] = []

        # Generate expirations: weekly for next 2 months, monthly after
        expirations = self._generate_expirations(quote_date)

        for exp_date in expirations:
            dte = (exp_date - quote_date).days
            tte_years = dte / 252  # Trading days

            if tte_years <= 0:
                continue

            # Get SVI params for this expiration
            svi = self._generate_svi_params(tte_years, quote_date)

            # Calculate forward price using put-call parity approximation
            forward = spot * math.exp((self.risk_free_rate - self.dividend_yield) * tte_years)

            # Generate strikes around ATM
            strikes = self._generate_strikes(spot, tte_years)

            for strike in strikes:
                # Calculate log-moneyness
                log_money = math.log(strike / forward)

                # Get IV from SVI model
                try:
                    iv = svi.implied_vol(log_money, tte_years)
                    iv = max(0.05, min(1.0, iv))  # Clamp to reasonable range
                except (ValueError, ZeroDivisionError):
                    iv = self.base_volatility

                # Generate both call and put
                for opt_type in [OptionType.CALL, OptionType.PUT]:
                    is_call = opt_type == OptionType.CALL

                    # Calculate theoretical price
                    theo_price = self._black_scholes_price(
                        spot,
                        strike,
                        tte_years,
                        iv,
                        self.risk_free_rate,
                        self.dividend_yield,
                        is_call,
                    )

                    if theo_price < 0.01:
                        continue  # Skip very cheap options

                    # Generate bid-ask spread (wider for OTM)
                    moneyness = spot / strike if is_call else strike / spot
                    base_spread = 0.02 + 0.08 * max(0, 1 - moneyness)
                    spread_pct = base_spread * (1 + rng.uniform(-0.2, 0.2))

                    mid = theo_price
                    half_spread = mid * spread_pct / 2
                    bid = max(0.01, mid - half_spread)
                    ask = mid + half_spread

                    # Calculate Greeks
                    greeks = self._black_scholes_greeks(
                        spot,
                        strike,
                        tte_years,
                        iv,
                        self.risk_free_rate,
                        self.dividend_yield,
                        is_call,
                    )

                    # Generate volume and OI (higher for ATM, liquid expirations)
                    atm_factor = math.exp(-2 * log_money**2)
                    liquidity_factor = math.exp(-tte_years * 0.5)  # Less liquid further out
                    base_oi = 5000 * atm_factor * liquidity_factor
                    base_vol = base_oi * 0.1 * liquidity_factor

                    open_interest = max(0, int(rng.lognormal(math.log(max(1, base_oi)), 1.0)))
                    volume = max(0, int(rng.lognormal(math.log(max(1, base_vol)), 1.5)))

                    quotes.append(
                        OptionQuote(
                            symbol=symbol,
                            quote_date=quote_date,
                            expiration_date=exp_date,
                            strike=Decimal(str(strike)),
                            option_type=opt_type,
                            bid=Decimal(str(round(bid, 2))),
                            ask=Decimal(str(round(ask, 2))),
                            mid=Decimal(str(round(mid, 2))),
                            last=Decimal(str(round(mid + rng.uniform(-half_spread, half_spread), 2))),
                            volume=volume,
                            open_interest=open_interest,
                            delta=Decimal(str(round(greeks["delta"], 6))),
                            gamma=Decimal(str(round(greeks["gamma"], 6))),
                            theta=Decimal(str(round(greeks["theta"], 6))),
                            vega=Decimal(str(round(greeks["vega"], 6))),
                            rho=Decimal(str(round(greeks["rho"], 6))),
                            implied_volatility=Decimal(str(round(iv, 6))),
                            underlying_price=Decimal(str(round(spot, 2))),
                        )
                    )

        logger.debug(f"Generated {len(quotes)} option quotes for {symbol} on {quote_date}")

        return OptionsChain(
            symbol=symbol,
            quote_date=quote_date,
            underlying_price=Decimal(str(round(spot, 2))),
            quotes=quotes,
        )

    def _generate_expirations(self, quote_date: date) -> list[date]:
        """Generate realistic expiration dates."""
        expirations = []

        # Find next Friday (weekly expirations)
        days_until_friday = (4 - quote_date.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        next_friday = quote_date + timedelta(days=days_until_friday)

        # Weekly expirations for next 8 weeks
        for i in range(8):
            exp = next_friday + timedelta(weeks=i)
            if exp > quote_date:
                expirations.append(exp)

        # Monthly expirations (3rd Friday) for next 6 months
        current = quote_date.replace(day=1)
        for _ in range(12):
            # Find 3rd Friday
            first_day = current.replace(day=1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

            if third_friday > quote_date and third_friday not in expirations:
                expirations.append(third_friday)

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Add some quarterly expirations (LEAPs)
        for months_out in [12, 18, 24]:
            leap_date = quote_date + timedelta(days=months_out * 30)
            # Find nearest 3rd Friday
            first_day = leap_date.replace(day=1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            if third_friday not in expirations:
                expirations.append(third_friday)

        return sorted(set(expirations))

    def _generate_strikes(self, spot: float, tte_years: float) -> list[float]:
        """Generate strike prices based on spot and time to expiry."""
        # Strike spacing: $1 for near-term, $5 for far-term
        if tte_years < 0.1:  # < ~25 days
            spacing = 1.0
            range_pct = 0.15
        elif tte_years < 0.25:  # < ~63 days
            spacing = 1.0
            range_pct = 0.20
        elif tte_years < 0.5:  # < ~126 days
            spacing = 2.0
            range_pct = 0.25
        else:
            spacing = 5.0
            range_pct = 0.30

        # Generate strikes around spot
        min_strike = spot * (1 - range_pct)
        max_strike = spot * (1 + range_pct)

        # Round to nearest spacing
        min_strike = math.floor(min_strike / spacing) * spacing
        max_strike = math.ceil(max_strike / spacing) * spacing

        strikes = []
        current = min_strike
        while current <= max_strike:
            strikes.append(current)
            current += spacing

        return strikes
