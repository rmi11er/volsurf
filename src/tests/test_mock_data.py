"""Tests for mock data generator."""

from datetime import date
from decimal import Decimal

import pytest

from volsurf.ingestion.mock_data import MockDataGenerator
from volsurf.models.schemas import OptionType, SVIParams


class TestSVIParams:
    """Tests for SVI parameter model."""

    def test_total_variance_at_atm(self):
        """Total variance at ATM should equal 'a' parameter (when m=0)."""
        svi = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.1)
        # At k=0 (ATM), w(0) = a + b * sqrt(sigma^2) = a + b*sigma
        w_atm = svi.total_variance(0.0)
        expected = 0.04 + 0.15 * 0.1  # a + b*sigma
        assert abs(w_atm - expected) < 1e-10

    def test_total_variance_put_wing(self):
        """Total variance should be higher for OTM puts (negative log-moneyness)."""
        svi = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.1)
        w_atm = svi.total_variance(0.0)
        w_otm_put = svi.total_variance(-0.2)  # OTM put (K > F)
        # With negative rho, OTM puts should have higher variance
        assert w_otm_put > w_atm

    def test_implied_vol_positive(self):
        """Implied vol should always be positive for valid parameters."""
        svi = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.1)
        for k in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            iv = svi.implied_vol(k, tte_years=0.25)
            assert iv > 0, f"IV should be positive for k={k}"

    def test_implied_vol_scaling_with_tte(self):
        """Implied vol should scale approximately with sqrt(1/T) for fixed variance."""
        svi = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
        # With b=0, variance is constant = a
        iv_short = svi.implied_vol(0.0, tte_years=0.1)
        iv_long = svi.implied_vol(0.0, tte_years=0.4)
        # IV ratio should be sqrt(0.4/0.1) = 2
        assert abs(iv_short / iv_long - 2.0) < 0.1


class TestMockDataGenerator:
    """Tests for mock data generator."""

    @pytest.fixture
    def generator(self):
        """Create a mock data generator."""
        return MockDataGenerator(
            base_price=590.0,
            base_volatility=0.18,
            risk_free_rate=0.045,
            dividend_yield=0.013,
        )

    def test_generate_underlying_price(self, generator):
        """Test underlying price generation."""
        price = generator.generate_underlying_price("SPY", date(2024, 12, 13))

        assert price.symbol == "SPY"
        assert price.date == date(2024, 12, 13)
        assert float(price.open) > 0
        assert float(price.high) >= float(price.low)
        assert float(price.high) >= float(price.close)
        assert float(price.low) <= float(price.close)
        assert price.volume > 0

    def test_underlying_price_reproducibility(self, generator):
        """Same date should produce same price (deterministic)."""
        price1 = generator.generate_underlying_price("SPY", date(2024, 12, 13))
        price2 = generator.generate_underlying_price("SPY", date(2024, 12, 13))

        assert price1.close == price2.close
        assert price1.volume == price2.volume

    def test_generate_options_chain(self, generator):
        """Test options chain generation."""
        chain = generator.generate_options_chain("SPY", date(2024, 12, 13))

        assert chain.symbol == "SPY"
        assert chain.quote_date == date(2024, 12, 13)
        assert len(chain.quotes) > 0

        # Check that we have both calls and puts
        calls = [q for q in chain.quotes if q.option_type == OptionType.CALL]
        puts = [q for q in chain.quotes if q.option_type == OptionType.PUT]
        assert len(calls) > 0
        assert len(puts) > 0

    def test_options_chain_prices_positive(self, generator):
        """All option prices should be positive."""
        chain = generator.generate_options_chain("SPY", date(2024, 12, 13))

        for quote in chain.quotes:
            assert quote.bid is not None and float(quote.bid) > 0
            assert quote.ask is not None and float(quote.ask) > float(quote.bid)
            assert quote.mid is not None and float(quote.mid) > 0

    def test_options_chain_greeks(self, generator):
        """Greeks should be present and reasonable."""
        chain = generator.generate_options_chain("SPY", date(2024, 12, 13))

        for quote in chain.quotes:
            # Delta should be in [-1, 1]
            if quote.delta is not None:
                delta = float(quote.delta)
                if quote.option_type == OptionType.CALL:
                    assert 0 <= delta <= 1
                else:
                    assert -1 <= delta <= 0

            # Gamma should be positive
            if quote.gamma is not None:
                assert float(quote.gamma) >= 0

            # Vega should be positive
            if quote.vega is not None:
                assert float(quote.vega) >= 0

    def test_options_chain_multiple_expirations(self, generator):
        """Chain should have multiple expirations."""
        chain = generator.generate_options_chain("SPY", date(2024, 12, 13))

        expirations = set(q.expiration_date for q in chain.quotes)
        assert len(expirations) > 5, "Should have multiple expirations"

    def test_implied_vol_smile(self, generator):
        """Test that IVs form a smile pattern (higher for OTM options)."""
        chain = generator.generate_options_chain("SPY", date(2024, 12, 13))
        spot = float(chain.underlying_price)

        # Get calls for a specific expiration
        exp = sorted(set(q.expiration_date for q in chain.quotes))[2]  # 3rd expiration
        exp_calls = [q for q in chain.quotes if q.expiration_date == exp and q.option_type == OptionType.CALL]

        if len(exp_calls) < 5:
            pytest.skip("Not enough strikes for smile test")

        # Sort by strike
        exp_calls.sort(key=lambda q: float(q.strike))

        # Find ATM and OTM options
        atm_call = min(exp_calls, key=lambda q: abs(float(q.strike) - spot))
        otm_calls = [q for q in exp_calls if float(q.strike) > spot * 1.05]

        if atm_call.implied_volatility and otm_calls:
            atm_iv = float(atm_call.implied_volatility)
            # At least some OTM calls should have higher IV (smile)
            higher_iv_count = sum(
                1 for q in otm_calls
                if q.implied_volatility and float(q.implied_volatility) > atm_iv
            )
            # Allow some flexibility due to noise
            assert higher_iv_count >= len(otm_calls) * 0.3
