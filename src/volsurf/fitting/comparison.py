"""Model comparison framework for volatility surface models.

Provides tools to compare different volatility models (SVI, SABR, Polynomial)
on the same data, enabling model selection and validation.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection
from volsurf.fitting.svi import fit_svi_slice, svi_implied_vol, SVIFitResult
from volsurf.fitting.sabr import fit_sabr_slice, sabr_implied_vol_vectorized, SABRFitResult
from volsurf.fitting.polynomial import fit_polynomial_slice, polynomial_implied_vol, PolynomialFitResult


class ModelType(str, Enum):
    """Available volatility surface models."""
    SVI = "SVI"
    SABR = "SABR"
    POLYNOMIAL = "POLYNOMIAL"


@dataclass
class ModelComparisonResult:
    """Comparison result for a single expiration."""

    expiration_date: date
    tte_years: float
    num_points: int
    forward_price: float

    # Model results
    svi_result: Optional[SVIFitResult] = None
    sabr_result: Optional[SABRFitResult] = None
    polynomial_result: Optional[PolynomialFitResult] = None

    # Best model selection
    best_model: Optional[ModelType] = None
    best_rmse: Optional[float] = None

    def get_all_rmse(self) -> Dict[ModelType, float]:
        """Get RMSE for all successfully fitted models."""
        results = {}
        if self.svi_result and self.svi_result.success:
            results[ModelType.SVI] = self.svi_result.rmse
        if self.sabr_result and self.sabr_result.success:
            results[ModelType.SABR] = self.sabr_result.rmse
        if self.polynomial_result and self.polynomial_result.success:
            results[ModelType.POLYNOMIAL] = self.polynomial_result.rmse
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "tte_years": self.tte_years,
            "num_points": self.num_points,
            "forward_price": self.forward_price,
            "svi_rmse": self.svi_result.rmse if self.svi_result else None,
            "sabr_rmse": self.sabr_result.rmse if self.sabr_result else None,
            "poly_rmse": self.polynomial_result.rmse if self.polynomial_result else None,
            "best_model": self.best_model.value if self.best_model else None,
            "best_rmse": self.best_rmse,
        }


@dataclass
class FullComparisonResult:
    """Full comparison across all expirations for a date."""

    symbol: str
    quote_date: date
    expiration_results: List[ModelComparisonResult]

    # Aggregate statistics
    avg_svi_rmse: Optional[float] = None
    avg_sabr_rmse: Optional[float] = None
    avg_poly_rmse: Optional[float] = None
    overall_best_model: Optional[ModelType] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.expiration_results])


class ModelComparator:
    """Compare multiple volatility surface models."""

    def __init__(
        self,
        models: Optional[List[ModelType]] = None,
        polynomial_degree: int = 4,
        sabr_beta: float = 0.5,
    ):
        """
        Initialize model comparator.

        Args:
            models: List of models to compare. Default: all models.
            polynomial_degree: Degree for polynomial fit.
            sabr_beta: Beta parameter for SABR (often fixed).
        """
        self.models = models or [ModelType.SVI, ModelType.SABR, ModelType.POLYNOMIAL]
        self.polynomial_degree = polynomial_degree
        self.sabr_beta = sabr_beta

    def compare_slice(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        forward_price: float,
        tte_years: float,
        expiration_date: date,
        weights: Optional[np.ndarray] = None,
    ) -> ModelComparisonResult:
        """
        Compare models on a single expiration slice.

        Args:
            strikes: Strike prices
            implied_vols: Implied volatilities
            forward_price: Forward price
            tte_years: Time to expiration in years
            expiration_date: Expiration date
            weights: Optional weights for fitting

        Returns:
            ModelComparisonResult with all model fits
        """
        result = ModelComparisonResult(
            expiration_date=expiration_date,
            tte_years=tte_years,
            num_points=len(strikes),
            forward_price=forward_price,
        )

        # Fit each model
        if ModelType.SVI in self.models:
            try:
                result.svi_result = fit_svi_slice(
                    strikes, implied_vols, forward_price, tte_years, weights
                )
            except Exception as e:
                logger.warning(f"SVI fit failed: {e}")

        if ModelType.SABR in self.models:
            try:
                result.sabr_result = fit_sabr_slice(
                    strikes, implied_vols, forward_price, tte_years,
                    beta=self.sabr_beta, weights=weights
                )
            except Exception as e:
                logger.warning(f"SABR fit failed: {e}")

        if ModelType.POLYNOMIAL in self.models:
            try:
                result.polynomial_result = fit_polynomial_slice(
                    strikes, implied_vols, forward_price, tte_years,
                    degree=self.polynomial_degree, weights=weights
                )
            except Exception as e:
                logger.warning(f"Polynomial fit failed: {e}")

        # Select best model
        rmse_dict = result.get_all_rmse()
        if rmse_dict:
            result.best_model = min(rmse_dict, key=rmse_dict.get)
            result.best_rmse = rmse_dict[result.best_model]

        return result

    def compare_date(
        self,
        symbol: str,
        quote_date: date,
    ) -> FullComparisonResult:
        """
        Compare models across all expirations for a given date.

        Args:
            symbol: Ticker symbol
            quote_date: Quote date

        Returns:
            FullComparisonResult with all expiration comparisons
        """
        conn = get_connection()

        # Get liquid options data
        query = """
            SELECT
                expiration_date,
                strike,
                implied_volatility,
                underlying_price
            FROM raw_options_chains
            WHERE symbol = ?
              AND quote_date = ?
              AND is_liquid = TRUE
              AND implied_volatility IS NOT NULL
              AND implied_volatility > 0
            ORDER BY expiration_date, strike
        """
        df = conn.execute(query, [symbol, quote_date]).fetchdf()

        if df.empty:
            logger.warning(f"No liquid options data for {symbol} on {quote_date}")
            return FullComparisonResult(
                symbol=symbol,
                quote_date=quote_date,
                expiration_results=[],
            )

        underlying_price = df["underlying_price"].iloc[0]

        # Group by expiration and fit each
        expiration_results = []

        for exp_date, group in df.groupby("expiration_date"):
            # Convert to proper types
            if hasattr(exp_date, "date"):
                exp_date = exp_date.date()

            strikes = group["strike"].values.astype(float)
            ivs = group["implied_volatility"].values.astype(float)

            # Calculate TTE
            tte_days = (exp_date - quote_date).days
            tte_years = tte_days / 365.0

            if tte_years <= 0 or len(strikes) < 5:
                continue

            # Estimate forward price
            forward = underlying_price * np.exp(0.035 * tte_years)  # Simple estimate

            result = self.compare_slice(
                strikes, ivs, forward, tte_years, exp_date
            )
            expiration_results.append(result)

        # Calculate aggregate statistics
        full_result = FullComparisonResult(
            symbol=symbol,
            quote_date=quote_date,
            expiration_results=expiration_results,
        )

        svi_rmse = [r.svi_result.rmse for r in expiration_results
                    if r.svi_result and r.svi_result.success]
        sabr_rmse = [r.sabr_result.rmse for r in expiration_results
                    if r.sabr_result and r.sabr_result.success]
        poly_rmse = [r.polynomial_result.rmse for r in expiration_results
                    if r.polynomial_result and r.polynomial_result.success]

        if svi_rmse:
            full_result.avg_svi_rmse = np.mean(svi_rmse)
        if sabr_rmse:
            full_result.avg_sabr_rmse = np.mean(sabr_rmse)
        if poly_rmse:
            full_result.avg_poly_rmse = np.mean(poly_rmse)

        # Select overall best model
        avg_rmse = {}
        if full_result.avg_svi_rmse:
            avg_rmse[ModelType.SVI] = full_result.avg_svi_rmse
        if full_result.avg_sabr_rmse:
            avg_rmse[ModelType.SABR] = full_result.avg_sabr_rmse
        if full_result.avg_poly_rmse:
            avg_rmse[ModelType.POLYNOMIAL] = full_result.avg_poly_rmse

        if avg_rmse:
            full_result.overall_best_model = min(avg_rmse, key=avg_rmse.get)

        return full_result

    def generate_smile_comparison(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        forward_price: float,
        tte_years: float,
        comparison_result: ModelComparisonResult,
        num_points: int = 100,
    ) -> pd.DataFrame:
        """
        Generate fitted smile data for visualization.

        Returns DataFrame with strike, market_iv, and fitted IVs for each model.
        """
        # Create fine grid of strikes for smooth curves
        k_min = np.log(strikes.min() / forward_price)
        k_max = np.log(strikes.max() / forward_price)
        k_grid = np.linspace(k_min, k_max, num_points)
        strike_grid = forward_price * np.exp(k_grid)

        data = {
            "strike": strike_grid,
            "log_moneyness": k_grid,
        }

        # Add fitted curves
        if comparison_result.svi_result and comparison_result.svi_result.success:
            params = comparison_result.svi_result.params
            data["svi_iv"] = svi_implied_vol(
                k_grid, tte_years, params.a, params.b, params.rho, params.m, params.sigma
            )

        if comparison_result.sabr_result and comparison_result.sabr_result.success:
            params = comparison_result.sabr_result.params
            data["sabr_iv"] = sabr_implied_vol_vectorized(
                forward_price, strike_grid, tte_years,
                params.alpha, params.beta, params.rho, params.nu
            )

        if comparison_result.polynomial_result and comparison_result.polynomial_result.success:
            params = comparison_result.polynomial_result.params
            data["poly_iv"] = polynomial_implied_vol(k_grid, params.coefficients)

        return pd.DataFrame(data)


def print_comparison_summary(result: FullComparisonResult) -> None:
    """Print a formatted comparison summary."""
    print(f"\n{'='*60}")
    print(f"Model Comparison: {result.symbol} on {result.quote_date}")
    print(f"{'='*60}")
    print(f"Expirations compared: {len(result.expiration_results)}")
    print()

    # Average RMSE by model
    print("Average RMSE by Model:")
    print("-" * 40)
    if result.avg_svi_rmse:
        marker = " <-- BEST" if result.overall_best_model == ModelType.SVI else ""
        print(f"  SVI:        {result.avg_svi_rmse:.4%}{marker}")
    if result.avg_sabr_rmse:
        marker = " <-- BEST" if result.overall_best_model == ModelType.SABR else ""
        print(f"  SABR:       {result.avg_sabr_rmse:.4%}{marker}")
    if result.avg_poly_rmse:
        marker = " <-- BEST" if result.overall_best_model == ModelType.POLYNOMIAL else ""
        print(f"  Polynomial: {result.avg_poly_rmse:.4%}{marker}")
    print()

    # Per-expiration results
    print("Per-Expiration Results:")
    print("-" * 60)
    print(f"{'Expiration':<12} {'DTE':>6} {'SVI':>10} {'SABR':>10} {'Poly':>10} {'Best':>8}")
    print("-" * 60)

    for r in result.expiration_results:
        dte = int(r.tte_years * 365)
        svi_str = f"{r.svi_result.rmse:.4%}" if r.svi_result and r.svi_result.success else "N/A"
        sabr_str = f"{r.sabr_result.rmse:.4%}" if r.sabr_result and r.sabr_result.success else "N/A"
        poly_str = f"{r.polynomial_result.rmse:.4%}" if r.polynomial_result and r.polynomial_result.success else "N/A"
        best_str = r.best_model.value if r.best_model else "N/A"
        print(f"{str(r.expiration_date):<12} {dte:>6} {svi_str:>10} {sabr_str:>10} {poly_str:>10} {best_str:>8}")

    print("=" * 60)
