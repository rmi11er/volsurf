"""Surface fitting pipeline - orchestrates SVI fitting for options data."""

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from volsurf.database.connection import get_connection
from volsurf.fitting.arbitrage import validate_surface, validate_term_structure
from volsurf.fitting.implied_vol import (
    calculate_forward_from_options,
    calculate_iv_from_mid_prices,
)
from volsurf.fitting.svi import (
    SVIFitResult,
    compute_vega_weights,
    estimate_forward_price,
    fit_svi_slice,
)
from volsurf.models.schemas import FittedSurface, SVIParams


@dataclass
class PipelineConfig:
    """Configuration for the fitting pipeline."""

    # Minimum data requirements
    min_strikes_per_expiration: int = 10
    min_tte_days: int = 1  # Skip same-day expiration
    max_tte_days: int = 365  # Skip very long-dated options

    # Filtering
    moneyness_min: float = 0.7  # Min strike/spot ratio
    moneyness_max: float = 1.3  # Max strike/spot ratio

    # Fitting parameters
    use_vega_weights: bool = True
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.015

    # Quality thresholds
    max_rmse: float = 0.02  # 2% RMSE threshold for warnings


@dataclass
class FitStats:
    """Statistics from a fitting run."""

    symbol: str
    quote_date: date
    expirations_processed: int
    expirations_skipped: int
    total_points_fitted: int
    avg_rmse: float
    arbitrage_violations: int


class SurfaceFittingPipeline:
    """Pipeline for fitting SVI surfaces to options data."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.conn = get_connection()

    def fit_date(
        self,
        symbol: str,
        quote_date: date,
        store_results: bool = True,
    ) -> Tuple[List[FittedSurface], FitStats]:
        """
        Fit SVI surfaces for all expirations on a given date.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            quote_date: Date to fit
            store_results: Whether to store results in database

        Returns:
            (list of FittedSurface objects, FitStats)
        """
        # Ensure quote_date is a date object
        if hasattr(quote_date, 'date'):
            quote_date = quote_date.date()

        logger.info(f"Fitting surfaces for {symbol} on {quote_date}")

        # Query liquid options from database
        options_df = self._query_options(symbol, quote_date)

        if options_df.empty:
            logger.warning(f"No liquid options found for {symbol} on {quote_date}")
            return [], FitStats(
                symbol=symbol,
                quote_date=quote_date,
                expirations_processed=0,
                expirations_skipped=0,
                total_points_fitted=0,
                avg_rmse=0.0,
                arbitrage_violations=0,
            )

        # Get underlying price
        underlying_price = float(options_df["underlying_price"].iloc[0])
        logger.debug(f"Underlying price: {underlying_price}")

        # Get unique expirations (convert to date if needed)
        expirations_raw = options_df["expiration_date"].unique()
        expirations = sorted([
            exp.date() if hasattr(exp, 'date') else exp
            for exp in expirations_raw
        ])
        logger.info(f"Found {len(expirations)} expirations to process")

        surfaces: List[FittedSurface] = []
        skipped = 0
        total_violations = 0

        for exp_date in expirations:
            # Filter to this expiration (handle both date and Timestamp)
            exp_df = options_df[
                options_df["expiration_date"].apply(
                    lambda x: x.date() if hasattr(x, 'date') else x
                ) == exp_date
            ].copy()

            # Calculate TTE
            tte_days = (exp_date - quote_date).days
            tte_years = tte_days / 365.0

            # Skip invalid TTEs
            if tte_days < self.config.min_tte_days:
                logger.debug(f"Skipping {exp_date}: TTE too short ({tte_days} days)")
                skipped += 1
                continue

            if tte_days > self.config.max_tte_days:
                logger.debug(f"Skipping {exp_date}: TTE too long ({tte_days} days)")
                skipped += 1
                continue

            # Skip if not enough strikes
            if len(exp_df) < self.config.min_strikes_per_expiration:
                logger.debug(
                    f"Skipping {exp_date}: only {len(exp_df)} strikes "
                    f"(need {self.config.min_strikes_per_expiration})"
                )
                skipped += 1
                continue

            # Fit this expiration
            surface = self._fit_expiration(
                exp_df, symbol, quote_date, exp_date, tte_years, underlying_price
            )

            if surface is not None:
                surfaces.append(surface)
                if not surface.passes_no_arbitrage:
                    total_violations += (
                        surface.butterfly_violations + surface.calendar_violations
                    )

        # Check calendar arbitrage across term structure
        if len(surfaces) >= 2:
            term_surfaces = [
                (s.tte_years, s.svi_params) for s in surfaces
            ]
            cal_result = validate_term_structure(term_surfaces)
            if not cal_result.passes:
                logger.warning(
                    f"Calendar arbitrage detected: {cal_result.calendar_violations} violations"
                )
                total_violations += cal_result.calendar_violations

        # Compute stats
        avg_rmse = np.mean([s.rmse for s in surfaces]) if surfaces else 0.0
        total_points = sum(s.num_points for s in surfaces)

        stats = FitStats(
            symbol=symbol,
            quote_date=quote_date,
            expirations_processed=len(surfaces),
            expirations_skipped=skipped,
            total_points_fitted=total_points,
            avg_rmse=avg_rmse,
            arbitrage_violations=total_violations,
        )

        logger.info(
            f"Fitted {stats.expirations_processed} expirations, "
            f"skipped {stats.expirations_skipped}, "
            f"avg RMSE: {stats.avg_rmse:.4f}"
        )

        # Store results
        if store_results and surfaces:
            self._store_surfaces(surfaces)

        return surfaces, stats

    def _fit_expiration(
        self,
        exp_df: pd.DataFrame,
        symbol: str,
        quote_date: date,
        exp_date: date,
        tte_years: float,
        underlying_price: float,
    ) -> Optional[FittedSurface]:
        """Fit SVI to a single expiration."""
        logger.debug(f"Fitting expiration {exp_date} (TTE: {tte_years:.4f}y)")

        # Separate calls and puts
        calls = exp_df[exp_df["option_type"] == "CALL"]
        puts = exp_df[exp_df["option_type"] == "PUT"]

        # Calculate forward price using put-call parity
        if len(calls) > 0 and len(puts) > 0:
            # Find strikes with both calls and puts
            common_strikes = set(calls["strike"].values) & set(puts["strike"].values)
            if len(common_strikes) >= 3:
                common_strikes = np.array(sorted(common_strikes))
                call_mids = np.array(
                    [calls[calls["strike"] == k]["mid"].values[0] for k in common_strikes]
                )
                put_mids = np.array(
                    [puts[puts["strike"] == k]["mid"].values[0] for k in common_strikes]
                )
                forward = calculate_forward_from_options(
                    call_mids,
                    put_mids,
                    common_strikes,
                    underlying_price,
                    self.config.risk_free_rate,
                    tte_years,
                )
            else:
                forward = estimate_forward_price(
                    underlying_price,
                    tte_years,
                    self.config.risk_free_rate,
                    self.config.dividend_yield,
                )
        else:
            forward = estimate_forward_price(
                underlying_price,
                tte_years,
                self.config.risk_free_rate,
                self.config.dividend_yield,
            )

        logger.debug(f"Forward price: {forward:.2f}")

        # Prepare data for IV calculation
        # Use OTM options: calls for K > F, puts for K < F
        strikes = exp_df["strike"].values.astype(float)
        mids = exp_df["mid"].values.astype(float)
        is_call = exp_df["option_type"].values == "CALL"

        # Filter to OTM options (more liquid, better for fitting)
        otm_mask = np.where(strikes >= forward, is_call, ~is_call)
        strikes_otm = strikes[otm_mask]
        mids_otm = mids[otm_mask]
        is_call_otm = is_call[otm_mask]

        if len(strikes_otm) < self.config.min_strikes_per_expiration:
            # Fall back to all options
            strikes_otm = strikes
            mids_otm = mids
            is_call_otm = is_call

        # Check for pre-computed IV or calculate from prices
        if "implied_volatility" in exp_df.columns:
            ivs_raw = exp_df["implied_volatility"].values[otm_mask if len(strikes_otm) == sum(otm_mask) else slice(None)]
            # Handle NaN values - recalculate if needed
            if np.any(pd.isna(ivs_raw)):
                ivs, valid = calculate_iv_from_mid_prices(
                    strikes_otm,
                    mids_otm,
                    is_call_otm,
                    underlying_price,
                    tte_years,
                    self.config.risk_free_rate,
                )
            else:
                ivs = ivs_raw.astype(float)
                valid = np.ones(len(ivs), dtype=bool)
        else:
            ivs, valid = calculate_iv_from_mid_prices(
                strikes_otm,
                mids_otm,
                is_call_otm,
                underlying_price,
                tte_years,
                self.config.risk_free_rate,
            )

        # Filter to valid IVs
        strikes_valid = strikes_otm[valid]
        ivs_valid = ivs[valid]

        if len(strikes_valid) < self.config.min_strikes_per_expiration:
            logger.warning(
                f"Not enough valid IVs for {exp_date}: {len(strikes_valid)} points"
            )
            return None

        # Compute weights
        if self.config.use_vega_weights:
            weights = compute_vega_weights(strikes_valid, forward, ivs_valid, tte_years)
        else:
            weights = None

        # Fit SVI
        fit_result: SVIFitResult = fit_svi_slice(
            strikes_valid,
            ivs_valid,
            forward,
            tte_years,
            weights=weights,
        )

        if not fit_result.success:
            logger.warning(f"SVI fit failed for {exp_date}: {fit_result.message}")
            # Still continue with best-effort params

        if fit_result.rmse > self.config.max_rmse:
            logger.warning(
                f"High RMSE for {exp_date}: {fit_result.rmse:.4f} > {self.config.max_rmse}"
            )

        # Validate arbitrage
        arb_result = validate_surface(fit_result.params, tte_years)

        # Calculate ATM vol
        atm_vol = fit_result.params.implied_vol(0.0, tte_years)

        # Calculate 25-delta skew (approximate)
        # 25-delta call is roughly at k ≈ 0.67 * sigma * sqrt(T)
        # 25-delta put is roughly at k ≈ -0.67 * sigma * sqrt(T)
        skew_width = 0.67 * atm_vol * math.sqrt(tte_years)
        iv_25d_call = fit_result.params.implied_vol(skew_width, tte_years)
        iv_25d_put = fit_result.params.implied_vol(-skew_width, tte_years)
        skew_25delta = iv_25d_put - iv_25d_call

        return FittedSurface(
            symbol=symbol,
            quote_date=quote_date,
            expiration_date=exp_date,
            tte_years=tte_years,
            forward_price=Decimal(str(round(forward, 4))),
            svi_params=fit_result.params,
            atm_vol=atm_vol,
            skew_25delta=skew_25delta,
            rmse=fit_result.rmse,
            mae=fit_result.mae,
            max_error=fit_result.max_error,
            num_points=fit_result.num_points,
            passes_no_arbitrage=arb_result.passes,
            butterfly_violations=arb_result.butterfly_violations,
            calendar_violations=arb_result.calendar_violations,
        )

    def _query_options(self, symbol: str, quote_date: date) -> pd.DataFrame:
        """Query liquid options from database."""
        query = """
        SELECT
            symbol,
            quote_date,
            expiration_date,
            strike,
            option_type,
            bid,
            ask,
            mid,
            volume,
            open_interest,
            implied_volatility,
            underlying_price,
            is_liquid
        FROM raw_options_chains
        WHERE symbol = ?
          AND quote_date = ?
          AND is_liquid = true
          AND mid > 0
          AND bid > 0
          AND ask > 0
        ORDER BY expiration_date, strike
        """
        result = self.conn.execute(query, [symbol, quote_date]).fetchdf()
        return result

    def _store_surfaces(self, surfaces: List[FittedSurface]) -> None:
        """Store fitted surfaces in database."""
        logger.info(f"Storing {len(surfaces)} fitted surfaces")

        for surface in surfaces:
            # Get next ID
            surface_id = self.conn.execute(
                "SELECT nextval('seq_surface_id')"
            ).fetchone()[0]

            # Insert surface
            insert_sql = """
            INSERT INTO fitted_surfaces (
                surface_id, symbol, quote_date, expiration_date,
                tte_years, forward_price,
                svi_a, svi_b, svi_rho, svi_m, svi_sigma,
                atm_vol, skew_25delta,
                rmse, mae, max_error, num_points,
                passes_no_arbitrage, butterfly_arbitrage_violations,
                calendar_arbitrage_violations,
                model_type, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, quote_date, expiration_date, model_type)
            DO UPDATE SET
                tte_years = EXCLUDED.tte_years,
                forward_price = EXCLUDED.forward_price,
                svi_a = EXCLUDED.svi_a,
                svi_b = EXCLUDED.svi_b,
                svi_rho = EXCLUDED.svi_rho,
                svi_m = EXCLUDED.svi_m,
                svi_sigma = EXCLUDED.svi_sigma,
                atm_vol = EXCLUDED.atm_vol,
                skew_25delta = EXCLUDED.skew_25delta,
                rmse = EXCLUDED.rmse,
                mae = EXCLUDED.mae,
                max_error = EXCLUDED.max_error,
                num_points = EXCLUDED.num_points,
                passes_no_arbitrage = EXCLUDED.passes_no_arbitrage,
                butterfly_arbitrage_violations = EXCLUDED.butterfly_arbitrage_violations,
                calendar_arbitrage_violations = EXCLUDED.calendar_arbitrage_violations,
                fit_timestamp = now()
            """

            params = surface.svi_params
            self.conn.execute(
                insert_sql,
                [
                    surface_id,
                    surface.symbol,
                    surface.quote_date,
                    surface.expiration_date,
                    surface.tte_years,
                    float(surface.forward_price),
                    params.a,
                    params.b,
                    params.rho,
                    params.m,
                    params.sigma,
                    surface.atm_vol,
                    surface.skew_25delta,
                    surface.rmse,
                    surface.mae,
                    surface.max_error,
                    surface.num_points,
                    surface.passes_no_arbitrage,
                    surface.butterfly_violations,
                    surface.calendar_violations,
                    "SVI",
                    "1.0",
                ],
            )

        logger.info("Surfaces stored successfully")

    def batch_fit(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        store_results: bool = True,
    ) -> Dict[str, int]:
        """
        Batch fit surfaces for a date range.

        Args:
            symbol: Underlying symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            store_results: Whether to store results

        Returns:
            Summary statistics
        """
        logger.info(f"Batch fitting {symbol} from {start_date} to {end_date}")

        # Get available quote dates
        query = """
        SELECT DISTINCT quote_date
        FROM raw_options_chains
        WHERE symbol = ?
          AND quote_date >= ?
          AND quote_date <= ?
        ORDER BY quote_date
        """
        dates_df = self.conn.execute(
            query, [symbol, start_date, end_date]
        ).fetchdf()

        quote_dates_raw = dates_df["quote_date"].tolist()
        # Convert Timestamps to dates if needed
        quote_dates = [
            qd.date() if hasattr(qd, 'date') else qd
            for qd in quote_dates_raw
        ]
        logger.info(f"Found {len(quote_dates)} dates with data")

        total_surfaces = 0
        total_skipped = 0
        total_violations = 0
        dates_processed = 0

        for qd in quote_dates:
            try:
                _, stats = self.fit_date(symbol, qd, store_results=store_results)
                total_surfaces += stats.expirations_processed
                total_skipped += stats.expirations_skipped
                total_violations += stats.arbitrage_violations
                dates_processed += 1
            except Exception as e:
                logger.error(f"Error fitting {qd}: {e}")
                continue

        return {
            "dates_processed": dates_processed,
            "total_surfaces": total_surfaces,
            "total_skipped": total_skipped,
            "arbitrage_violations": total_violations,
        }


def fit_surfaces_for_date(
    symbol: str,
    quote_date: date,
    config: Optional[PipelineConfig] = None,
    store_results: bool = True,
) -> Tuple[List[FittedSurface], FitStats]:
    """
    Convenience function to fit surfaces for a single date.

    Args:
        symbol: Underlying symbol
        quote_date: Date to fit
        config: Optional pipeline configuration
        store_results: Whether to store results

    Returns:
        (list of FittedSurface, FitStats)
    """
    pipeline = SurfaceFittingPipeline(config)
    return pipeline.fit_date(symbol, quote_date, store_results)


def batch_fit_surfaces(
    symbol: str,
    start_date: date,
    end_date: date,
    config: Optional[PipelineConfig] = None,
    store_results: bool = True,
) -> Dict[str, int]:
    """
    Convenience function to batch fit surfaces.

    Args:
        symbol: Underlying symbol
        start_date: Start date
        end_date: End date
        config: Optional pipeline configuration
        store_results: Whether to store results

    Returns:
        Summary statistics dict
    """
    pipeline = SurfaceFittingPipeline(config)
    return pipeline.batch_fit(symbol, start_date, end_date, store_results)
