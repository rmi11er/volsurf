"""Intraday surface fitting pipeline - fits SVI surfaces from point-in-time quotes."""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger

from volsurf.config.settings import Settings, get_settings
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
from volsurf.ingestion.intraday_client import ThetaIntradayClient
from volsurf.ingestion.mock_data import MockIntradayGenerator
from volsurf.models.schemas import (
    IntradayFittedSurface,
    IntradaySurfaceResult,
    SVIParams,
)


@dataclass
class IntradayPipelineConfig:
    """Configuration for the intraday fitting pipeline."""

    # Minimum data requirements
    min_strikes_per_expiration: int = 10
    min_tte_days: int = 1  # Skip same-day expiration
    max_tte_days: int = 365  # Skip very long-dated options

    # Filtering
    moneyness_min: float = 0.7  # Min strike/spot ratio
    moneyness_max: float = 1.3  # Max strike/spot ratio
    max_bid_ask_spread_pct: float = 0.20  # Max spread as fraction of mid

    # Fitting parameters
    use_vega_weights: bool = True
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.015

    # Quality thresholds
    max_rmse: float = 0.02  # 2% RMSE threshold for warnings


class IntradaySurfacePipeline:
    """
    Pipeline for fitting SVI surfaces from intraday quotes.

    Fetches quotes at a specific timestamp using the Theta v3 API
    (or mock data for testing), then fits SVI surfaces for each
    expiration.
    """

    def __init__(
        self,
        config: Optional[IntradayPipelineConfig] = None,
        settings: Optional[Settings] = None,
        use_mock: bool = False,
    ):
        self.config = config or IntradayPipelineConfig()
        self.settings = settings or get_settings()
        self.use_mock = use_mock or self.settings.use_mock_data

        if self.use_mock:
            self._mock_generator = MockIntradayGenerator()
            self._client = None
        else:
            self._client = ThetaIntradayClient(self.settings)
            self._mock_generator = None

    def close(self) -> None:
        """Close any open connections."""
        if self._client is not None:
            self._client.close()

    def fit_at_time(
        self,
        symbol: str,
        target_date: date,
        time_of_day: time,
    ) -> IntradaySurfaceResult:
        """
        Fit SVI surfaces for all expirations at a specific timestamp.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            target_date: Date to fetch data for
            time_of_day: Time of day in America/New_York timezone

        Returns:
            IntradaySurfaceResult with fitted surfaces
        """
        timestamp = datetime.combine(target_date, time_of_day)
        logger.info(f"Fitting intraday surfaces for {symbol} at {timestamp}")

        # Fetch data
        if self.use_mock:
            chain_df = self._mock_generator.generate_intraday_chain(
                symbol, target_date, time_of_day
            )
            underlying_price = self._mock_generator.get_underlying_at_time(
                symbol, target_date, time_of_day
            )
        else:
            chain_df = self._client.get_options_at_time(
                symbol, target_date, time_of_day
            )
            underlying_price = self._client.get_underlying_at_time(
                symbol, target_date, time_of_day
            )
            if underlying_price is None:
                # Fall back to mid of ATM options
                underlying_price = self._estimate_underlying_from_chain(chain_df)

        if chain_df.is_empty():
            logger.warning(f"No options data found for {symbol} at {timestamp}")
            return IntradaySurfaceResult(
                symbol=symbol,
                timestamp=timestamp,
                underlying_price=underlying_price or Decimal("0"),
                surfaces=[],
                total_expirations=0,
                successful_fits=0,
                failed_fits=0,
            )

        underlying_price_float = float(underlying_price)
        logger.debug(f"Underlying price: {underlying_price_float}")

        # Apply liquidity filters
        chain_df = self._apply_filters(chain_df, underlying_price_float)

        if chain_df.is_empty():
            logger.warning(f"No liquid options after filtering for {symbol}")
            return IntradaySurfaceResult(
                symbol=symbol,
                timestamp=timestamp,
                underlying_price=underlying_price,
                surfaces=[],
                total_expirations=0,
                successful_fits=0,
                failed_fits=0,
            )

        # Get unique expirations
        expirations = sorted(chain_df["expiration_date"].unique().to_list())
        logger.info(f"Found {len(expirations)} expirations to process")

        surfaces: List[IntradayFittedSurface] = []
        failed = 0
        total_violations = 0

        for exp_date in expirations:
            # Filter to this expiration
            exp_df = chain_df.filter(pl.col("expiration_date") == exp_date)

            # Calculate TTE
            tte_days = (exp_date - target_date).days
            tte_years = tte_days / 365.0

            # Skip invalid TTEs
            if tte_days < self.config.min_tte_days:
                logger.debug(f"Skipping {exp_date}: TTE too short ({tte_days} days)")
                failed += 1
                continue

            if tte_days > self.config.max_tte_days:
                logger.debug(f"Skipping {exp_date}: TTE too long ({tte_days} days)")
                failed += 1
                continue

            # Skip if not enough strikes
            if len(exp_df) < self.config.min_strikes_per_expiration:
                logger.debug(
                    f"Skipping {exp_date}: only {len(exp_df)} strikes "
                    f"(need {self.config.min_strikes_per_expiration})"
                )
                failed += 1
                continue

            # Fit this expiration
            surface = self._fit_expiration(
                exp_df, symbol, timestamp, exp_date, tte_years, underlying_price_float
            )

            if surface is not None:
                surfaces.append(surface)
                if not surface.passes_no_arbitrage:
                    total_violations += (
                        surface.butterfly_violations + surface.calendar_violations
                    )
            else:
                failed += 1

        # Check calendar arbitrage across term structure
        if len(surfaces) >= 2:
            term_surfaces = [(s.tte_years, s.svi_params) for s in surfaces]
            cal_result = validate_term_structure(term_surfaces)
            if not cal_result.passes:
                logger.warning(
                    f"Calendar arbitrage detected: {cal_result.calendar_violations} violations"
                )

        # Compute average RMSE
        avg_rmse = np.mean([s.rmse for s in surfaces]) if surfaces else None

        result = IntradaySurfaceResult(
            symbol=symbol,
            timestamp=timestamp,
            underlying_price=underlying_price,
            surfaces=surfaces,
            total_expirations=len(expirations),
            successful_fits=len(surfaces),
            failed_fits=failed,
            avg_rmse=avg_rmse,
        )

        logger.info(
            f"Fitted {result.successful_fits}/{result.total_expirations} expirations, "
            f"avg RMSE: {result.avg_rmse:.4f}" if result.avg_rmse else "no fits"
        )

        return result

    def fit_time_series(
        self,
        symbol: str,
        target_date: date,
        start_time: time = time(9, 35),
        end_time: time = time(15, 55),
        interval_minutes: int = 30,
    ) -> List[IntradaySurfaceResult]:
        """
        Fit surfaces at regular intervals throughout the trading day.

        Args:
            symbol: Underlying symbol
            target_date: Date to process
            start_time: Start time (default 9:35 AM ET)
            end_time: End time (default 3:55 PM ET)
            interval_minutes: Interval between fits in minutes

        Returns:
            List of IntradaySurfaceResult for each timestamp
        """
        logger.info(
            f"Fitting time series for {symbol} on {target_date} "
            f"from {start_time} to {end_time} every {interval_minutes} min"
        )

        results = []
        current_time = start_time

        while True:
            # Check if we've passed end time
            if (current_time.hour > end_time.hour or
                (current_time.hour == end_time.hour and
                 current_time.minute > end_time.minute)):
                break

            try:
                result = self.fit_at_time(symbol, target_date, current_time)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to fit at {current_time}: {e}")

            # Advance time
            current_dt = datetime.combine(target_date, current_time)
            next_dt = current_dt + timedelta(minutes=interval_minutes)
            current_time = next_dt.time()

        logger.info(f"Completed {len(results)} time series fits")
        return results

    def _apply_filters(self, df: pl.DataFrame, underlying_price: float) -> pl.DataFrame:
        """Apply liquidity and moneyness filters."""
        # Calculate derived columns
        df = df.with_columns([
            (pl.lit(underlying_price) / pl.col("strike")).alias("moneyness"),
            pl.when(pl.col("mid") > 0)
            .then((pl.col("ask") - pl.col("bid")) / pl.col("mid"))
            .otherwise(pl.lit(1.0))
            .alias("spread_pct"),
        ])

        # Apply filters
        df = df.filter(
            (pl.col("bid") > 0) &
            (pl.col("ask") > pl.col("bid")) &
            (pl.col("mid") > 0) &
            (pl.col("spread_pct") <= self.config.max_bid_ask_spread_pct) &
            (pl.col("moneyness") >= self.config.moneyness_min) &
            (pl.col("moneyness") <= self.config.moneyness_max)
        )

        return df

    def _estimate_underlying_from_chain(self, df: pl.DataFrame) -> Decimal:
        """Estimate underlying price from options chain (mid of ATM strikes)."""
        if df.is_empty():
            return Decimal("0")

        # Find approximate ATM by looking at where call-put parity holds
        # For now, use the average of near-ATM strikes
        strikes = df["strike"].unique().sort()
        if len(strikes) == 0:
            return Decimal("0")

        # Use median strike as approximation
        median_strike = strikes[len(strikes) // 2]
        return Decimal(str(median_strike))

    def _fit_expiration(
        self,
        exp_df: pl.DataFrame,
        symbol: str,
        timestamp: datetime,
        exp_date: date,
        tte_years: float,
        underlying_price: float,
    ) -> Optional[IntradayFittedSurface]:
        """Fit SVI to a single expiration slice."""
        logger.debug(f"Fitting expiration {exp_date} (TTE: {tte_years:.4f}y)")

        # Convert to numpy for compatibility with existing fitting code
        df_pd = exp_df.to_pandas()

        # Separate calls and puts
        calls = df_pd[df_pd["option_type"] == "CALL"]
        puts = df_pd[df_pd["option_type"] == "PUT"]

        # Calculate forward price using put-call parity
        if len(calls) > 0 and len(puts) > 0:
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
        strikes = df_pd["strike"].values.astype(float)
        mids = df_pd["mid"].values.astype(float)
        is_call = df_pd["option_type"].values == "CALL"

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

        # Calculate IVs from mid prices
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

        if fit_result.rmse > self.config.max_rmse:
            logger.warning(
                f"High RMSE for {exp_date}: {fit_result.rmse:.4f} > {self.config.max_rmse}"
            )

        # Validate arbitrage
        arb_result = validate_surface(fit_result.params, tte_years)

        # Calculate ATM vol
        atm_vol = fit_result.params.implied_vol(0.0, tte_years)

        # Calculate 25-delta skew
        skew_width = 0.67 * atm_vol * math.sqrt(tte_years)
        iv_25d_call = fit_result.params.implied_vol(skew_width, tte_years)
        iv_25d_put = fit_result.params.implied_vol(-skew_width, tte_years)
        skew_25delta = iv_25d_put - iv_25d_call

        return IntradayFittedSurface(
            symbol=symbol,
            timestamp=timestamp,
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


def fit_intraday_surface(
    symbol: str,
    target_date: date,
    time_of_day: time,
    config: Optional[IntradayPipelineConfig] = None,
    use_mock: bool = False,
) -> IntradaySurfaceResult:
    """
    Convenience function to fit surfaces at a specific timestamp.

    Args:
        symbol: Underlying symbol
        target_date: Date
        time_of_day: Time of day (ET)
        config: Optional pipeline configuration
        use_mock: Use mock data for testing

    Returns:
        IntradaySurfaceResult with fitted surfaces
    """
    pipeline = IntradaySurfacePipeline(config, use_mock=use_mock)
    try:
        return pipeline.fit_at_time(symbol, target_date, time_of_day)
    finally:
        pipeline.close()


def fit_intraday_time_series(
    symbol: str,
    target_date: date,
    interval_minutes: int = 30,
    config: Optional[IntradayPipelineConfig] = None,
    use_mock: bool = False,
) -> List[IntradaySurfaceResult]:
    """
    Convenience function to fit surfaces at regular intervals.

    Args:
        symbol: Underlying symbol
        target_date: Date
        interval_minutes: Interval between fits
        config: Optional pipeline configuration
        use_mock: Use mock data for testing

    Returns:
        List of IntradaySurfaceResult
    """
    pipeline = IntradaySurfacePipeline(config, use_mock=use_mock)
    try:
        return pipeline.fit_time_series(
            symbol, target_date, interval_minutes=interval_minutes
        )
    finally:
        pipeline.close()
