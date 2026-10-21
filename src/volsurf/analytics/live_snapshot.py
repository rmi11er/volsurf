"""Live snapshot — fetch current surfaces from Theta Terminal and produce SurfaceMetricsSnapshot."""

import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Callable, Optional

import numpy as np
from loguru import logger

from volsurf.analytics.surface_metrics import get_atm_vol_from_svi, get_vol_at_moneyness
from volsurf.analytics.surface_metrics_history import (
    SurfaceMetricsHistoryCalculator,
    SurfaceMetricsSnapshot,
    _NORM_PPF_75,
)
from volsurf.config.settings import get_settings
from volsurf.fitting.intraday_pipeline import IntradaySurfacePipeline
from volsurf.models.schemas import IntradaySurfaceResult


@dataclass
class LiveSnapshotResult:
    """Container for a batch of live snapshots."""

    snapshots: dict[str, SurfaceMetricsSnapshot] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    elapsed_seconds: float = 0.0


def intraday_result_to_snapshot(result: IntradaySurfaceResult) -> Optional[SurfaceMetricsSnapshot]:
    """Convert an IntradaySurfaceResult into a SurfaceMetricsSnapshot.

    Replicates the interpolation logic from SurfaceMetricsHistoryCalculator.calculate_for_date
    but operates on in-memory intraday surfaces instead of the database.
    """
    if not result.surfaces:
        return None

    today = result.timestamp.date() if isinstance(result.timestamp, datetime) else date.today()
    snap = SurfaceMetricsSnapshot(symbol=result.symbol, quote_date=today)

    # Sort surfaces by TTE
    surfaces = sorted(result.surfaces, key=lambda s: s.tte_years)
    settings = get_settings()

    tte = np.array([s.tte_years for s in surfaces])
    atm_vols = np.array([s.atm_vol for s in surfaces])
    skews = np.array([s.skew_25delta if s.skew_25delta is not None else np.nan for s in surfaces])

    # Butterfly per expiration
    calc = SurfaceMetricsHistoryCalculator
    butterflies = np.array([
        calc._compute_butterfly(
            s.svi_params.a, s.svi_params.b, s.svi_params.rho,
            s.svi_params.m, s.svi_params.sigma, s.tte_years,
        )
        for s in surfaces
    ])

    # Interpolate to standard tenors
    for tenor_days, atm_attr, skew_attr in [
        (30, "atm_vol_30d", "skew_25d_30d"),
        (60, "atm_vol_60d", "skew_25d_60d"),
        (90, "atm_vol_90d", "skew_25d_90d"),
    ]:
        target_tte = tenor_days / float(settings.annualization_days_per_year)
        setattr(snap, atm_attr, calc._sanitize(calc._interpolate_metric(tte, atm_vols, target_tte)))
        setattr(snap, skew_attr, calc._sanitize(calc._interpolate_metric(tte, skews, target_tte)))

    snap.butterfly_30d = calc._sanitize(
        calc._interpolate_metric(tte, butterflies, 30 / float(settings.annualization_days_per_year))
    )

    if snap.atm_vol_90d is not None and snap.atm_vol_30d is not None:
        snap.term_slope = calc._sanitize(snap.atm_vol_90d - snap.atm_vol_30d)

    snap.underlying_close = float(result.underlying_price)
    return snap


class LiveSnapshotManager:
    """Orchestrates live snapshot fetching for the watchlist."""

    def __init__(self):
        self._settings = get_settings()
        self._calc = SurfaceMetricsHistoryCalculator()

    def check_terminal(self) -> bool:
        """Probe whether the Theta Terminal is reachable."""
        from volsurf.ingestion.intraday_client import ThetaIntradayClient

        client = ThetaIntradayClient(self._settings)
        try:
            return client.check_terminal_status()
        finally:
            client.close()

    def _now_et(self) -> time:
        """Current time in US/Eastern."""
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).time()

    def fetch_single_symbol(
        self, symbol: str, pipeline: IntradaySurfacePipeline,
    ) -> tuple[Optional[SurfaceMetricsSnapshot], Optional[str]]:
        """Fetch and convert a live snapshot for one symbol.

        Returns (snapshot, error_message). Exactly one is None.
        """
        today = date.today()
        now_et = self._now_et()

        try:
            result = pipeline.fit_at_time(symbol, today, now_et)
        except Exception as exc:
            return None, f"Fit failed: {exc}"

        if not result.surfaces:
            return None, "No surfaces (market may be closed)"

        snap = intraday_result_to_snapshot(result)
        if snap is None:
            return None, "Conversion failed"

        # Copy RV / VRP from most recent EOD snapshot (these are daily metrics)
        eod_snap = self._calc.get_latest_snapshot(symbol)
        if eod_snap:
            snap.rv_21d = eod_snap.rv_21d
            snap.rv_63d = eod_snap.rv_63d
            snap.vrp_30d = eod_snap.vrp_30d
            # Day-over-day changes vs EOD
            self._calc.calculate_day_over_day(snap, eod_snap)

        return snap, None

    def fetch_all(
        self,
        symbols: list[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> LiveSnapshotResult:
        """Sequentially fetch live snapshots for all symbols.

        Args:
            symbols: List of symbols to fetch.
            progress_callback: Called with (completed, total, current_symbol) after each symbol.
        """
        result = LiveSnapshotResult()
        t0 = _time.monotonic()
        pipeline = IntradaySurfacePipeline()

        try:
            for i, sym in enumerate(symbols):
                snap, err = self.fetch_single_symbol(sym, pipeline)
                if snap is not None:
                    result.snapshots[sym] = snap
                else:
                    result.errors[sym] = err or "Unknown error"

                if progress_callback:
                    progress_callback(i + 1, len(symbols), sym)
        finally:
            pipeline.close()

        result.elapsed_seconds = _time.monotonic() - t0
        result.timestamp = datetime.now()
        logger.info(
            f"Live snapshot: {len(result.snapshots)} OK, {len(result.errors)} errors "
            f"in {result.elapsed_seconds:.1f}s"
        )
        return result

    def enrich_with_percentiles(
        self,
        snapshots: dict[str, SurfaceMetricsSnapshot],
        lookback_days: int = 252,
    ) -> dict[str, dict]:
        """Compute percentile ranks and z-scores for live snapshots against DB history.

        Returns {symbol: {"pctile": {metric: value}, "zscore": {metric: value}}}
        """
        key_metrics = ["atm_vol_30d", "skew_25d_30d", "term_slope", "butterfly_30d", "vrp_30d"]
        enrichment: dict[str, dict] = {}

        for sym, snap in snapshots.items():
            pctiles: dict[str, Optional[float]] = {}
            zscores: dict[str, Optional[float]] = {}

            for metric in key_metrics:
                val = getattr(snap, metric, None)
                if val is None:
                    pctiles[metric] = None
                    zscores[metric] = None
                    continue
                pctiles[metric] = self._calc.calculate_percentile_rank(
                    sym, metric, val, snap.quote_date, lookback_days,
                )
                zscores[metric] = self._calc.calculate_zscore(
                    sym, metric, val, snap.quote_date, lookback_days,
                )

            enrichment[sym] = {"pctile": pctiles, "zscore": zscores}

        return enrichment
