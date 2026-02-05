"""Macro context analytics: correlations, breadth, momentum, regime."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from volsurf.config.settings import get_settings
from volsurf.database.connection import get_connection


class MacroContextCalculator:
    """Compute cross-asset macro context from underlying_prices."""

    def _get_prices(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch underlying_prices for given symbols and date range."""
        conn = get_connection()
        placeholders = ", ".join(["?"] * len(symbols))
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM underlying_prices
            WHERE symbol IN ({placeholders})
              AND date BETWEEN ? AND ?
            ORDER BY symbol, date
        """
        result = conn.execute(query, [*symbols, start, end]).fetchall()
        if not result:
            return pl.DataFrame(
                schema={"symbol": pl.Utf8, "date": pl.Date, "open": pl.Float64,
                        "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                        "volume": pl.Int64}
            )
        return pl.DataFrame(
            result,
            schema={"symbol": pl.Utf8, "date": pl.Date, "open": pl.Float64,
                    "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                    "volume": pl.Int64},
            orient="row",
        )

    def get_macro_dashboard_data(self, target_date: date) -> pl.DataFrame:
        """Per macro symbol: close, 1d/5d/21d/63d % change, 252d percentile, sparkline data.

        Returns DataFrame with columns:
            symbol, close, chg_1d, chg_5d, chg_21d, chg_63d, pctile_252d, sparkline
        """
        settings = get_settings()
        macro_names = list(settings.macro_symbols.keys())

        # Fetch enough history for 252d percentile
        start = target_date - timedelta(days=400)
        prices = self._get_prices(macro_names, start, target_date)

        if prices.is_empty():
            return pl.DataFrame(
                schema={"symbol": pl.Utf8, "close": pl.Float64,
                        "chg_1d": pl.Float64, "chg_5d": pl.Float64,
                        "chg_21d": pl.Float64, "chg_63d": pl.Float64,
                        "pctile_252d": pl.Float64, "sparkline": pl.List(pl.Float64)}
            )

        rows = []
        for sym in macro_names:
            sym_df = prices.filter(pl.col("symbol") == sym).sort("date")
            if sym_df.is_empty():
                continue

            closes = sym_df["close"].to_list()
            dates = sym_df["date"].to_list()

            # Current close = last available on or before target_date
            current_close = closes[-1]

            # Percentage changes
            def pct_change(n: int) -> float | None:
                if len(closes) > n:
                    prev = closes[-(n + 1)]
                    if prev and prev != 0:
                        return (current_close - prev) / prev
                return None

            chg_1d = pct_change(1)
            chg_5d = pct_change(5)
            chg_21d = pct_change(21)
            chg_63d = pct_change(63)

            # 252d percentile rank of current close
            last_252 = closes[-252:] if len(closes) >= 252 else closes
            pctile = float(np.sum(np.array(last_252) <= current_close)) / len(last_252) * 100

            # Sparkline: last 30 closes
            sparkline = closes[-30:] if len(closes) >= 30 else closes

            rows.append({
                "symbol": sym,
                "close": current_close,
                "chg_1d": chg_1d,
                "chg_5d": chg_5d,
                "chg_21d": chg_21d,
                "chg_63d": chg_63d,
                "pctile_252d": pctile,
                "sparkline": sparkline,
            })

        if not rows:
            return pl.DataFrame(
                schema={"symbol": pl.Utf8, "close": pl.Float64,
                        "chg_1d": pl.Float64, "chg_5d": pl.Float64,
                        "chg_21d": pl.Float64, "chg_63d": pl.Float64,
                        "pctile_252d": pl.Float64, "sparkline": pl.List(pl.Float64)}
            )

        return pl.DataFrame(rows)

    def get_correlation_matrix(
        self,
        symbols: list[str],
        end_date: date,
        window: int = 63,
    ) -> pl.DataFrame:
        """Pairwise correlation matrix of log returns over trailing window.

        Returns DataFrame with 'symbol' column and one column per symbol.
        """
        start = end_date - timedelta(days=int(window * 1.6))
        prices = self._get_prices(symbols, start, end_date)

        if prices.is_empty():
            return pl.DataFrame()

        # Pivot to wide: date x symbol closes
        wide = prices.pivot(on="symbol", index="date", values="close").sort("date")

        # Keep only last `window` rows
        if len(wide) > window:
            wide = wide.tail(window)

        # Compute log returns
        sym_cols = [c for c in wide.columns if c != "date"]
        returns = {}
        for col in sym_cols:
            vals = wide[col].to_numpy().astype(float)
            log_ret = np.diff(np.log(vals[~np.isnan(vals)])) if np.sum(~np.isnan(vals)) > 1 else np.array([])
            returns[col] = log_ret

        # Align lengths (use min length)
        min_len = min((len(v) for v in returns.values()), default=0)
        if min_len < 5:
            return pl.DataFrame()

        for col in returns:
            returns[col] = returns[col][-min_len:]

        # Build correlation matrix
        matrix = np.corrcoef(np.array([returns[c] for c in sym_cols]))
        rows = []
        for i, sym in enumerate(sym_cols):
            row: dict[str, object] = {"symbol": sym}
            for j, sym2 in enumerate(sym_cols):
                row[sym2] = float(matrix[i, j])
            rows.append(row)

        return pl.DataFrame(rows)

    def get_rolling_correlations(
        self,
        watchlist_sym: str,
        macro_sym: str,
        end_date: date,
        windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """Rolling correlation between a watchlist symbol and a macro symbol.

        Returns DataFrame with columns: date, corr_21d, corr_63d (or per window).
        """
        if windows is None:
            windows = [21, 63]

        max_window = max(windows)
        start = end_date - timedelta(days=int(max_window * 2.5))
        prices = self._get_prices([watchlist_sym, macro_sym], start, end_date)

        if prices.is_empty():
            return pl.DataFrame()

        # Pivot to wide
        wide = prices.pivot(on="symbol", index="date", values="close").sort("date")
        if watchlist_sym not in wide.columns or macro_sym not in wide.columns:
            return pl.DataFrame()

        dates = wide["date"].to_list()
        a_vals = wide[watchlist_sym].to_numpy().astype(float)
        b_vals = wide[macro_sym].to_numpy().astype(float)

        # Log returns
        a_ret = np.diff(np.log(a_vals))
        b_ret = np.diff(np.log(b_vals))
        ret_dates = dates[1:]

        result_rows = []
        for i in range(len(a_ret)):
            row: dict[str, object] = {"date": ret_dates[i]}
            for w in windows:
                if i >= w - 1:
                    a_win = a_ret[i - w + 1: i + 1]
                    b_win = b_ret[i - w + 1: i + 1]
                    valid = ~(np.isnan(a_win) | np.isnan(b_win))
                    if np.sum(valid) > 5:
                        corr = np.corrcoef(a_win[valid], b_win[valid])[0, 1]
                        row[f"corr_{w}d"] = float(corr)
                    else:
                        row[f"corr_{w}d"] = None
                else:
                    row[f"corr_{w}d"] = None
            result_rows.append(row)

        if not result_rows:
            return pl.DataFrame()

        return pl.DataFrame(result_rows)

    def get_breadth_indicators(
        self,
        symbols: list[str],
        end_date: date,
        ma_window: int = 50,
    ) -> pl.DataFrame:
        """Daily breadth: pct_above_50d_ma, advance_decline_ratio.

        Returns DataFrame with columns: date, pct_above_50d_ma, adv_decl_ratio.
        """
        start = end_date - timedelta(days=int(ma_window * 2.5))
        prices = self._get_prices(symbols, start, end_date)

        if prices.is_empty():
            return pl.DataFrame()

        # For each symbol, compute MA and whether close > MA
        wide = prices.pivot(on="symbol", index="date", values="close").sort("date")
        sym_cols = [c for c in wide.columns if c != "date"]
        dates = wide["date"].to_list()

        result_rows = []
        for i in range(len(dates)):
            if i < ma_window - 1:
                continue
            above = 0
            advancing = 0
            declining = 0
            total = 0
            for col in sym_cols:
                vals = wide[col].to_numpy().astype(float)
                if np.isnan(vals[i]):
                    continue
                ma = np.nanmean(vals[max(0, i - ma_window + 1): i + 1])
                total += 1
                if vals[i] > ma:
                    above += 1
                if i > 0 and not np.isnan(vals[i - 1]):
                    if vals[i] > vals[i - 1]:
                        advancing += 1
                    elif vals[i] < vals[i - 1]:
                        declining += 1

            pct_above = above / total if total > 0 else None
            adv_decl = advancing / declining if declining > 0 else (
                float(advancing) if advancing > 0 else None
            )

            result_rows.append({
                "date": dates[i],
                "pct_above_50d_ma": pct_above,
                "adv_decl_ratio": adv_decl,
            })

        if not result_rows:
            return pl.DataFrame()

        return pl.DataFrame(result_rows)

    def get_current_breadth_snapshot(
        self,
        symbols: list[str],
        target_date: date,
        ma_window: int = 50,
    ) -> list[dict]:
        """Per-symbol detail: close, 50d MA, above/below."""
        start = target_date - timedelta(days=int(ma_window * 2))
        prices = self._get_prices(symbols, start, target_date)

        if prices.is_empty():
            return []

        details = []
        for sym in symbols:
            sym_df = prices.filter(pl.col("symbol") == sym).sort("date")
            if sym_df.is_empty():
                continue
            closes = sym_df["close"].to_numpy().astype(float)
            current = closes[-1]
            ma = float(np.nanmean(closes[-ma_window:])) if len(closes) >= ma_window else float(np.nanmean(closes))
            details.append({
                "symbol": sym,
                "close": float(current),
                "ma_50d": ma,
                "above_ma": current > ma,
            })

        return details

    def get_regime_context(self, target_date: date) -> dict:
        """Regime snapshot: VIX level/regime, term structure, credit, USD.

        Returns dict with keys:
            vix_level, vix_regime, vix_ts_ratio, vix_ts_label,
            tlt_63d_return, rate_label, hyg_21d_return, credit_label,
            uup_21d_return, usd_label
        """
        start = target_date - timedelta(days=120)
        macro_syms = ["VIX", "VIX9D", "VVIX", "TLT", "HYG", "UUP"]
        prices = self._get_prices(macro_syms, start, target_date)

        result: dict[str, object] = {}

        def _get_latest_close(sym: str) -> float | None:
            sym_df = prices.filter(pl.col("symbol") == sym).sort("date")
            if sym_df.is_empty():
                return None
            return float(sym_df["close"].to_list()[-1])

        def _get_return(sym: str, lookback: int) -> float | None:
            sym_df = prices.filter(pl.col("symbol") == sym).sort("date")
            if sym_df.is_empty() or len(sym_df) <= lookback:
                return None
            closes = sym_df["close"].to_list()
            prev = closes[-(lookback + 1)]
            curr = closes[-1]
            if prev and prev != 0:
                return (curr - prev) / prev
            return None

        # VIX regime
        vix = _get_latest_close("VIX")
        result["vix_level"] = vix
        if vix is not None:
            if vix < 15:
                result["vix_regime"] = "Low"
            elif vix < 20:
                result["vix_regime"] = "Normal"
            elif vix < 30:
                result["vix_regime"] = "Elevated"
            else:
                result["vix_regime"] = "High"
        else:
            result["vix_regime"] = "N/A"

        # VVIX
        result["vvix_level"] = _get_latest_close("VVIX")

        # VIX term structure: VIX9D / VIX
        vix9d = _get_latest_close("VIX9D")
        if vix and vix9d and vix != 0:
            ratio = vix9d / vix
            result["vix_ts_ratio"] = ratio
            result["vix_ts_label"] = "Backwardation" if ratio > 1.0 else "Contango"
        else:
            result["vix_ts_ratio"] = None
            result["vix_ts_label"] = "N/A"

        # Rates (TLT 63d return)
        tlt_ret = _get_return("TLT", 63)
        result["tlt_63d_return"] = tlt_ret
        if tlt_ret is not None:
            result["rate_label"] = "Falling rates" if tlt_ret > 0 else "Rising rates"
        else:
            result["rate_label"] = "N/A"

        # Credit (HYG 21d return)
        hyg_ret = _get_return("HYG", 21)
        result["hyg_21d_return"] = hyg_ret
        if hyg_ret is not None:
            result["credit_label"] = "Tightening" if hyg_ret > 0 else "Widening"
        else:
            result["credit_label"] = "N/A"

        # USD (UUP 21d return)
        uup_ret = _get_return("UUP", 21)
        result["uup_21d_return"] = uup_ret
        if uup_ret is not None:
            result["usd_label"] = "Strengthening" if uup_ret > 0 else "Weakening"
        else:
            result["usd_label"] = "N/A"

        return result

    def get_vix_timeseries(self, end_date: date, lookback_days: int = 252) -> pl.DataFrame:
        """VIX close time series for regime chart."""
        start = end_date - timedelta(days=int(lookback_days * 1.5))
        return self._get_prices(["VIX"], start, end_date).sort("date")
