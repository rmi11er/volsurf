"""Visualization utilities for volatility analytics."""

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

console = Console()


# =============================================================================
# Terminal (Rich) Visualizations
# =============================================================================


def print_ascii_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    width: int = 50,
    value_format: str = ".2%",
    color: str = "cyan",
) -> None:
    """
    Print ASCII bar chart for time series data.

    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        title: Chart title
        width: Width of bars in characters
        value_format: Format string for values (e.g., ".2%" for percentage)
        color: Rich color for bars
    """
    if df.empty:
        console.print(f"[yellow]No data available for {title}[/yellow]")
        return

    values = df[value_col].values
    dates = df[date_col].values

    # Handle NaN values
    valid_mask = ~pd.isna(values)
    if not valid_mask.any():
        console.print(f"[yellow]No valid data for {title}[/yellow]")
        return

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    val_range = max_val - min_val if max_val != min_val else 1

    console.print(f"\n[bold]{title}[/bold]")
    console.print("=" * (width + 25))

    for i, (d, v) in enumerate(zip(dates, values)):
        if pd.isna(v):
            continue

        # Format date
        if hasattr(d, "strftime"):
            date_str = d.strftime("%Y-%m-%d")
        else:
            date_str = str(d)[:10]

        # Calculate bar length
        normalized = (v - min_val) / val_range if val_range > 0 else 0.5
        bar_len = int(normalized * width)
        bar = "█" * bar_len

        # Format value
        if "%" in value_format:
            val_str = f"{v:{value_format}}"
        else:
            val_str = f"{v:{value_format}}"

        console.print(f"{date_str}  {val_str}  [{color}]{bar}[/{color}]")

    console.print()


def print_term_structure_table(
    data: List[dict], title: str = "Volatility Term Structure"
) -> None:
    """
    Print term structure as a Rich table.

    Args:
        data: List of dicts with expiration, tte_days, atm_vol, skew data
        title: Table title
    """
    if not data:
        console.print("[yellow]No term structure data available[/yellow]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")

    table.add_column("Expiration", style="dim")
    table.add_column("Days", justify="right")
    table.add_column("ATM Vol", justify="right")
    table.add_column("25δ Skew", justify="right")

    for row in data:
        exp = row["expiration"]
        if hasattr(exp, "strftime"):
            exp_str = exp.strftime("%Y-%m-%d")
        else:
            exp_str = str(exp)

        tte_str = str(row["tte_days"])
        atm_str = f"{row['atm_vol']:.2%}" if row["atm_vol"] else "-"
        skew_str = f"{row['skew_25delta']:.2%}" if row["skew_25delta"] else "-"

        table.add_row(exp_str, tte_str, atm_str, skew_str)

    console.print(table)


def print_surface_summary(summary) -> None:
    """
    Print surface summary as a Rich panel.

    Args:
        summary: SurfaceSummary dataclass
    """
    lines = [
        f"[bold]Symbol:[/bold] {summary.symbol}",
        f"[bold]Date:[/bold] {summary.quote_date}",
        f"[bold]Surfaces:[/bold] {summary.num_surfaces} across {summary.num_expirations} expirations",
        "",
        "[bold cyan]ATM Volatility[/bold cyan]",
        f"  Min: {summary.atm_vol_min:.2%}" if summary.atm_vol_min else "  Min: -",
        f"  Max: {summary.atm_vol_max:.2%}" if summary.atm_vol_max else "  Max: -",
        f"  Mean: {summary.atm_vol_mean:.2%}" if summary.atm_vol_mean else "  Mean: -",
        "",
        "[bold cyan]25-Delta Skew[/bold cyan]",
        f"  Min: {summary.skew_min:.2%}" if summary.skew_min else "  Min: -",
        f"  Max: {summary.skew_max:.2%}" if summary.skew_max else "  Max: -",
        f"  Mean: {summary.skew_mean:.2%}" if summary.skew_mean else "  Mean: -",
        "",
        "[bold cyan]Fit Quality[/bold cyan]",
        f"  Avg RMSE: {summary.avg_rmse:.4%}" if summary.avg_rmse else "  Avg RMSE: -",
        f"  Max RMSE: {summary.max_rmse:.4%}" if summary.max_rmse else "  Max RMSE: -",
        f"  Total points: {summary.total_points}",
        "",
        "[bold cyan]Arbitrage[/bold cyan]",
        f"  Passing: {summary.surfaces_passing_arb}/{summary.num_surfaces}",
        f"  Butterfly violations: {summary.total_butterfly_violations}",
        f"  Calendar violations: {summary.total_calendar_violations}",
    ]

    panel = Panel("\n".join(lines), title="Surface Summary", border_style="blue")
    console.print(panel)


def print_vrp_summary(df: pd.DataFrame) -> None:
    """
    Print VRP summary statistics.

    Args:
        df: DataFrame with VRP data
    """
    if df.empty:
        console.print("[yellow]No VRP data available[/yellow]")
        return

    table = Table(title="Variance Risk Premium Summary", show_header=True)

    table.add_column("Metric", style="bold")
    table.add_column("30-Day", justify="right")
    table.add_column("60-Day", justify="right")
    table.add_column("90-Day", justify="right")

    # Calculate stats
    for stat_name, stat_func in [("Mean", np.nanmean), ("Std", np.nanstd), ("Min", np.nanmin), ("Max", np.nanmax)]:
        row = [stat_name]
        for col in ["vrp_30d", "vrp_60d", "vrp_90d"]:
            if col in df.columns:
                val = stat_func(df[col].values)
                row.append(f"{val:.2%}" if not np.isnan(val) else "-")
            else:
                row.append("-")
        table.add_row(*row)

    # Latest values
    if len(df) > 0:
        latest = df.iloc[-1]
        row = ["[cyan]Latest[/cyan]"]
        for col in ["vrp_30d", "vrp_60d", "vrp_90d"]:
            if col in df.columns and not pd.isna(latest[col]):
                row.append(f"[cyan]{latest[col]:.2%}[/cyan]")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def print_realized_vol_table(result) -> None:
    """
    Print realized volatility results as a table.

    Args:
        result: RealizedVolResult dataclass
    """
    table = Table(title=f"Realized Volatility - {result.symbol} ({result.date})")

    table.add_column("Estimator", style="bold")
    table.add_column("10-Day", justify="right")
    table.add_column("21-Day", justify="right")
    table.add_column("63-Day", justify="right")
    table.add_column("252-Day", justify="right")

    # Close-to-close
    table.add_row(
        "Close-to-Close",
        f"{result.rv_10d:.2%}" if result.rv_10d else "-",
        f"{result.rv_21d:.2%}" if result.rv_21d else "-",
        f"{result.rv_63d:.2%}" if result.rv_63d else "-",
        f"{result.rv_252d:.2%}" if result.rv_252d else "-",
    )

    # Parkinson
    table.add_row(
        "Parkinson",
        f"{result.parkinson_10d:.2%}" if result.parkinson_10d else "-",
        f"{result.parkinson_21d:.2%}" if result.parkinson_21d else "-",
        "-",
        "-",
    )

    # Garman-Klass
    table.add_row(
        "Garman-Klass",
        f"{result.gk_10d:.2%}" if result.gk_10d else "-",
        f"{result.gk_21d:.2%}" if result.gk_21d else "-",
        "-",
        "-",
    )

    console.print(table)


# =============================================================================
# Plotly Visualizations
# =============================================================================


def plot_atm_vol_timeseries(
    df: pd.DataFrame,
    title: str = "ATM Implied Volatility",
    show: bool = False,
    save_path: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create interactive ATM vol time series plot.

    Args:
        df: DataFrame with date and atm_vol columns
        title: Plot title
        show: Whether to display plot
        save_path: Path to save HTML file

    Returns:
        Plotly Figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]Plotly not available for interactive plots[/yellow]")
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["atm_vol"] * 100,  # Convert to percentage
            mode="lines+markers",
            name="ATM Vol",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    if show:
        fig.show()

    return fig


def plot_term_structure(
    data: List[dict],
    title: str = "Volatility Term Structure",
    show: bool = False,
    save_path: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create interactive term structure plot.

    Args:
        data: List of dicts with tte_days, atm_vol, skew data
        title: Plot title
        show: Whether to display plot
        save_path: Path to save HTML file

    Returns:
        Plotly Figure object or None
    """
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]Plotly not available for interactive plots[/yellow]")
        return None

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("ATM Volatility", "25-Delta Skew")
    )

    tte_days = [d["tte_days"] for d in data]
    atm_vols = [d["atm_vol"] * 100 if d["atm_vol"] else None for d in data]
    skews = [d["skew_25delta"] * 100 if d["skew_25delta"] else None for d in data]

    # ATM vol
    fig.add_trace(
        go.Scatter(
            x=tte_days,
            y=atm_vols,
            mode="lines+markers",
            name="ATM Vol",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Skew
    fig.add_trace(
        go.Scatter(
            x=tte_days,
            y=skews,
            mode="lines+markers",
            name="25δ Skew",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Days to Expiration", row=2, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="Skew (%)", row=2, col=1)

    fig.update_layout(title=title, template="plotly_white", height=600)

    if save_path:
        fig.write_html(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    if show:
        fig.show()

    return fig


def plot_vol_smile(
    smile_data: dict,
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create interactive volatility smile plot.

    Args:
        smile_data: Dict from SurfaceMetrics.get_smile_data()
        title: Plot title
        show: Whether to display plot
        save_path: Path to save HTML file

    Returns:
        Plotly Figure object or None
    """
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]Plotly not available for interactive plots[/yellow]")
        return None

    if smile_data is None:
        console.print("[yellow]No smile data available[/yellow]")
        return None

    k = smile_data["smile"]["log_moneyness"]
    vols = [v * 100 if v and not np.isnan(v) else None for v in smile_data["smile"]["implied_vol"]]

    if title is None:
        title = f"{smile_data['symbol']} Vol Smile - {smile_data['expiration_date']}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k,
            y=vols,
            mode="lines",
            name="IV Smile",
            line=dict(color="blue", width=2),
        )
    )

    # Mark ATM
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")

    fig.update_layout(
        title=title,
        xaxis_title="Log Moneyness (ln(K/F))",
        yaxis_title="Implied Volatility (%)",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    if show:
        fig.show()

    return fig


def plot_iv_vs_rv(
    iv_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    title: str = "Implied vs Realized Volatility",
    show: bool = False,
    save_path: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create IV vs RV comparison plot.

    Args:
        iv_df: DataFrame with date, atm_vol columns
        rv_df: DataFrame with date, realized_vol columns
        title: Plot title
        show: Whether to display
        save_path: Path to save HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]Plotly not available for interactive plots[/yellow]")
        return None

    fig = go.Figure()

    # IV
    fig.add_trace(
        go.Scatter(
            x=iv_df["date"],
            y=iv_df["atm_vol"] * 100,
            mode="lines",
            name="Implied Vol (30d)",
            line=dict(color="blue", width=2),
        )
    )

    # RV
    fig.add_trace(
        go.Scatter(
            x=rv_df["date"],
            y=rv_df["realized_vol"] * 100,
            mode="lines",
            name="Realized Vol (21d)",
            line=dict(color="orange", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    if save_path:
        fig.write_html(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    if show:
        fig.show()

    return fig


def plot_vrp_timeseries(
    df: pd.DataFrame,
    title: str = "Variance Risk Premium",
    show: bool = False,
    save_path: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    Create VRP time series plot.

    Args:
        df: DataFrame with VRP metrics
        title: Plot title
        show: Whether to display
        save_path: Path to save HTML

    Returns:
        Plotly Figure or None
    """
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]Plotly not available for interactive plots[/yellow]")
        return None

    fig = go.Figure()

    # VRP 30d
    if "vrp_30d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["vrp_30d"] * 100,
                mode="lines+markers",
                name="VRP 30d",
                line=dict(color="green", width=2),
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="VRP (%)",
        hovermode="x unified",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    if show:
        fig.show()

    return fig
