"""Analytics CLI commands."""

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Analytics commands")
console = Console()


@app.command()
def vrp(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to analyze (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Calculate variance risk premium metrics."""
    console.print(f"[yellow]VRP analysis for {symbol} - coming in Phase 3[/yellow]")


@app.command()
def realized_vol(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    window: int = typer.Option(21, "--window", "-w", help="Lookback window in trading days"),
) -> None:
    """Calculate realized volatility."""
    console.print(f"[yellow]Realized vol for {symbol} - coming in Phase 3[/yellow]")


@app.command()
def surface_summary(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to summarize (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Show summary of fitted surface for a date."""
    console.print(f"[yellow]Surface summary for {symbol} - coming in Phase 3[/yellow]")
