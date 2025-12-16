"""Surface fitting CLI commands."""

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Surface fitting commands")
console = Console()


@app.command()
def surfaces(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to fit (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Fit volatility surfaces for a specific date."""
    console.print(f"[yellow]Surface fitting for {symbol} - coming in Phase 2[/yellow]")


@app.command()
def term_structure(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to fit (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Fit term structure model for a specific date."""
    console.print(f"[yellow]Term structure fitting for {symbol} - coming in Phase 2[/yellow]")


@app.command()
def batch(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
) -> None:
    """Batch fit surfaces for a date range."""
    console.print(f"[yellow]Batch fitting for {symbol} - coming in Phase 2[/yellow]")
