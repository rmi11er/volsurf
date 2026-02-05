"""Macro data ingestion CLI commands."""

from datetime import date, datetime
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

app = typer.Typer(help="Macro context data commands")
console = Console()


def _parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)


@app.command()
def backfill(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated display names (e.g. VIX,TLT). Defaults to all."
    ),
) -> None:
    """Backfill macro symbol data from yfinance into underlying_prices."""
    from volsurf.config.settings import get_settings
    from volsurf.ingestion.macro_client import MacroClient

    settings = get_settings()
    start_date = _parse_date(start)
    end_date = _parse_date(end)

    # Filter to requested symbols
    all_macro = settings.macro_symbols
    if symbols:
        requested = [s.strip() for s in symbols.split(",")]
        macro_to_fetch = {k: v for k, v in all_macro.items() if k in requested}
        unknown = [s for s in requested if s not in all_macro]
        if unknown:
            console.print(f"[yellow]Unknown macro symbols (skipped): {', '.join(unknown)}[/yellow]")
            console.print(f"[dim]Available: {', '.join(all_macro.keys())}[/dim]")
    else:
        macro_to_fetch = all_macro

    if not macro_to_fetch:
        console.print("[red]No valid macro symbols to fetch.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[cyan]Backfilling {len(macro_to_fetch)} macro symbols "
        f"from {start_date} to {end_date}[/cyan]"
    )

    client = MacroClient()
    failed: list[str] = []
    total_rows = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching...", total=len(macro_to_fetch))

        for display_name, yf_ticker in macro_to_fetch.items():
            progress.update(task, description=f"Fetching [cyan]{display_name}[/cyan]...")
            try:
                count = client.backfill_symbol(display_name, yf_ticker, start_date, end_date)
                total_rows += count
                console.print(f"  [green]{display_name}[/green] ({yf_ticker}): {count} rows")
            except Exception as e:
                failed.append(display_name)
                console.print(f"  [red]{display_name}[/red] ({yf_ticker}): {e}")
            progress.advance(task)

    console.print(f"\n[bold]Macro Backfill Complete[/bold]")
    console.print(
        f"  Symbols: {len(macro_to_fetch) - len(failed)}/{len(macro_to_fetch)} succeeded"
    )
    console.print(f"  Total rows: {total_rows:,}")
    if failed:
        console.print(f"  [red]Failed: {', '.join(failed)}[/red]")
