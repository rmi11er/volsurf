"""Data ingestion CLI commands."""

from datetime import date
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Data ingestion commands")
console = Console()


@app.command()
def daily(
    symbol: str = typer.Argument("SPY", help="Symbol to ingest"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to ingest (YYYY-MM-DD), defaults to today"
    ),
) -> None:
    """Run daily data ingestion for a symbol."""
    from volsurf.ingestion.pipeline import IngestionPipeline
    from volsurf.config.settings import get_settings
    import asyncio

    settings = get_settings()
    target = date.fromisoformat(target_date) if target_date else date.today()

    console.print(f"Ingesting data for [cyan]{symbol}[/cyan] on [cyan]{target}[/cyan]...")

    if settings.use_mock_data:
        console.print("[yellow]Using mock data (no API key configured)[/yellow]")

    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run_daily_ingestion(symbol, target))
    console.print("[green]Daily ingestion complete![/green]")


@app.command()
def backfill(
    symbol: str = typer.Argument("SPY", help="Symbol to backfill"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
) -> None:
    """Backfill historical data for a symbol."""
    from volsurf.ingestion.pipeline import IngestionPipeline
    from volsurf.config.settings import get_settings
    import asyncio

    settings = get_settings()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    console.print(
        f"Backfilling [cyan]{symbol}[/cyan] from [cyan]{start_date}[/cyan] to [cyan]{end_date}[/cyan]..."
    )

    if settings.use_mock_data:
        console.print("[yellow]Using mock data (no API key configured)[/yellow]")

    pipeline = IngestionPipeline()
    asyncio.run(pipeline.backfill_historical(symbol, start_date, end_date))
    console.print("[green]Backfill complete![/green]")


@app.command()
def status() -> None:
    """Show ingestion status and data coverage."""
    from volsurf.database.connection import get_connection
    from rich.table import Table

    conn = get_connection()

    # Check if tables exist
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = [t[0] for t in tables]

    if "raw_options_chains" not in table_names:
        console.print("[yellow]Database not initialized. Run 'volsurf init-db' first.[/yellow]")
        return

    # Get data coverage
    result = conn.execute("""
        SELECT
            symbol,
            MIN(quote_date) as min_date,
            MAX(quote_date) as max_date,
            COUNT(DISTINCT quote_date) as num_days,
            COUNT(*) as total_records
        FROM raw_options_chains
        GROUP BY symbol
    """).fetchall()

    if not result:
        console.print("[yellow]No data ingested yet.[/yellow]")
        return

    table = Table(title="Data Coverage")
    table.add_column("Symbol")
    table.add_column("Start Date")
    table.add_column("End Date")
    table.add_column("Days")
    table.add_column("Records")

    for row in result:
        table.add_row(str(row[0]), str(row[1]), str(row[2]), str(row[3]), f"{row[4]:,}")

    console.print(table)
