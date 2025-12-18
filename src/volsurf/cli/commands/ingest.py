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
    from volsurf.ingestion.terminal import ensure_terminal_running
    from volsurf.config.settings import get_settings

    settings = get_settings()
    target = date.fromisoformat(target_date) if target_date else date.today()

    console.print(f"Ingesting data for [cyan]{symbol}[/cyan] on [cyan]{target}[/cyan]...")

    if settings.use_mock_data:
        console.print("[yellow]Using mock data (no Theta username configured)[/yellow]")
    else:
        # Auto-start terminal if needed
        if not ensure_terminal_running():
            console.print("[red]Failed to start Theta Terminal.[/red]")
            raise typer.Exit(1)

    pipeline = IngestionPipeline()
    records = pipeline.run_daily_ingestion(symbol, target)
    console.print(f"[green]Daily ingestion complete! Inserted {records} records.[/green]")


@app.command()
def backfill(
    symbol: str = typer.Argument("SPY", help="Symbol to backfill"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
) -> None:
    """Backfill historical data for a symbol."""
    from volsurf.ingestion.pipeline import IngestionPipeline
    from volsurf.ingestion.terminal import ensure_terminal_running
    from volsurf.config.settings import get_settings

    settings = get_settings()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    console.print(
        f"Backfilling [cyan]{symbol}[/cyan] from [cyan]{start_date}[/cyan] to [cyan]{end_date}[/cyan]..."
    )

    if settings.use_mock_data:
        console.print("[yellow]Using mock data (no Theta username configured)[/yellow]")
    else:
        # Auto-start terminal if needed
        if not ensure_terminal_running():
            console.print("[red]Failed to start Theta Terminal.[/red]")
            raise typer.Exit(1)

    pipeline = IngestionPipeline()
    stats = pipeline.backfill_historical(symbol, start_date, end_date)

    console.print(f"[green]Backfill complete![/green]")
    console.print(f"  Days processed: {stats['days_processed']}")
    console.print(f"  Records inserted: {stats['records_inserted']}")
    console.print(f"  Days skipped: {stats['days_skipped']}")
    console.print(f"  Days failed: {stats['days_failed']}")


@app.command()
def status() -> None:
    """Show ingestion status and data coverage."""
    from volsurf.database.connection import get_connection
    from volsurf.config.settings import get_settings
    from rich.table import Table

    settings = get_settings()
    conn = get_connection()

    # Show configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Theta Terminal: {settings.theta_terminal_url}")
    console.print(f"  Using mock data: {settings.use_mock_data}")
    console.print(f"  Database: {settings.duckdb_path}")

    # Check if tables exist
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = [t[0] for t in tables]

    if "raw_options_chains" not in table_names:
        console.print("\n[yellow]Database not initialized. Run 'volsurf init-db' first.[/yellow]")
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
        console.print("\n[yellow]No data ingested yet.[/yellow]")
        return

    table = Table(title="\nOptions Data Coverage")
    table.add_column("Symbol")
    table.add_column("Start Date")
    table.add_column("End Date")
    table.add_column("Days")
    table.add_column("Records")

    for row in result:
        table.add_row(str(row[0]), str(row[1]), str(row[2]), str(row[3]), f"{row[4]:,}")

    console.print(table)

    # Underlying prices coverage
    result = conn.execute("""
        SELECT
            symbol,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as num_days
        FROM underlying_prices
        GROUP BY symbol
    """).fetchall()

    if result:
        table = Table(title="\nUnderlying Price Coverage")
        table.add_column("Symbol")
        table.add_column("Start Date")
        table.add_column("End Date")
        table.add_column("Days")

        for row in result:
            table.add_row(str(row[0]), str(row[1]), str(row[2]), str(row[3]))

        console.print(table)


@app.command()
def check() -> None:
    """Check Theta Terminal connectivity."""
    from volsurf.ingestion.pipeline import IngestionPipeline
    from volsurf.config.settings import get_settings

    settings = get_settings()
    console.print(f"\n[bold]Checking Theta Terminal connectivity...[/bold]")
    console.print(f"  URL: {settings.theta_terminal_url}")

    if settings.use_mock_data:
        console.print("\n[yellow]Mock mode enabled (THETA_USERNAME not set)[/yellow]")
        console.print("Set THETA_USERNAME in .env to use real data.")
        return

    pipeline = IngestionPipeline()

    if pipeline.check_terminal():
        console.print("\n[green]Theta Terminal is accessible![/green]")

        # Try to get some basic info
        try:
            expirations = pipeline.client.get_expirations("SPY")
            console.print(f"  Found {len(expirations)} SPY expirations")
            if expirations:
                console.print(f"  Nearest: {expirations[0]}")
                console.print(f"  Farthest: {expirations[-1]}")
        except Exception as e:
            console.print(f"[yellow]Could not fetch expirations: {e}[/yellow]")
    else:
        console.print("\n[red]Cannot connect to Theta Terminal![/red]")
        console.print("\nTroubleshooting:")
        console.print("  1. Ensure the terminal is running: java -jar ThetaTerminal.jar")
        console.print("  2. Check the port in your config file")
        console.print(f"  3. Try opening {settings.theta_terminal_url}/option/list/expirations?symbol=SPY in a browser")
