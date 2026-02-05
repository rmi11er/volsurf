"""Batch operations across the full watchlist."""

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
from rich.table import Table

app = typer.Typer(help="Batch operations across all watchlist symbols")
console = Console()


def _get_watchlist() -> list[str]:
    """Get the configured watchlist."""
    from volsurf.config.settings import get_settings

    return get_settings().watchlist


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string to date object."""
    if date_str is None:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)


@app.command()
def ingest(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbols (overrides watchlist)"
    ),
) -> None:
    """Backfill chains and underlying prices for all watchlist symbols."""
    from volsurf.config.settings import get_settings
    from volsurf.ingestion.pipeline import IngestionPipeline
    from volsurf.ingestion.terminal import ensure_terminal_running

    settings = get_settings()
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    symbol_list = symbols.split(",") if symbols else _get_watchlist()

    if not settings.use_mock_data:
        if not ensure_terminal_running():
            console.print("[red]Failed to start Theta Terminal.[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Using mock data (no Theta username configured)[/yellow]")

    pipeline = IngestionPipeline()
    totals = {"days_processed": 0, "records_inserted": 0, "days_skipped": 0, "days_failed": 0}
    failed_symbols: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(symbol_list))

        for sym in symbol_list:
            progress.update(task, description=f"Ingesting [cyan]{sym}[/cyan]...")
            try:
                stats = pipeline.backfill_historical(sym, start_date, end_date)
                for key in totals:
                    totals[key] += stats[key]
                console.print(
                    f"  [green]{sym}[/green]: {stats['records_inserted']:,} records, "
                    f"{stats['days_processed']} days"
                )
            except Exception as e:
                failed_symbols.append(sym)
                console.print(f"  [red]{sym}[/red]: {e}")
            progress.advance(task)

    # Summary
    console.print(f"\n[bold]Batch Ingest Complete[/bold]")
    console.print(f"  Symbols: {len(symbol_list) - len(failed_symbols)}/{len(symbol_list)} succeeded")
    console.print(f"  Days processed: {totals['days_processed']}")
    console.print(f"  Records inserted: {totals['records_inserted']:,}")
    console.print(f"  Days skipped: {totals['days_skipped']}")
    if failed_symbols:
        console.print(f"  [red]Failed: {', '.join(failed_symbols)}[/red]")


@app.command()
def fit(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbols (overrides watchlist)"
    ),
) -> None:
    """Batch fit SVI surfaces for all watchlist symbols."""
    from volsurf.fitting.pipeline import batch_fit_surfaces

    start_date = _parse_date(start)
    end_date = _parse_date(end)
    symbol_list = symbols.split(",") if symbols else _get_watchlist()

    totals = {"dates_processed": 0, "total_surfaces": 0, "total_skipped": 0, "arbitrage_violations": 0}
    failed_symbols: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fitting...", total=len(symbol_list))

        for sym in symbol_list:
            progress.update(task, description=f"Fitting [cyan]{sym}[/cyan]...")
            try:
                stats = batch_fit_surfaces(sym, start_date, end_date)
                for key in totals:
                    totals[key] += stats[key]
                console.print(
                    f"  [green]{sym}[/green]: {stats['total_surfaces']} surfaces, "
                    f"{stats['dates_processed']} dates"
                )
            except Exception as e:
                failed_symbols.append(sym)
                console.print(f"  [red]{sym}[/red]: {e}")
            progress.advance(task)

    console.print(f"\n[bold]Batch Fit Complete[/bold]")
    console.print(f"  Symbols: {len(symbol_list) - len(failed_symbols)}/{len(symbol_list)} succeeded")
    console.print(f"  Dates processed: {totals['dates_processed']}")
    console.print(f"  Total surfaces: {totals['total_surfaces']}")
    if totals["arbitrage_violations"] > 0:
        console.print(f"  [yellow]Arbitrage violations: {totals['arbitrage_violations']}[/yellow]")
    if failed_symbols:
        console.print(f"  [red]Failed: {', '.join(failed_symbols)}[/red]")


@app.command()
def analyze(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbols (overrides watchlist)"
    ),
) -> None:
    """Run RV, VRP, and surface metrics history for all watchlist symbols."""
    from volsurf.analytics import (
        RealizedVolCalculator,
        SurfaceMetricsHistoryCalculator,
        VRPCalculator,
    )
    from volsurf.database.schema import init_schema

    init_schema()
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    symbol_list = symbols.split(",") if symbols else _get_watchlist()

    rv_calc = RealizedVolCalculator()
    vrp_calc = VRPCalculator()
    smh_calc = SurfaceMetricsHistoryCalculator()
    failed_symbols: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(symbol_list))

        for sym in symbol_list:
            progress.update(task, description=f"Analyzing [cyan]{sym}[/cyan]...")
            try:
                rv_count = rv_calc.backfill(sym, start_date, end_date, store=True)
                vrp_count = vrp_calc.backfill(sym, start_date, end_date, store=True)
                smh_count = smh_calc.backfill(sym, start_date, end_date, store=True)
                console.print(
                    f"  [green]{sym}[/green]: {rv_count} RV, {vrp_count} VRP, "
                    f"{smh_count} SMH days"
                )
            except Exception as e:
                failed_symbols.append(sym)
                console.print(f"  [red]{sym}[/red]: {e}")
            progress.advance(task)

    console.print(f"\n[bold]Batch Analyze Complete[/bold]")
    console.print(f"  Symbols: {len(symbol_list) - len(failed_symbols)}/{len(symbol_list)} succeeded")
    if failed_symbols:
        console.print(f"  [red]Failed: {', '.join(failed_symbols)}[/red]")


@app.command()
def status(
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbols (overrides watchlist)"
    ),
) -> None:
    """Show data coverage across all watchlist symbols."""
    from volsurf.database.connection import get_connection

    conn = get_connection()
    symbol_list = symbols.split(",") if symbols else _get_watchlist()
    placeholders = ", ".join(["?"] * len(symbol_list))

    # Options chains coverage
    chains_query = f"""
        SELECT
            symbol,
            MIN(quote_date) as min_date,
            MAX(quote_date) as max_date,
            COUNT(DISTINCT quote_date) as num_days,
            COUNT(*) as total_records
        FROM raw_options_chains
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        ORDER BY symbol
    """
    chains = conn.execute(chains_query, symbol_list).fetchall()
    chains_by_sym = {row[0]: row for row in chains}

    # Fitted surfaces coverage
    fitted_query = f"""
        SELECT
            symbol,
            COUNT(DISTINCT quote_date) as num_days
        FROM fitted_surfaces
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """
    fitted = conn.execute(fitted_query, symbol_list).fetchall()
    fitted_by_sym = {row[0]: row[1] for row in fitted}

    # Realized vol coverage
    rv_query = f"""
        SELECT
            symbol,
            COUNT(*) as num_days
        FROM realized_volatility
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """
    rv = conn.execute(rv_query, symbol_list).fetchall()
    rv_by_sym = {row[0]: row[1] for row in rv}

    # VRP coverage
    vrp_query = f"""
        SELECT
            symbol,
            COUNT(*) as num_days
        FROM vrp_metrics
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """
    vrp = conn.execute(vrp_query, symbol_list).fetchall()
    vrp_by_sym = {row[0]: row[1] for row in vrp}

    # Surface metrics history coverage
    smh_query = f"""
        SELECT
            symbol,
            COUNT(*) as num_days
        FROM surface_metrics_history
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """
    smh = conn.execute(smh_query, symbol_list).fetchall()
    smh_by_sym = {row[0]: row[1] for row in smh}

    # Build table
    table = Table(title="Watchlist Data Coverage")
    table.add_column("Symbol", style="cyan")
    table.add_column("Chains", justify="right")
    table.add_column("Start", justify="center")
    table.add_column("End", justify="center")
    table.add_column("Fitted", justify="right")
    table.add_column("RV", justify="right")
    table.add_column("VRP", justify="right")
    table.add_column("SMH", justify="right")

    for sym in symbol_list:
        if sym in chains_by_sym:
            row = chains_by_sym[sym]
            chain_days = str(row[3])
            start_d = str(row[1])
            end_d = str(row[2])
        else:
            chain_days = "[dim]—[/dim]"
            start_d = "[dim]—[/dim]"
            end_d = "[dim]—[/dim]"

        fitted_days = str(fitted_by_sym.get(sym, 0)) or "[dim]—[/dim]"
        rv_days = str(rv_by_sym.get(sym, 0)) or "[dim]—[/dim]"
        vrp_days = str(vrp_by_sym.get(sym, 0)) or "[dim]—[/dim]"
        smh_days = str(smh_by_sym.get(sym, 0)) or "[dim]—[/dim]"

        table.add_row(sym, chain_days, start_d, end_d, fitted_days, rv_days, vrp_days, smh_days)

    console.print(table)
