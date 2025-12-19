"""Utility CLI commands for validation and export."""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Utility commands for validation and export")
console = Console()


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string to date object."""
    if date_str is None:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    symbol: str = typer.Argument("SPY", help="Symbol to validate"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to validate (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Validate data quality for a specific date."""
    from volsurf.database.connection import get_connection
    from volsurf.utils.validation import print_validation_summary

    if target_date:
        quote_date = parse_date(target_date)
    else:
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    print_validation_summary(symbol, quote_date)


@app.command()
def health_report(
    symbol: str = typer.Argument("SPY", help="Symbol to check"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
) -> None:
    """Generate health report across a date range."""
    from volsurf.utils.validation import DataValidator

    start_date = parse_date(start)
    end_date = parse_date(end)

    console.print(
        f"Generating health report for [cyan]{symbol}[/cyan] "
        f"from [cyan]{start_date}[/cyan] to [cyan]{end_date}[/cyan]\n"
    )

    validator = DataValidator()
    report_df = validator.generate_health_report(symbol, start_date, end_date)

    if report_df.empty:
        console.print("[yellow]No data found in date range[/yellow]")
        return

    # Display table
    table = Table(title=f"Health Report: {symbol}")
    table.add_column("Date", style="cyan")
    table.add_column("Options", justify="right")
    table.add_column("Liquid", justify="right")
    table.add_column("Expirations", justify="right")
    table.add_column("Surfaces", justify="right")
    table.add_column("Avg RMSE", justify="right")
    table.add_column("Data", justify="center")
    table.add_column("Fits", justify="center")

    for _, row in report_df.iterrows():
        data_status = "[green]OK[/green]" if row["data_healthy"] else "[red]![/red]"
        fits_status = "[green]OK[/green]" if row["fits_healthy"] else "[red]![/red]"
        rmse_str = f"{row['avg_rmse']*100:.3f}%" if row["avg_rmse"] else "N/A"

        table.add_row(
            str(row["date"]),
            str(row["total_options"]),
            str(row["liquid_options"]),
            str(row["num_expirations"]),
            str(row["num_surfaces"]),
            rmse_str,
            data_status,
            fits_status,
        )

    console.print(table)

    # Summary
    healthy_days = report_df["data_healthy"].sum()
    total_days = len(report_df)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Days with data: {total_days}")
    console.print(f"  Healthy days: {healthy_days} ({healthy_days/total_days*100:.1f}%)")


@app.command()
def check_gaps(
    symbol: str = typer.Argument("SPY", help="Symbol to check"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
) -> None:
    """Find missing trading days in the data."""
    from volsurf.utils.validation import check_data_gaps

    start_date = parse_date(start)
    end_date = parse_date(end)

    missing = check_data_gaps(symbol, start_date, end_date)

    if not missing:
        console.print(f"[green]No data gaps found for {symbol}[/green]")
    else:
        console.print(f"[yellow]Found {len(missing)} missing trading days:[/yellow]")
        for d in missing:
            console.print(f"  - {d}")


@app.command()
def export_options(
    symbol: str = typer.Argument("SPY", help="Symbol to export"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to export (YYYY-MM-DD), defaults to latest"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (without extension)"
    ),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv or json"),
    all_options: bool = typer.Option(
        False, "--all", help="Export all options, not just liquid"
    ),
) -> None:
    """Export options chain data to file."""
    from volsurf.database.connection import get_connection
    from volsurf.utils.export import export_options_chain

    if target_date:
        quote_date = parse_date(target_date)
    else:
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    # Default output path
    if output is None:
        output = f"output/{symbol}_{quote_date.strftime('%Y%m%d')}_options"

    file_path = export_options_chain(
        symbol, quote_date, output, format=format, liquid_only=not all_options
    )
    console.print(f"[green]Exported options to {file_path}[/green]")


@app.command()
def export_surfaces(
    symbol: str = typer.Argument("SPY", help="Symbol to export"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to export (YYYY-MM-DD), defaults to latest"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (without extension)"
    ),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv or json"),
) -> None:
    """Export fitted surfaces to file."""
    from volsurf.database.connection import get_connection
    from volsurf.utils.export import export_fitted_surfaces

    if target_date:
        quote_date = parse_date(target_date)
    else:
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(quote_date) FROM fitted_surfaces WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No fitted surfaces found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    if output is None:
        output = f"output/{symbol}_{quote_date.strftime('%Y%m%d')}_surfaces"

    file_path = export_fitted_surfaces(symbol, quote_date, output, format=format)
    console.print(f"[green]Exported surfaces to {file_path}[/green]")


@app.command()
def export_snapshot(
    symbol: str = typer.Argument("SPY", help="Symbol to export"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to export (YYYY-MM-DD), defaults to latest"
    ),
    output_dir: str = typer.Option(
        "output", "--output-dir", "-o", help="Output directory"
    ),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv or json"),
) -> None:
    """Export complete snapshot of all data for a date."""
    from volsurf.database.connection import get_connection
    from volsurf.utils.export import export_full_snapshot

    if target_date:
        quote_date = parse_date(target_date)
    else:
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    paths = export_full_snapshot(symbol, quote_date, output_dir, format=format)

    console.print(f"[green]Exported full snapshot:[/green]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")
