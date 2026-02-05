"""Intraday surface fitting CLI commands."""

from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

app = typer.Typer(help="Intraday surface commands")
console = Console()


def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)


def parse_time(time_str: str) -> time:
    """Parse time string to time object."""
    try:
        # Try HH:MM:SS format first
        if len(time_str.split(":")) == 3:
            return datetime.strptime(time_str, "%H:%M:%S").time()
        # Fall back to HH:MM
        return datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        console.print(f"[red]Invalid time format: {time_str}. Use HH:MM or HH:MM:SS[/red]")
        raise typer.Exit(1)


@app.command()
def surface(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: str = typer.Option(
        ..., "--date", "-d", help="Date (YYYY-MM-DD)"
    ),
    target_time: str = typer.Option(
        ..., "--time", "-t", help="Time (HH:MM or HH:MM:SS) in ET"
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use mock data for testing"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Export to Parquet file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
) -> None:
    """
    Fit SVI volatility surfaces at a specific intraday timestamp.

    Example:
        volsurf intraday surface SPY -d 2024-12-20 -t 10:30
    """
    from volsurf.fitting.intraday_pipeline import (
        IntradaySurfacePipeline,
        IntradayPipelineConfig,
    )

    quote_date = parse_date(target_date)
    time_of_day = parse_time(target_time)
    timestamp = datetime.combine(quote_date, time_of_day)

    console.print(
        f"Fitting intraday surfaces for [cyan]{symbol}[/cyan] at "
        f"[cyan]{timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/cyan] ET"
    )

    if mock:
        console.print("[yellow]Using mock data[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching quotes and fitting surfaces...", total=None)

        config = IntradayPipelineConfig()
        pipeline = IntradaySurfacePipeline(config, use_mock=mock)

        try:
            result = pipeline.fit_at_time(symbol, quote_date, time_of_day)
        finally:
            pipeline.close()

        progress.update(task, completed=True)

    # Display results
    if not result.surfaces:
        console.print("[yellow]No surfaces fitted (insufficient data)[/yellow]")
        return

    # Summary stats
    console.print(f"\n[bold]Underlying:[/bold] ${float(result.underlying_price):.2f}")
    console.print(
        f"[bold]Fitted:[/bold] {result.successful_fits}/{result.total_expirations} expirations"
    )
    if result.avg_rmse:
        console.print(f"[bold]Avg RMSE:[/bold] {result.avg_rmse*100:.3f}%")

    # Surface table
    table = Table(title=f"Intraday Surfaces at {timestamp.strftime('%H:%M:%S')} ET")
    table.add_column("Expiration", style="cyan")
    table.add_column("TTE (days)", justify="right")
    table.add_column("ATM Vol", justify="right", style="green")
    table.add_column("Skew (25d)", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("Arb-Free", justify="center")

    for s in result.surfaces:
        tte_days = int(s.tte_years * 365)
        arb_status = "[green]Y[/green]" if s.passes_no_arbitrage else "[red]N[/red]"
        skew_str = f"{s.skew_25delta*100:.2f}%" if s.skew_25delta else "N/A"

        table.add_row(
            str(s.expiration_date),
            str(tte_days),
            f"{s.atm_vol*100:.2f}%",
            skew_str,
            f"{s.rmse*100:.3f}%",
            str(s.num_points),
            arb_status,
        )

    console.print(table)

    # Show SVI params if verbose
    if verbose:
        console.print("\n[bold]SVI Parameters:[/bold]")
        for s in result.surfaces[:5]:  # Limit to first 5
            params = s.svi_params
            console.print(
                f"  {s.expiration_date}: "
                f"a={params.a:.6f}, b={params.b:.4f}, "
                f"rho={params.rho:.4f}, m={params.m:.4f}, sigma={params.sigma:.4f}"
            )
        if len(result.surfaces) > 5:
            console.print(f"  ... and {len(result.surfaces) - 5} more")

    # Export if requested
    if output:
        output_path = Path(output)
        result.to_parquet(str(output_path))
        console.print(f"\n[green]Exported to {output_path}[/green]")


@app.command("time-series")
def time_series(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: str = typer.Option(
        ..., "--date", "-d", help="Date (YYYY-MM-DD)"
    ),
    start_time: str = typer.Option(
        "09:35", "--start", help="Start time (HH:MM)"
    ),
    end_time: str = typer.Option(
        "15:55", "--end", help="End time (HH:MM)"
    ),
    interval: int = typer.Option(
        30, "--interval", "-i", help="Interval in minutes"
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use mock data for testing"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Export directory for Parquet files"
    ),
) -> None:
    """
    Fit surfaces at regular intervals throughout the trading day.

    Example:
        volsurf intraday time-series SPY -d 2024-12-20 -i 30
    """
    from volsurf.fitting.intraday_pipeline import (
        IntradaySurfacePipeline,
        IntradayPipelineConfig,
    )

    quote_date = parse_date(target_date)
    start = parse_time(start_time)
    end = parse_time(end_time)

    console.print(
        f"Fitting time series for [cyan]{symbol}[/cyan] on [cyan]{quote_date}[/cyan]"
    )
    console.print(
        f"  From [cyan]{start.strftime('%H:%M')}[/cyan] to "
        f"[cyan]{end.strftime('%H:%M')}[/cyan] every [cyan]{interval}[/cyan] min"
    )

    if mock:
        console.print("[yellow]Using mock data[/yellow]")

    config = IntradayPipelineConfig()
    pipeline = IntradaySurfacePipeline(config, use_mock=mock)

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            task = progress.add_task("Processing...", total=None)
            results = pipeline.fit_time_series(
                symbol, quote_date, start, end, interval
            )
            progress.update(task, completed=True)
        finally:
            pipeline.close()

    if not results:
        console.print("[yellow]No results generated[/yellow]")
        return

    # Summary table
    table = Table(title=f"Time Series Summary for {symbol} on {quote_date}")
    table.add_column("Time", style="cyan")
    table.add_column("Underlying", justify="right")
    table.add_column("Fits", justify="right")
    table.add_column("Avg ATM Vol", justify="right", style="green")
    table.add_column("Avg RMSE", justify="right")

    for r in results:
        time_str = r.timestamp.strftime("%H:%M")
        underlying = f"${float(r.underlying_price):.2f}"
        fits = f"{r.successful_fits}/{r.total_expirations}"

        if r.surfaces:
            avg_atm = sum(s.atm_vol for s in r.surfaces) / len(r.surfaces)
            avg_atm_str = f"{avg_atm*100:.2f}%"
        else:
            avg_atm_str = "N/A"

        rmse_str = f"{r.avg_rmse*100:.3f}%" if r.avg_rmse else "N/A"

        table.add_row(time_str, underlying, fits, avg_atm_str, rmse_str)

    console.print(table)

    # Export if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for r in results:
            if r.surfaces:
                filename = f"{symbol}_{quote_date}_{r.timestamp.strftime('%H%M')}.parquet"
                r.to_parquet(str(output_path / filename))

        console.print(f"\n[green]Exported {len(results)} files to {output_path}[/green]")


@app.command("batch")
def batch(
    symbol: str = typer.Argument("SPY", help="Symbol to fetch"),
    start_date: str = typer.Option(
        ..., "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        ..., "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Interval in minutes (default: 60 for hourly)"
    ),
    start_time: str = typer.Option(
        "09:35", "--start-time", help="Daily start time (HH:MM)"
    ),
    end_time: str = typer.Option(
        "15:55", "--end-time", help="Daily end time (HH:MM)"
    ),
    output_dir: str = typer.Option(
        "data/intraday", "--output-dir", "-o", help="Output directory for Parquet files"
    ),
    aggregate_daily: bool = typer.Option(
        False, "--aggregate-daily", "-a", help="Store all timestamps for each day in a single file"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Skip dates/timestamps that already have data"
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use mock data for testing"
    ),
) -> None:
    """
    Batch fetch and fit intraday surfaces for a date range.

    Saves fitted surfaces to Parquet files organized by date.

    Examples:
        volsurf intraday batch SPY -s 2024-12-16 -e 2024-12-20 -i 60
        volsurf intraday batch SPY -s 2024-12-02 -e 2024-12-20 -i 1 --aggregate-daily
        volsurf intraday batch SPY -s 2024-12-02 -e 2024-12-20 -i 1 -a --resume
    """
    from datetime import timedelta
    import polars as pl
    from volsurf.fitting.intraday_pipeline import (
        IntradaySurfacePipeline,
        IntradayPipelineConfig,
    )
    from volsurf.ingestion.terminal import ensure_terminal_running
    from volsurf.config.settings import get_settings

    start = parse_date(start_date)
    end = parse_date(end_date)
    daily_start = parse_time(start_time)
    daily_end = parse_time(end_time)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate expected timestamps per day and total
    start_minutes = daily_start.hour * 60 + daily_start.minute
    end_minutes = daily_end.hour * 60 + daily_end.minute
    timestamps_per_day = ((end_minutes - start_minutes) // interval) + 1

    # Count trading days
    trading_days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)

    total_timestamps = len(trading_days) * timestamps_per_day

    console.print(f"[bold]Batch Intraday Surface Fitting[/bold]")
    console.print(f"  Symbol: [cyan]{symbol}[/cyan]")
    console.print(f"  Date range: [cyan]{start}[/cyan] to [cyan]{end}[/cyan] ({len(trading_days)} trading days)")
    console.print(f"  Interval: [cyan]{interval}[/cyan] minutes ({timestamps_per_day} per day, {total_timestamps} total)")
    console.print(f"  Daily hours: [cyan]{daily_start.strftime('%H:%M')}[/cyan] - [cyan]{daily_end.strftime('%H:%M')}[/cyan]")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")
    if aggregate_daily:
        console.print(f"  Mode: [cyan]Daily aggregation[/cyan] (one file per day)")
    if resume:
        console.print(f"  Resume: [cyan]Enabled[/cyan] (skipping existing data)")

    # Check terminal or use mock
    settings = get_settings()
    if not mock and not settings.use_mock_data:
        console.print("\n[dim]Checking Theta Terminal...[/dim]")
        if not ensure_terminal_running():
            from volsurf.ingestion.terminal import get_terminal_diagnostics
            diagnostics = get_terminal_diagnostics()
            console.print(f"[red]Failed to connect to Theta Terminal[/red]")
            console.print(f"[red]Status: {diagnostics.status.value}[/red]")
            console.print(f"[red]Message: {diagnostics.message}[/red]")

            # Show environment info
            console.print("\n[yellow]Environment:[/yellow]")
            console.print(f"  JAR exists: {diagnostics.jar_exists} ({diagnostics.jar_path})")
            console.print(f"  Credentials exist: {diagnostics.creds_exists}")
            console.print(f"  Config exists: {diagnostics.config_exists}")
            console.print(f"  Java available: {diagnostics.java_available}")
            if diagnostics.java_version:
                console.print(f"  Java version: {diagnostics.java_version}")

            # Show process output if available
            if diagnostics.process_stderr:
                console.print("\n[yellow]Terminal stderr (last 10 lines):[/yellow]")
                for line in diagnostics.process_stderr[-10:]:
                    console.print(f"  {line}")
            if diagnostics.process_stdout:
                console.print("\n[yellow]Terminal stdout (last 10 lines):[/yellow]")
                for line in diagnostics.process_stdout[-10:]:
                    console.print(f"  {line}")

            if diagnostics.connection_error:
                console.print(f"\n[red]Connection error: {diagnostics.connection_error}[/red]")

            console.print("\n[dim]Use --mock for testing without the terminal.[/dim]")
            raise typer.Exit(1)
        console.print("[green]Theta Terminal connected[/green]")
    elif mock or settings.use_mock_data:
        console.print("[yellow]Using mock data[/yellow]")
        mock = True

    config = IntradayPipelineConfig()
    pipeline = IntradaySurfacePipeline(config, use_mock=mock)

    total_fits = 0
    total_days = 0
    skipped_days = 0
    failed_days = []

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "Processing...", total=len(trading_days)
            )

            for current_date in trading_days:
                day_dir = output_path / current_date.strftime("%Y-%m-%d")

                # Check if we should skip this day (resume mode)
                if resume:
                    if aggregate_daily:
                        daily_file = day_dir / f"{symbol}_daily.parquet"
                        if daily_file.exists():
                            skipped_days += 1
                            progress.update(main_task, advance=1, description=f"Skipped {current_date}")
                            continue
                    else:
                        # Check if all expected timestamps exist
                        if day_dir.exists():
                            existing_files = list(day_dir.glob(f"{symbol}_*.parquet"))
                            if len(existing_files) >= timestamps_per_day:
                                skipped_days += 1
                                progress.update(main_task, advance=1, description=f"Skipped {current_date}")
                                continue

                progress.update(main_task, description=f"Processing {current_date}...")

                try:
                    results = pipeline.fit_time_series(
                        symbol,
                        current_date,
                        daily_start,
                        daily_end,
                        interval,
                    )

                    if results:
                        day_dir.mkdir(parents=True, exist_ok=True)

                        if aggregate_daily:
                            # Aggregate all timestamps into single file
                            all_records = []
                            for r in results:
                                if r.surfaces:
                                    for s in r.surfaces:
                                        all_records.append({
                                            "symbol": s.symbol,
                                            "timestamp": s.timestamp,
                                            "expiration_date": s.expiration_date,
                                            "tte_years": s.tte_years,
                                            "forward_price": float(s.forward_price),
                                            "underlying_price": float(r.underlying_price),
                                            "svi_a": s.svi_params.a,
                                            "svi_b": s.svi_params.b,
                                            "svi_rho": s.svi_params.rho,
                                            "svi_m": s.svi_params.m,
                                            "svi_sigma": s.svi_params.sigma,
                                            "atm_vol": s.atm_vol,
                                            "skew_25delta": s.skew_25delta,
                                            "rmse": s.rmse,
                                            "mae": s.mae,
                                            "max_error": s.max_error,
                                            "num_points": s.num_points,
                                            "passes_no_arbitrage": s.passes_no_arbitrage,
                                        })
                            if all_records:
                                df = pl.DataFrame(all_records)
                                daily_file = day_dir / f"{symbol}_daily.parquet"
                                df.write_parquet(str(daily_file))
                                total_fits += len([r for r in results if r.surfaces])
                        else:
                            # Save each timestamp separately
                            for r in results:
                                if r.surfaces:
                                    filename = f"{symbol}_{r.timestamp.strftime('%H%M')}.parquet"
                                    r.to_parquet(str(day_dir / filename))
                                    total_fits += 1

                        total_days += 1

                except Exception as e:
                    console.print(f"\n[red]Failed {current_date}: {e}[/red]")
                    failed_days.append(current_date)

                progress.update(main_task, advance=1)

    finally:
        pipeline.close()

    # Summary
    console.print("\n[bold]Batch Complete[/bold]")
    console.print(f"  Days processed: [green]{total_days}[/green]")
    if skipped_days > 0:
        console.print(f"  Days skipped (resume): [yellow]{skipped_days}[/yellow]")
    console.print(f"  Total surface sets: [green]{total_fits}[/green]")
    console.print(f"  Output directory: [cyan]{output_path}[/cyan]")

    if failed_days:
        console.print(f"  [red]Failed days: {len(failed_days)}[/red]")
        for d in failed_days[:5]:
            console.print(f"    - {d}")
        if len(failed_days) > 5:
            console.print(f"    ... and {len(failed_days) - 5} more")


@app.command("compare-eod")
def compare_eod(
    symbol: str = typer.Argument("SPY", help="Symbol to compare"),
    target_date: str = typer.Option(
        ..., "--date", "-d", help="Date (YYYY-MM-DD)"
    ),
    target_time: str = typer.Option(
        ..., "--time", "-t", help="Intraday time (HH:MM) to compare"
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use mock data for testing"
    ),
) -> None:
    """
    Compare intraday surface to EOD surface for the same date.

    Example:
        volsurf intraday compare-eod SPY -d 2024-12-20 -t 10:30
    """
    from volsurf.database.connection import get_connection
    from volsurf.fitting.intraday_pipeline import (
        IntradaySurfacePipeline,
        IntradayPipelineConfig,
    )

    quote_date = parse_date(target_date)
    time_of_day = parse_time(target_time)
    timestamp = datetime.combine(quote_date, time_of_day)

    console.print(
        f"Comparing [cyan]{timestamp.strftime('%H:%M')}[/cyan] vs EOD for "
        f"[cyan]{symbol}[/cyan] on [cyan]{quote_date}[/cyan]"
    )

    # Fetch intraday surface
    if mock:
        console.print("[yellow]Using mock data for intraday[/yellow]")

    config = IntradayPipelineConfig()
    pipeline = IntradaySurfacePipeline(config, use_mock=mock)

    try:
        intraday_result = pipeline.fit_at_time(symbol, quote_date, time_of_day)
    finally:
        pipeline.close()

    if not intraday_result.surfaces:
        console.print("[yellow]No intraday surfaces fitted[/yellow]")
        return

    # Fetch EOD surfaces from database
    conn = get_connection()
    eod_query = """
    SELECT expiration_date, atm_vol, skew_25delta, rmse
    FROM fitted_surfaces
    WHERE symbol = ? AND quote_date = ?
    ORDER BY expiration_date
    """
    eod_df = conn.execute(eod_query, [symbol, quote_date]).fetchdf()

    if eod_df.empty:
        console.print("[yellow]No EOD surfaces found in database for this date[/yellow]")
        console.print("Run 'volsurf fit surfaces' first to generate EOD surfaces")
        return

    # Build comparison table
    table = Table(title=f"Intraday vs EOD Comparison")
    table.add_column("Expiration", style="cyan")
    table.add_column("Intraday ATM", justify="right")
    table.add_column("EOD ATM", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Intraday Skew", justify="right")
    table.add_column("EOD Skew", justify="right")

    # Create lookup for intraday surfaces
    intraday_by_exp = {s.expiration_date: s for s in intraday_result.surfaces}

    for _, row in eod_df.iterrows():
        exp_date = row["expiration_date"]
        if hasattr(exp_date, 'date'):
            exp_date = exp_date.date()

        eod_atm = row["atm_vol"]
        eod_skew = row["skew_25delta"]

        if exp_date in intraday_by_exp:
            intra = intraday_by_exp[exp_date]
            intra_atm = intra.atm_vol
            intra_skew = intra.skew_25delta

            diff = (intra_atm - eod_atm) * 100
            diff_str = f"{diff:+.2f}%" if diff else "0.00%"
            diff_color = "[green]" if diff < 0 else "[red]" if diff > 0 else ""

            table.add_row(
                str(exp_date),
                f"{intra_atm*100:.2f}%",
                f"{eod_atm*100:.2f}%",
                f"{diff_color}{diff_str}[/]" if diff_color else diff_str,
                f"{intra_skew*100:.2f}%" if intra_skew else "N/A",
                f"{eod_skew*100:.2f}%" if eod_skew else "N/A",
            )
        else:
            table.add_row(
                str(exp_date),
                "[dim]N/A[/dim]",
                f"{eod_atm*100:.2f}%",
                "[dim]N/A[/dim]",
                "[dim]N/A[/dim]",
                f"{eod_skew*100:.2f}%" if eod_skew else "N/A",
            )

    console.print(table)


@app.command("diagnose")
def diagnose() -> None:
    """
    Diagnose Theta Terminal connection issues.

    Shows detailed information about the terminal setup and any connection errors.

    Example:
        volsurf intraday diagnose
    """
    from volsurf.ingestion.terminal import (
        ThetaTerminalManager,
        TerminalStatus,
    )

    console.print("[bold]Theta Terminal Diagnostics[/bold]\n")

    # Get diagnostics (this will attempt connection)
    manager = ThetaTerminalManager.get_instance()
    diagnostics = manager.get_diagnostics()

    # Status
    status_color = "green" if diagnostics.status == TerminalStatus.RUNNING else "red"
    console.print(f"[bold]Status:[/bold] [{status_color}]{diagnostics.status.value}[/{status_color}]")
    console.print(f"[bold]Message:[/bold] {diagnostics.message}\n")

    # Environment
    console.print("[bold]Environment:[/bold]")
    jar_status = "[green]✓[/green]" if diagnostics.jar_exists else "[red]✗[/red]"
    console.print(f"  {jar_status} JAR file: {diagnostics.jar_path}")

    creds_status = "[green]✓[/green]" if diagnostics.creds_exists else "[red]✗[/red]"
    console.print(f"  {creds_status} Credentials: {diagnostics.creds_path}")

    config_status = "[green]✓[/green]" if diagnostics.config_exists else "[yellow]○[/yellow]"
    console.print(f"  {config_status} Config: {diagnostics.config_path}")

    java_status = "[green]✓[/green]" if diagnostics.java_available else "[red]✗[/red]"
    java_info = diagnostics.java_version if diagnostics.java_version else "not found"
    console.print(f"  {java_status} Java: {java_info}")

    # Process info
    console.print("\n[bold]Process:[/bold]")
    if diagnostics.process_running:
        console.print("  [green]Running[/green]")
    elif diagnostics.process_exit_code is not None:
        console.print(f"  [red]Exited with code {diagnostics.process_exit_code}[/red]")
    else:
        console.print("  [dim]Not started[/dim]")

    # Connection info
    if diagnostics.connection_error:
        console.print(f"\n[bold]Connection Error:[/bold]\n  [red]{diagnostics.connection_error}[/red]")

    if diagnostics.http_status_code:
        console.print(f"\n[bold]HTTP Status:[/bold] {diagnostics.http_status_code}")

    # Process output
    if diagnostics.process_stderr:
        console.print("\n[bold]Terminal stderr:[/bold]")
        for line in diagnostics.process_stderr[-15:]:
            console.print(f"  {line}")

    if diagnostics.process_stdout:
        console.print("\n[bold]Terminal stdout:[/bold]")
        for line in diagnostics.process_stdout[-10:]:
            console.print(f"  {line}")

    # Suggestions
    if diagnostics.status != TerminalStatus.RUNNING:
        console.print("\n[bold]Suggestions:[/bold]")
        if not diagnostics.jar_exists:
            console.print("  • Download ThetaTerminal.jar and place it in vendor/")
        if not diagnostics.creds_exists:
            console.print("  • Create vendor/creds.txt with your Theta Data credentials")
            console.print("    (email on line 1, password on line 2)")
        if not diagnostics.java_available:
            console.print("  • Install Java 21 or later and ensure it's in your PATH")
        if diagnostics.status == TerminalStatus.CONNECTION_REFUSED:
            console.print("  • The terminal process may need more time to start")
            console.print("  • Check if another process is using port 25510")
        if diagnostics.status == TerminalStatus.AUTH_FAILED:
            console.print("  • Verify your credentials in vendor/creds.txt")
            console.print("  • Check your Theta Data subscription status")
        if diagnostics.status == TerminalStatus.SESSION_EXPIRED:
            console.print("  • The terminal session has expired or is invalid")
            console.print("  • Restart the Theta Terminal application")
            console.print("  • If running externally, close and reopen ThetaTerminal.jar")
