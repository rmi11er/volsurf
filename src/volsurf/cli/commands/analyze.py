"""Analytics CLI commands."""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from volsurf.analytics import (
    RealizedVolCalculator,
    SurfaceMetrics,
    TermStructureAnalyzer,
    VRPCalculator,
    plot_atm_vol_timeseries,
    plot_iv_vs_rv,
    plot_term_structure,
    plot_vrp_timeseries,
    print_ascii_timeseries,
    print_realized_vol_table,
    print_surface_summary,
    print_term_structure_table,
    print_vrp_summary,
)
from volsurf.database.schema import init_schema

app = typer.Typer(help="Analytics commands")
console = Console()


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string to date object."""
    if date_str is None:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@app.command("realized-vol")
def realized_vol(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to calculate (YYYY-MM-DD), defaults to latest"
    ),
    backfill: bool = typer.Option(
        False, "--backfill", "-b", help="Backfill realized vol for date range"
    ),
    start_date: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date for backfill (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date for backfill (YYYY-MM-DD)"
    ),
) -> None:
    """Calculate realized volatility from underlying prices."""
    init_schema()
    calculator = RealizedVolCalculator()

    if backfill:
        if not start_date or not end_date:
            console.print("[red]--start and --end required for backfill[/red]")
            raise typer.Exit(1)

        start = parse_date(start_date)
        end = parse_date(end_date)

        with console.status(f"Calculating realized vol for {symbol}..."):
            count = calculator.backfill(symbol, start, end, store=True)

        console.print(f"[green]Calculated and stored realized vol for {count} dates[/green]")
    else:
        # Single date calculation
        if target_date:
            calc_date = parse_date(target_date)
        else:
            # Get latest date from underlying prices
            metrics = SurfaceMetrics()
            calc_date = metrics.get_latest_date(symbol)
            if calc_date is None:
                console.print(f"[yellow]No data found for {symbol}[/yellow]")
                raise typer.Exit(1)

        result = calculator.calculate_for_date(symbol, calc_date)
        print_realized_vol_table(result)


@app.command()
def vrp(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to analyze (YYYY-MM-DD), defaults to latest"
    ),
    backfill: bool = typer.Option(
        False, "--backfill", "-b", help="Backfill VRP for date range"
    ),
    start_date: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    timeseries: bool = typer.Option(
        False, "--timeseries", "-t", help="Show VRP time series"
    ),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate interactive plot"),
) -> None:
    """Calculate variance risk premium metrics."""
    init_schema()
    calculator = VRPCalculator()

    if backfill:
        if not start_date or not end_date:
            console.print("[red]--start and --end required for backfill[/red]")
            raise typer.Exit(1)

        start = parse_date(start_date)
        end = parse_date(end_date)

        with console.status(f"Calculating VRP for {symbol}..."):
            count = calculator.backfill(symbol, start, end, store=True)

        console.print(f"[green]Calculated and stored VRP for {count} dates[/green]")

    elif timeseries:
        if not start_date or not end_date:
            console.print("[red]--start and --end required for timeseries[/red]")
            raise typer.Exit(1)

        start = parse_date(start_date)
        end = parse_date(end_date)

        df = calculator.get_vrp_timeseries(symbol, start, end)

        if df.empty:
            console.print(f"[yellow]No VRP data for {symbol} in range[/yellow]")
            raise typer.Exit(1)

        print_vrp_summary(df)
        print_ascii_timeseries(
            df, "date", "vrp_30d", f"{symbol} VRP (30-Day)", color="green"
        )

        if plot:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plot_vrp_timeseries(
                df,
                title=f"{symbol} Variance Risk Premium",
                save_path=str(output_dir / f"vrp_{symbol}.html"),
            )

    else:
        # Single date
        metrics = SurfaceMetrics()
        if target_date:
            calc_date = parse_date(target_date)
        else:
            calc_date = metrics.get_latest_date(symbol)
            if calc_date is None:
                console.print(f"[yellow]No data found for {symbol}[/yellow]")
                raise typer.Exit(1)

        result = calculator.calculate_for_date(symbol, calc_date)

        console.print(f"\n[bold]VRP for {symbol} on {calc_date}[/bold]")
        console.print("-" * 40)
        if result.vrp_30d is not None:
            console.print(f"VRP (30-day):  {result.vrp_30d:.2%}")
            console.print(f"  IV (30d):    {result.implied_vol_30d:.2%}")
            console.print(f"  RV (21d):    {result.realized_vol_30d:.2%}")
        else:
            console.print("[yellow]Insufficient data to calculate VRP[/yellow]")

        if result.vrp_60d is not None:
            console.print(f"VRP (60-day):  {result.vrp_60d:.2%}")
        if result.vrp_90d is not None:
            console.print(f"VRP (90-day):  {result.vrp_90d:.2%}")


@app.command("surface-summary")
def surface_summary(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to summarize (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Show summary of fitted surface for a date."""
    init_schema()
    metrics = SurfaceMetrics()

    if target_date:
        calc_date = parse_date(target_date)
    else:
        calc_date = metrics.get_latest_date(symbol)
        if calc_date is None:
            console.print(f"[yellow]No fitted surfaces found for {symbol}[/yellow]")
            raise typer.Exit(1)

    summary = metrics.get_surface_summary(symbol, calc_date)

    if summary.num_surfaces == 0:
        console.print(f"[yellow]No surfaces found for {symbol} on {calc_date}[/yellow]")
        raise typer.Exit(1)

    print_surface_summary(summary)


@app.command("atm-vol")
def atm_vol(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    start_date: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    tenor: int = typer.Option(30, "--tenor", "-t", help="Target tenor in days"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate interactive plot"),
) -> None:
    """Show ATM implied volatility time series."""
    init_schema()
    metrics = SurfaceMetrics()

    # Get date range
    if not start_date or not end_date:
        available_dates = metrics.get_available_dates(symbol)
        if not available_dates:
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            raise typer.Exit(1)
        start = available_dates[0]
        end = available_dates[-1]
    else:
        start = parse_date(start_date)
        end = parse_date(end_date)

    df = metrics.get_atm_vol_timeseries(symbol, start, end, tte_target_days=tenor)

    if df.empty:
        console.print(f"[yellow]No ATM vol data for {symbol} in range[/yellow]")
        raise typer.Exit(1)

    title = f"{symbol} {tenor}-Day ATM Implied Volatility"
    print_ascii_timeseries(df, "date", "atm_vol", title, color="cyan")

    if plot:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_atm_vol_timeseries(
            df,
            title=title,
            save_path=str(output_dir / f"atm_vol_{symbol}_{tenor}d.html"),
        )
        console.print(f"[green]Plot saved to output/atm_vol_{symbol}_{tenor}d.html[/green]")


@app.command("term-structure")
def term_structure(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to analyze (YYYY-MM-DD), defaults to latest"
    ),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate interactive plot"),
    fit: bool = typer.Option(False, "--fit", "-f", help="Fit and store term structure model"),
) -> None:
    """Show volatility term structure for a date."""
    init_schema()
    analyzer = TermStructureAnalyzer()
    metrics = SurfaceMetrics()

    if target_date:
        calc_date = parse_date(target_date)
    else:
        calc_date = metrics.get_latest_date(symbol)
        if calc_date is None:
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            raise typer.Exit(1)

    # Get term structure data
    data = analyzer.get_term_structure_data(symbol, calc_date)

    if not data:
        console.print(f"[yellow]No term structure data for {symbol} on {calc_date}[/yellow]")
        raise typer.Exit(1)

    print_term_structure_table(data, title=f"{symbol} Term Structure ({calc_date})")

    # Optionally fit and store model
    if fit:
        result = analyzer.analyze_date(symbol, calc_date)
        if result.atm_fit:
            analyzer.store_result(result)
            console.print(f"\n[bold]ATM Term Structure Fit:[/bold]")
            console.print(f"  σ(T) = {result.atm_fit.a:.4f} × T^{result.atm_fit.b:.4f}")
            console.print(f"  RMSE: {result.atm_fit.rmse:.4%}")
        if result.skew_fit:
            console.print(f"\n[bold]Skew Term Structure Fit:[/bold]")
            console.print(f"  Skew(T) = {result.skew_fit.a:.4f} × T^{result.skew_fit.b:.4f}")
            console.print(f"  RMSE: {result.skew_fit.rmse:.4%}")

    if plot:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_term_structure(
            data,
            title=f"{symbol} Volatility Term Structure ({calc_date})",
            save_path=str(output_dir / f"term_structure_{symbol}_{calc_date}.html"),
        )
        console.print(f"[green]Plot saved to output/term_structure_{symbol}_{calc_date}.html[/green]")


@app.command()
def skew(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    start_date: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    tenor: int = typer.Option(30, "--tenor", "-t", help="Target tenor in days"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate interactive plot"),
) -> None:
    """Show 25-delta skew time series."""
    init_schema()
    metrics = SurfaceMetrics()

    # Get date range
    if not start_date or not end_date:
        available_dates = metrics.get_available_dates(symbol)
        if not available_dates:
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            raise typer.Exit(1)
        start = available_dates[0]
        end = available_dates[-1]
    else:
        start = parse_date(start_date)
        end = parse_date(end_date)

    df = metrics.get_skew_timeseries(symbol, start, end, tte_target_days=tenor)

    if df.empty:
        console.print(f"[yellow]No skew data for {symbol} in range[/yellow]")
        raise typer.Exit(1)

    title = f"{symbol} {tenor}-Day 25-Delta Skew"
    print_ascii_timeseries(df, "date", "skew", title, color="red")


@app.command("iv-vs-rv")
def iv_vs_rv(
    symbol: str = typer.Argument("SPY", help="Symbol to analyze"),
    start_date: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    plot: bool = typer.Option(True, "--plot/--no-plot", "-p", help="Generate plot"),
) -> None:
    """Compare implied vs realized volatility."""
    init_schema()
    metrics = SurfaceMetrics()
    rv_calc = RealizedVolCalculator()

    # Get date range
    if not start_date or not end_date:
        available_dates = metrics.get_available_dates(symbol)
        if not available_dates:
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            raise typer.Exit(1)
        start = available_dates[0]
        end = available_dates[-1]
    else:
        start = parse_date(start_date)
        end = parse_date(end_date)

    # Get IV (30-day ATM)
    iv_df = metrics.get_atm_vol_timeseries(symbol, start, end, tte_target_days=30)

    # Get RV (21-day)
    rv_df = rv_calc.get_vol_timeseries(symbol, start, end, metric="rv_21d")

    if iv_df.empty or rv_df.empty:
        console.print(f"[yellow]Insufficient IV or RV data for comparison[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{symbol} IV vs RV Comparison ({start} to {end})[/bold]")
    console.print("-" * 50)
    console.print(f"IV (30d ATM):  Mean {iv_df['atm_vol'].mean():.2%}, Latest {iv_df['atm_vol'].iloc[-1]:.2%}")
    console.print(f"RV (21d):      Mean {rv_df['realized_vol'].mean():.2%}, Latest {rv_df['realized_vol'].iloc[-1]:.2%}")

    if plot:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_iv_vs_rv(
            iv_df,
            rv_df,
            title=f"{symbol} Implied vs Realized Volatility",
            save_path=str(output_dir / f"iv_vs_rv_{symbol}.html"),
        )
        console.print(f"[green]Plot saved to output/iv_vs_rv_{symbol}.html[/green]")
