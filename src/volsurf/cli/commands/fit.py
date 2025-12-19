"""Surface fitting CLI commands."""

from datetime import date, datetime
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Surface fitting commands")
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
def surfaces(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to fit (YYYY-MM-DD), defaults to latest"
    ),
    no_store: bool = typer.Option(
        False, "--no-store", help="Don't store results in database"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Fit SVI volatility surfaces for a specific date."""
    from volsurf.database.connection import get_connection
    from volsurf.fitting.pipeline import SurfaceFittingPipeline, PipelineConfig

    # Get the quote date
    if target_date:
        quote_date = parse_date(target_date)
    else:
        # Get latest date from database
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    console.print(f"Fitting SVI surfaces for [cyan]{symbol}[/cyan] on [cyan]{quote_date}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fitting surfaces...", total=None)

        config = PipelineConfig()
        pipeline = SurfaceFittingPipeline(config)
        surfaces, stats = pipeline.fit_date(
            symbol, quote_date, store_results=not no_store
        )

        progress.update(task, completed=True)

    # Display results
    if not surfaces:
        console.print("[yellow]No surfaces fitted (insufficient data)[/yellow]")
        return

    # Summary table
    table = Table(title=f"Fitted Surfaces for {symbol} on {quote_date}")
    table.add_column("Expiration", style="cyan")
    table.add_column("TTE (days)", justify="right")
    table.add_column("ATM Vol", justify="right", style="green")
    table.add_column("Skew (25Δ)", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("Arb-Free", justify="center")

    for s in surfaces:
        tte_days = int(s.tte_years * 365)
        arb_status = "[green]✓[/green]" if s.passes_no_arbitrage else "[red]✗[/red]"
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

    # Summary stats
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Expirations fitted: {stats.expirations_processed}")
    console.print(f"  Expirations skipped: {stats.expirations_skipped}")
    console.print(f"  Total points: {stats.total_points_fitted}")
    console.print(f"  Average RMSE: {stats.avg_rmse*100:.3f}%")

    if stats.arbitrage_violations > 0:
        console.print(
            f"  [yellow]Arbitrage violations: {stats.arbitrage_violations}[/yellow]"
        )

    if not no_store:
        console.print(f"\n[green]Results stored in database[/green]")

    # Verbose output - show SVI parameters
    if verbose:
        console.print(f"\n[bold]SVI Parameters:[/bold]")
        params_table = Table()
        params_table.add_column("Expiration", style="cyan")
        params_table.add_column("a", justify="right")
        params_table.add_column("b", justify="right")
        params_table.add_column("ρ", justify="right")
        params_table.add_column("m", justify="right")
        params_table.add_column("σ", justify="right")

        for s in surfaces:
            p = s.svi_params
            params_table.add_row(
                str(s.expiration_date),
                f"{p.a:.6f}",
                f"{p.b:.6f}",
                f"{p.rho:.4f}",
                f"{p.m:.6f}",
                f"{p.sigma:.6f}",
            )

        console.print(params_table)


@app.command()
def term_structure(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to fit (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Display term structure from fitted surfaces."""
    from volsurf.database.connection import get_connection

    # Get the quote date
    conn = get_connection()
    if target_date:
        quote_date = parse_date(target_date)
    else:
        result = conn.execute(
            "SELECT MAX(quote_date) FROM fitted_surfaces WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No fitted surfaces found for {symbol}[/red]")
            console.print("Run 'volsurf fit surfaces' first to fit surfaces")
            raise typer.Exit(1)
        quote_date = result[0]

    # Query fitted surfaces
    query = """
    SELECT
        expiration_date,
        tte_years,
        atm_vol,
        skew_25delta,
        svi_a, svi_b, svi_rho, svi_m, svi_sigma,
        rmse
    FROM fitted_surfaces
    WHERE symbol = ? AND quote_date = ?
    ORDER BY tte_years
    """
    df = conn.execute(query, [symbol, quote_date]).fetchdf()

    if df.empty:
        console.print(f"[red]No fitted surfaces found for {symbol} on {quote_date}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Term Structure for {symbol} on {quote_date}[/bold]\n")

    # ATM Term Structure
    table = Table(title="ATM Volatility Term Structure")
    table.add_column("Expiration", style="cyan")
    table.add_column("TTE (days)", justify="right")
    table.add_column("ATM Vol", justify="right", style="green")
    table.add_column("Skew (25Δ)", justify="right")

    for _, row in df.iterrows():
        tte_days = int(row["tte_years"] * 365)
        skew_str = f"{row['skew_25delta']*100:.2f}%" if row["skew_25delta"] else "N/A"
        table.add_row(
            str(row["expiration_date"]),
            str(tte_days),
            f"{row['atm_vol']*100:.2f}%",
            skew_str,
        )

    console.print(table)

    # Basic term structure stats
    console.print(f"\n[bold]Term Structure Statistics:[/bold]")
    console.print(f"  Number of expirations: {len(df)}")
    console.print(f"  Min TTE: {int(df['tte_years'].min() * 365)} days")
    console.print(f"  Max TTE: {int(df['tte_years'].max() * 365)} days")
    console.print(f"  ATM Vol range: {df['atm_vol'].min()*100:.2f}% - {df['atm_vol'].max()*100:.2f}%")


@app.command()
def batch(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    no_store: bool = typer.Option(
        False, "--no-store", help="Don't store results in database"
    ),
) -> None:
    """Batch fit surfaces for a date range."""
    from volsurf.fitting.pipeline import batch_fit_surfaces

    start_date = parse_date(start)
    end_date = parse_date(end)

    console.print(
        f"Batch fitting [cyan]{symbol}[/cyan] from [cyan]{start_date}[/cyan] to [cyan]{end_date}[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Batch fitting surfaces...", total=None)

        results = batch_fit_surfaces(
            symbol, start_date, end_date, store_results=not no_store
        )

        progress.update(task, completed=True)

    # Display results
    console.print(f"\n[bold]Batch Fitting Complete[/bold]")
    console.print(f"  Dates processed: {results['dates_processed']}")
    console.print(f"  Total surfaces fitted: {results['total_surfaces']}")
    console.print(f"  Total expirations skipped: {results['total_skipped']}")

    if results["arbitrage_violations"] > 0:
        console.print(
            f"  [yellow]Arbitrage violations: {results['arbitrage_violations']}[/yellow]"
        )

    if not no_store:
        console.print(f"\n[green]Results stored in database[/green]")


@app.command()
def show(
    symbol: str = typer.Argument("SPY", help="Symbol to display"),
    expiration: str = typer.Option(
        ..., "--expiration", "-e", help="Expiration date (YYYY-MM-DD)"
    ),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Quote date (YYYY-MM-DD), defaults to latest"
    ),
) -> None:
    """Show detailed fitted surface parameters."""
    from volsurf.database.connection import get_connection

    conn = get_connection()
    exp_date = parse_date(expiration)

    if target_date:
        quote_date = parse_date(target_date)
    else:
        result = conn.execute(
            """SELECT MAX(quote_date) FROM fitted_surfaces
               WHERE symbol = ? AND expiration_date = ?""",
            [symbol, exp_date],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No fitted surface found for {symbol} expiring {exp_date}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    # Query the surface
    query = """
    SELECT *
    FROM fitted_surfaces
    WHERE symbol = ? AND quote_date = ? AND expiration_date = ?
    """
    df = conn.execute(query, [symbol, quote_date, exp_date]).fetchdf()

    if df.empty:
        console.print(f"[red]No fitted surface found[/red]")
        raise typer.Exit(1)

    row = df.iloc[0]
    tte_days = int(row["tte_years"] * 365)

    console.print(f"\n[bold]Fitted Surface: {symbol} expiring {exp_date}[/bold]")
    console.print(f"  Quote date: {quote_date}")
    console.print(f"  Time to expiration: {tte_days} days ({row['tte_years']:.4f} years)")
    console.print(f"  Forward price: ${row['forward_price']:.2f}")

    console.print(f"\n[bold]SVI Parameters:[/bold]")
    console.print(f"  a (ATM variance): {row['svi_a']:.6f}")
    console.print(f"  b (wing slope):   {row['svi_b']:.6f}")
    console.print(f"  ρ (skew):         {row['svi_rho']:.4f}")
    console.print(f"  m (shift):        {row['svi_m']:.6f}")
    console.print(f"  σ (smoothness):   {row['svi_sigma']:.6f}")

    console.print(f"\n[bold]Derived Quantities:[/bold]")
    console.print(f"  ATM Vol: {row['atm_vol']*100:.2f}%")
    if row["skew_25delta"]:
        console.print(f"  25Δ Skew: {row['skew_25delta']*100:.2f}%")

    console.print(f"\n[bold]Fit Quality:[/bold]")
    console.print(f"  RMSE: {row['rmse']*100:.4f}%")
    console.print(f"  MAE: {row['mae']*100:.4f}%")
    console.print(f"  Max Error: {row['max_error']*100:.4f}%")
    console.print(f"  Points fitted: {row['num_points']}")

    arb_status = "[green]Pass[/green]" if row["passes_no_arbitrage"] else "[red]Fail[/red]"
    console.print(f"\n[bold]Arbitrage Check:[/bold] {arb_status}")
    if not row["passes_no_arbitrage"]:
        console.print(f"  Butterfly violations: {row['butterfly_arbitrage_violations']}")
        console.print(f"  Calendar violations: {row['calendar_arbitrage_violations']}")


@app.command()
def compare(
    symbol: str = typer.Argument("SPY", help="Symbol to compare"),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date to compare (YYYY-MM-DD), defaults to latest"
    ),
    expiration: Optional[str] = typer.Option(
        None, "--expiration", "-e", help="Specific expiration to compare (YYYY-MM-DD)"
    ),
) -> None:
    """Compare SVI, SABR, and polynomial models for a date."""
    from volsurf.database.connection import get_connection
    from volsurf.fitting.comparison import ModelComparator, print_comparison_summary

    conn = get_connection()

    # Get the quote date
    if target_date:
        quote_date = parse_date(target_date)
    else:
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    console.print(
        f"Comparing models for [cyan]{symbol}[/cyan] on [cyan]{quote_date}[/cyan]\n"
    )

    comparator = ModelComparator()

    if expiration:
        # Compare single expiration
        exp_date = parse_date(expiration)
        result = comparator.compare_slice(symbol, quote_date, exp_date)

        if result is None:
            console.print(f"[red]No data found for expiration {exp_date}[/red]")
            raise typer.Exit(1)

        tte_days = int(result.tte_years * 365)
        console.print(f"[bold]Expiration: {exp_date} (TTE: {tte_days} days)[/bold]")
        console.print(f"Points used: {result.num_points}\n")

        table = Table(title="Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("RMSE", justify="right")
        table.add_column("Status", justify="center")

        for model, rmse in result.rmse_by_model.items():
            status = "[green]OK[/green]" if rmse is not None else "[red]Failed[/red]"
            rmse_str = f"{rmse*100:.4f}%" if rmse else "N/A"
            table.add_row(model.value, rmse_str, status)

        console.print(table)

        if result.best_model:
            console.print(f"\n[green]Best model: {result.best_model.value}[/green]")

    else:
        # Compare all expirations for the date
        full_result = comparator.compare_date(symbol, quote_date)

        if not full_result.slice_results:
            console.print(f"[red]No data found for {symbol} on {quote_date}[/red]")
            raise typer.Exit(1)

        print_comparison_summary(full_result)

        # Show best model summary
        console.print(f"\n[bold]Best Model Selection:[/bold]")
        console.print(f"  SVI best: {full_result.svi_best_count} expirations")
        console.print(f"  SABR best: {full_result.sabr_best_count} expirations")
        console.print(f"  Polynomial best: {full_result.poly_best_count} expirations")


@app.command()
def sabr(
    symbol: str = typer.Argument("SPY", help="Symbol to fit"),
    expiration: str = typer.Option(
        ..., "--expiration", "-e", help="Expiration date (YYYY-MM-DD)"
    ),
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Quote date (YYYY-MM-DD), defaults to latest"
    ),
    beta: float = typer.Option(0.7, "--beta", "-b", help="SABR beta parameter (0-1)"),
) -> None:
    """Fit SABR model to a specific expiration."""
    import numpy as np
    from volsurf.database.connection import get_connection
    from volsurf.fitting.sabr import fit_sabr_slice, sabr_atm_vol, sabr_skew

    conn = get_connection()
    exp_date = parse_date(expiration)

    if target_date:
        quote_date = parse_date(target_date)
    else:
        result = conn.execute(
            "SELECT MAX(quote_date) FROM raw_options_chains WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result[0] is None:
            console.print(f"[red]No data found for {symbol}[/red]")
            raise typer.Exit(1)
        quote_date = result[0]

    console.print(
        f"Fitting SABR for [cyan]{symbol}[/cyan] expiring [cyan]{exp_date}[/cyan]\n"
    )

    # Get options data
    query = """
        SELECT strike, implied_volatility, underlying_price
        FROM raw_options_chains
        WHERE symbol = ? AND quote_date = ? AND expiration_date = ?
          AND is_liquid = TRUE AND implied_volatility IS NOT NULL
        ORDER BY strike
    """
    df = conn.execute(query, [symbol, quote_date, exp_date]).fetchdf()

    if df.empty or len(df) < 5:
        console.print(f"[red]Insufficient data for {exp_date}[/red]")
        raise typer.Exit(1)

    # Calculate TTE
    tte_years = (exp_date - quote_date).days / 365.0
    forward = float(df["underlying_price"].iloc[0])  # Simplified

    strikes = df["strike"].values
    ivs = df["implied_volatility"].values

    result = fit_sabr_slice(strikes, ivs, forward, tte_years, beta=beta)

    if result is None:
        console.print("[red]SABR fitting failed[/red]")
        raise typer.Exit(1)

    # Display results
    console.print(f"[bold]SABR Parameters:[/bold]")
    console.print(f"  α (alpha): {result.alpha:.6f}")
    console.print(f"  β (beta):  {result.beta:.4f}")
    console.print(f"  ρ (rho):   {result.rho:.4f}")
    console.print(f"  ν (nu):    {result.nu:.6f}")

    atm_vol = sabr_atm_vol(forward, tte_years, result.alpha, result.beta, result.rho, result.nu)
    skew = sabr_skew(forward, tte_years, result.alpha, result.beta, result.rho, result.nu)

    console.print(f"\n[bold]Derived Quantities:[/bold]")
    console.print(f"  ATM Vol: {atm_vol*100:.2f}%")
    console.print(f"  Skew: {skew*100:.2f}%")

    console.print(f"\n[bold]Fit Quality:[/bold]")
    console.print(f"  RMSE: {result.rmse*100:.4f}%")
    console.print(f"  Points: {result.num_points}")
