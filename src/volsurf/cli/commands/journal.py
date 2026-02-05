"""Trade journal CLI commands."""

from datetime import date, datetime
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from volsurf.analytics.trade_journal import (
    VALID_CATEGORIES,
    VALID_DIRECTIONS,
    TradeEntry,
    TradeJournalManager,
    TradeResolution,
)

app = typer.Typer(help="Trade journal commands")
console = Console()


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if date_str is None:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)


@app.command()
def new(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
    thesis: str = typer.Option(..., "--thesis", "-t", help="Trade thesis"),
    category: str = typer.Option(
        ...,
        "--category",
        "-c",
        help=f"Category: {', '.join(sorted(VALID_CATEGORIES))}",
    ),
    direction: str = typer.Option(
        ...,
        "--direction",
        "-d",
        help=f"Direction: {', '.join(sorted(VALID_DIRECTIONS))}",
    ),
    structure: Optional[str] = typer.Option(None, "--structure", "-s", help="Structure description"),
    target: Optional[str] = typer.Option(None, "--target", help="Target description"),
    stop: Optional[str] = typer.Option(None, "--stop", help="Stop description"),
    horizon: Optional[int] = typer.Option(None, "--horizon", "-h", help="Time horizon in days"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    entry_date: Optional[str] = typer.Option(None, "--date", help="Entry date (YYYY-MM-DD, default today)"),
) -> None:
    """Log a new hypothetical trade."""
    from volsurf.database.schema import init_schema

    init_schema()

    entry = TradeEntry(
        symbol=symbol.upper(),
        thesis=thesis,
        category=category,
        direction=direction,
        entry_date=_parse_date(entry_date),
        structure_description=structure,
        target_description=target,
        stop_description=stop,
        time_horizon_days=horizon,
        tags=tags,
    )

    manager = TradeJournalManager()

    try:
        trade_id = manager.create_trade(entry)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Show what was created
    trade = manager.get_trade(trade_id)
    console.print(f"\n[green]Trade #{trade_id} created[/green]")

    table = Table(title=f"Trade #{trade_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Symbol", trade["symbol"])
    table.add_row("Date", str(trade["entry_date"]))
    table.add_row("Category", trade["category"])
    table.add_row("Direction", trade["direction"])
    table.add_row("Thesis", trade["thesis"])
    if trade.get("structure_description"):
        table.add_row("Structure", trade["structure_description"])
    if trade.get("entry_iv_level") is not None:
        table.add_row("Entry IV", f"{trade['entry_iv_level']:.2%}")
    if trade.get("entry_skew_level") is not None:
        table.add_row("Entry Skew", f"{trade['entry_skew_level']:.4f}")
    if trade.get("entry_underlying_price") is not None:
        table.add_row("Entry Price", f"${trade['entry_underlying_price']:.2f}")
    if trade.get("entry_atm_vol_pctile") is not None:
        table.add_row("ATM Vol Pctile", f"{trade['entry_atm_vol_pctile']:.0f}th")
    if trade.get("entry_vrp") is not None:
        table.add_row("VRP", f"{trade['entry_vrp']:.2%}")
    if trade.get("time_horizon_days"):
        table.add_row("Horizon", f"{trade['time_horizon_days']}d")
    if trade.get("tags"):
        table.add_row("Tags", trade["tags"])

    console.print(table)


@app.command()
def resolve(
    trade_id: int = typer.Argument(..., help="Trade ID to resolve"),
    outcome: str = typer.Option(..., "--outcome", "-o", help="Outcome: win, loss, scratch"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Outcome notes"),
    pnl: Optional[float] = typer.Option(None, "--pnl", help="P&L in vol points (auto-computed if omitted)"),
    exit_date: Optional[str] = typer.Option(None, "--date", help="Exit date (YYYY-MM-DD, default today)"),
) -> None:
    """Resolve an open trade."""
    from volsurf.database.schema import init_schema

    init_schema()

    resolution = TradeResolution(
        trade_id=trade_id,
        outcome=outcome,
        exit_date=_parse_date(exit_date),
        pnl_vol_points=pnl,
        outcome_notes=notes,
    )

    manager = TradeJournalManager()

    try:
        manager.resolve_trade(resolution)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    trade = manager.get_trade(trade_id)
    console.print(f"[green]Trade #{trade_id} resolved: {outcome}[/green]")
    if trade.get("pnl_vol_points") is not None:
        console.print(f"  P&L: {trade['pnl_vol_points']:.4f} vol points")


@app.command(name="auto-resolve")
def auto_resolve() -> None:
    """Auto-resolve trades past their time horizon."""
    from volsurf.database.schema import init_schema

    init_schema()

    manager = TradeJournalManager()
    count = manager.auto_resolve_expired()

    if count == 0:
        console.print("[dim]No expired trades to resolve.[/dim]")
    else:
        console.print(f"[green]Auto-resolved {count} trade(s)[/green]")


@app.command(name="open")
def open_trades() -> None:
    """Show all open trades."""
    from volsurf.database.schema import init_schema

    init_schema()

    manager = TradeJournalManager()
    df = manager.get_open_trades()

    if df.empty:
        console.print("[dim]No open trades.[/dim]")
        return

    table = Table(title="Open Trades")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Date", justify="center")
    table.add_column("Symbol", style="bold")
    table.add_column("Direction")
    table.add_column("Category")
    table.add_column("Thesis")
    table.add_column("Entry IV", justify="right")
    table.add_column("Horizon", justify="right")

    for _, row in df.iterrows():
        iv = f"{row['entry_iv_level']:.1%}" if pd.notna(row.get("entry_iv_level")) else "—"
        horizon = f"{int(row['time_horizon_days'])}d" if pd.notna(row.get("time_horizon_days")) else "—"
        thesis = str(row["thesis"])
        if len(thesis) > 40:
            thesis = thesis[:37] + "..."

        table.add_row(
            str(row["trade_id"]),
            str(row["entry_date"]),
            row["symbol"],
            row["direction"],
            row["category"],
            thesis,
            iv,
            horizon,
        )

    console.print(table)


@app.command()
def history(
    last: int = typer.Option(20, "--last", "-l", help="Number of recent trades to show"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
) -> None:
    """Show trade history."""
    from volsurf.database.schema import init_schema

    init_schema()

    manager = TradeJournalManager()
    df = manager.get_trade_history(symbol=symbol, category=category, limit=last)

    if df.empty:
        console.print("[dim]No trades found.[/dim]")
        return

    table = Table(title="Trade History")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Entry", justify="center")
    table.add_column("Exit", justify="center")
    table.add_column("Symbol", style="bold")
    table.add_column("Direction")
    table.add_column("Category")
    table.add_column("Outcome")
    table.add_column("P&L", justify="right")

    for _, row in df.iterrows():
        exit_str = str(row["exit_date"]) if pd.notna(row.get("exit_date")) else "—"
        pnl_str = f"{row['pnl_vol_points']:.4f}" if pd.notna(row.get("pnl_vol_points")) else "—"

        outcome = row["outcome"]
        if outcome == "win":
            outcome_str = "[green]WIN[/green]"
        elif outcome == "loss":
            outcome_str = "[red]LOSS[/red]"
        elif outcome == "scratch":
            outcome_str = "[yellow]SCRATCH[/yellow]"
        else:
            outcome_str = "[dim]OPEN[/dim]"

        table.add_row(
            str(row["trade_id"]),
            str(row["entry_date"]),
            exit_str,
            row["symbol"],
            row["direction"],
            row["category"],
            outcome_str,
            pnl_str,
        )

    console.print(table)


@app.command()
def stats(
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
) -> None:
    """Show performance summary."""
    from volsurf.database.schema import init_schema

    init_schema()

    manager = TradeJournalManager()
    summary = manager.get_performance_summary(category=category, symbol=symbol)

    if summary["total_trades"] == 0:
        console.print("[dim]No trades to summarize.[/dim]")
        return

    # Overall summary
    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(summary["total_trades"]))
    table.add_row("[green]Wins[/green]", str(summary["wins"]))
    table.add_row("[red]Losses[/red]", str(summary["losses"]))
    table.add_row("[yellow]Scratches[/yellow]", str(summary["scratches"]))
    table.add_row("[dim]Open[/dim]", str(summary["open"]))
    if summary["win_rate"] is not None:
        table.add_row("Win Rate", f"{summary['win_rate']:.1%}")
    if summary["avg_pnl"] is not None:
        table.add_row("Avg P&L (vol pts)", f"{summary['avg_pnl']:.4f}")
    if summary["avg_win"] is not None:
        table.add_row("Avg Win", f"{summary['avg_win']:.4f}")
    if summary["avg_loss"] is not None:
        table.add_row("Avg Loss", f"{summary['avg_loss']:.4f}")

    console.print(table)

    # Category breakdown
    cat_df = manager.get_category_performance()
    if not cat_df.empty:
        console.print()
        cat_table = Table(title="By Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Total", justify="right")
        cat_table.add_column("W/L/S", justify="center")
        cat_table.add_column("Win %", justify="right")
        cat_table.add_column("Avg P&L", justify="right")

        for _, row in cat_df.iterrows():
            wls = f"{int(row['wins'])}/{int(row['losses'])}/{int(row['scratches'])}"
            wr = f"{row['win_rate_pct']:.0f}%" if pd.notna(row.get("win_rate_pct")) else "—"
            avg = f"{row['avg_pnl']:.4f}" if pd.notna(row.get("avg_pnl")) else "—"
            cat_table.add_row(row["category"], str(int(row["total"])), wls, wr, avg)

        console.print(cat_table)
