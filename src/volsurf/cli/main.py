"""Main CLI application for volsurf."""

import typer
from rich.console import Console

from volsurf import __version__
from volsurf.cli.commands import ingest, fit, analyze

app = typer.Typer(
    name="volsurf",
    help="Volatility Surface Data Lake - Options analytics and surface fitting",
    no_args_is_help=True,
)
console = Console()

# Register command groups
app.add_typer(ingest.app, name="ingest", help="Data ingestion commands")
app.add_typer(fit.app, name="fit", help="Surface fitting commands")
app.add_typer(analyze.app, name="analyze", help="Analytics commands")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"volsurf version {__version__}")


@app.command()
def init_db() -> None:
    """Initialize the database schema."""
    from volsurf.database.schema import init_schema
    from volsurf.config.settings import get_settings

    settings = get_settings()
    console.print(f"Initializing database at [cyan]{settings.duckdb_path}[/cyan]...")
    init_schema()
    console.print("[green]Database initialized successfully![/green]")


@app.command()
def dashboard(symbol: str = typer.Argument("SPY", help="Symbol to display")) -> None:
    """Launch the terminal dashboard."""
    console.print(f"[yellow]Dashboard for {symbol} - coming in Phase 4[/yellow]")


@app.command()
def web(port: int = typer.Option(8501, help="Port for web dashboard")) -> None:
    """Launch the web dashboard."""
    console.print(f"[yellow]Web dashboard on port {port} - coming in Phase 4[/yellow]")


if __name__ == "__main__":
    app()
