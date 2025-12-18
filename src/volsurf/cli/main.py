"""Main CLI application for volsurf."""

import typer
from rich.console import Console

from volsurf import __version__
from volsurf.cli.commands import ingest, fit, analyze, terminal

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
app.add_typer(terminal.app, name="terminal", help="Theta Terminal management")


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
def web(
    port: int = typer.Option(8501, help="Port for web dashboard"),
    browser: bool = typer.Option(True, help="Open browser automatically"),
) -> None:
    """Launch the Streamlit web dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    # Get the path to the web app
    web_app_path = Path(__file__).parent.parent / "web" / "app.py"

    if not web_app_path.exists():
        console.print(f"[red]Web app not found at {web_app_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Starting web dashboard on port {port}...[/cyan]")
    console.print(f"[dim]URL: http://localhost:{port}[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if not browser else "false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start dashboard: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
