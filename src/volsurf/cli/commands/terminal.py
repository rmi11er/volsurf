"""Theta Terminal management CLI commands."""

import typer
from rich.console import Console

app = typer.Typer(help="Theta Terminal management commands")
console = Console()


@app.command()
def start() -> None:
    """Start the Theta Terminal."""
    from volsurf.ingestion.terminal import ThetaTerminalManager
    from volsurf.config.settings import get_settings

    settings = get_settings()
    manager = ThetaTerminalManager.get_instance(settings)

    if manager.is_running():
        console.print("[green]Theta Terminal is already running.[/green]")
        return

    console.print(f"Starting Theta Terminal from [cyan]{manager.jar_path}[/cyan]...")

    if manager.start():
        console.print("[green]Theta Terminal started successfully![/green]")
        console.print(f"  API URL: {settings.theta_terminal_url}")
        console.print("\n[dim]Terminal will stop when this process exits.[/dim]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]")

        # Keep running until interrupted
        try:
            import time
            while manager.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping terminal...[/yellow]")
            manager.stop()
            console.print("[green]Terminal stopped.[/green]")
    else:
        console.print("[red]Failed to start Theta Terminal.[/red]")
        raise typer.Exit(1)


@app.command()
def stop() -> None:
    """Stop the Theta Terminal (if started by volsurf)."""
    from volsurf.ingestion.terminal import stop_terminal, ThetaTerminalManager

    manager = ThetaTerminalManager.get_instance()

    if not manager.is_running():
        console.print("[yellow]Theta Terminal is not running.[/yellow]")
        return

    if not manager._started_by_us:
        console.print("[yellow]Terminal was not started by volsurf. Not stopping.[/yellow]")
        console.print("[dim]Use the terminal window to stop it manually.[/dim]")
        return

    stop_terminal()
    console.print("[green]Theta Terminal stopped.[/green]")


@app.command()
def status() -> None:
    """Check Theta Terminal status."""
    from volsurf.ingestion.terminal import ThetaTerminalManager
    from volsurf.config.settings import get_settings

    settings = get_settings()
    manager = ThetaTerminalManager.get_instance(settings)

    console.print("\n[bold]Theta Terminal Status[/bold]")
    console.print(f"  JAR path: {manager.jar_path}")
    console.print(f"  JAR exists: {manager.jar_path.exists()}")
    console.print(f"  Creds file: {manager.terminal_dir / 'creds.txt'}")
    console.print(f"  Creds exists: {(manager.terminal_dir / 'creds.txt').exists()}")
    console.print(f"  API URL: {settings.theta_terminal_url}")

    if manager.is_running():
        console.print("\n  [green]Status: RUNNING[/green]")

        # Try to get some info
        try:
            expirations = []
            from volsurf.ingestion.theta_client import ThetaTerminalClient
            client = ThetaTerminalClient(settings)
            expirations = client.get_expirations("SPY")
            console.print(f"  SPY expirations available: {len(expirations)}")
        except Exception as e:
            console.print(f"  [yellow]Could not query API: {e}[/yellow]")
    else:
        console.print("\n  [red]Status: NOT RUNNING[/red]")
        console.print("\n  To start: [cyan]volsurf terminal start[/cyan]")
        console.print("  Or it will auto-start when running ingest commands.")
