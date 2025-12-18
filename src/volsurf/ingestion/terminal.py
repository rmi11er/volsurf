"""Theta Terminal process manager."""

import atexit
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from volsurf.config.settings import Settings, get_settings


class ThetaTerminalManager:
    """
    Manages the Theta Terminal Java process.

    Starts the terminal as a subprocess and ensures it's ready before
    returning. Can be used as a context manager for automatic cleanup.

    Example:
        with ThetaTerminalManager() as terminal:
            # Terminal is running and ready
            client = ThetaTerminalClient()
            data = client.get_options_chain("SPY", date.today())
        # Terminal is stopped automatically

    Or for persistent use:
        manager = ThetaTerminalManager()
        manager.start()
        # ... use the terminal ...
        manager.stop()
    """

    _instance: Optional["ThetaTerminalManager"] = None
    _process: Optional[subprocess.Popen] = None

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._started_by_us = False

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "ThetaTerminalManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    @property
    def jar_path(self) -> Path:
        """Get absolute path to the terminal JAR."""
        jar = self.settings.theta_terminal_jar
        if not jar.is_absolute():
            # Resolve relative to project root (vol-modeling/)
            # __file__ is src/volsurf/ingestion/terminal.py
            # project root is 4 levels up: ingestion -> volsurf -> src -> vol-modeling
            project_root = Path(__file__).parent.parent.parent.parent
            jar = project_root / jar
        return jar.resolve()

    @property
    def terminal_dir(self) -> Path:
        """Get the directory containing the terminal (for creds.txt, config.toml)."""
        return self.jar_path.parent

    @property
    def base_url(self) -> str:
        """Get the terminal API base URL."""
        return self.settings.theta_terminal_url

    def is_running(self) -> bool:
        """Check if the terminal is running and responding."""
        try:
            response = httpx.get(
                f"{self.base_url}/option/list/expirations",
                params={"symbol": "SPY", "format": "json"},
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            return False
        except Exception as e:
            logger.debug(f"Error checking terminal status: {e}")
            return False

    def start(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """
        Start the Theta Terminal if not already running.

        Args:
            wait: Wait for terminal to be ready before returning
            timeout: Maximum time to wait for terminal to be ready

        Returns:
            True if terminal is running (started or was already running)
        """
        # Check if already running
        if self.is_running():
            logger.debug("Theta Terminal already running")
            return True

        # Validate JAR exists
        if not self.jar_path.exists():
            logger.error(f"Theta Terminal JAR not found: {self.jar_path}")
            logger.info("Please ensure ThetaTerminal.jar is in the vendor/ directory")
            return False

        # Check for credentials
        creds_path = self.terminal_dir / "creds.txt"
        if not creds_path.exists():
            logger.error(f"Credentials file not found: {creds_path}")
            logger.info("Create vendor/creds.txt with your email on line 1 and password on line 2")
            return False

        logger.info(f"Starting Theta Terminal from {self.jar_path}")

        try:
            # Start the terminal process
            # Run from the terminal directory so it finds config.toml
            # v3 terminal requires --config and --creds-file arguments
            ThetaTerminalManager._process = subprocess.Popen(
                [
                    "java", "-jar", str(self.jar_path),
                    "--config", "config.toml",
                    "--creds-file", "creds.txt",
                ],
                cwd=str(self.terminal_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Don't inherit our stdin - let it run headless
                stdin=subprocess.DEVNULL,
            )
            self._started_by_us = True

            # Register cleanup handler
            atexit.register(self._cleanup)

            if wait:
                return self._wait_for_ready(timeout)

            return True

        except FileNotFoundError:
            logger.error("Java not found. Please ensure Java 21+ is installed and in PATH")
            return False
        except Exception as e:
            logger.error(f"Failed to start Theta Terminal: {e}")
            return False

    def _wait_for_ready(self, timeout: float = 30.0) -> bool:
        """Wait for terminal to be ready to accept connections."""
        logger.info("Waiting for Theta Terminal to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info("Theta Terminal is ready!")
                return True

            # Check if process died
            if ThetaTerminalManager._process and ThetaTerminalManager._process.poll() is not None:
                logger.error("Theta Terminal process exited unexpectedly")
                # Try to get error output
                if ThetaTerminalManager._process.stderr:
                    stderr = ThetaTerminalManager._process.stderr.read()
                    if stderr:
                        logger.error(f"Terminal stderr: {stderr.decode()}")
                return False

            time.sleep(0.5)

        logger.error(f"Theta Terminal did not become ready within {timeout}s")
        return False

    def stop(self) -> None:
        """Stop the Theta Terminal if we started it."""
        if not self._started_by_us:
            logger.debug("Terminal was not started by us, not stopping")
            return

        if ThetaTerminalManager._process is None:
            return

        logger.info("Stopping Theta Terminal...")

        try:
            ThetaTerminalManager._process.terminate()
            try:
                ThetaTerminalManager._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Terminal did not stop gracefully, killing...")
                ThetaTerminalManager._process.kill()
                ThetaTerminalManager._process.wait()
        except Exception as e:
            logger.error(f"Error stopping terminal: {e}")
        finally:
            ThetaTerminalManager._process = None
            self._started_by_us = False

    def _cleanup(self) -> None:
        """Cleanup handler for atexit."""
        self.stop()

    def ensure_running(self) -> bool:
        """Ensure terminal is running, starting if necessary."""
        if self.is_running():
            return True
        return self.start()

    def __enter__(self) -> "ThetaTerminalManager":
        """Context manager entry - start terminal."""
        if not self.start():
            raise RuntimeError("Failed to start Theta Terminal")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop terminal."""
        self.stop()


def ensure_terminal_running(settings: Optional[Settings] = None) -> bool:
    """
    Convenience function to ensure the terminal is running.

    Uses a singleton manager instance for the process lifetime.
    """
    manager = ThetaTerminalManager.get_instance(settings)
    return manager.ensure_running()


def stop_terminal() -> None:
    """Stop the terminal if it was started by us."""
    if ThetaTerminalManager._instance:
        ThetaTerminalManager._instance.stop()
