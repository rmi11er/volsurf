"""Theta Terminal process manager."""

import atexit
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import httpx
from loguru import logger

from volsurf.config.settings import Settings, get_settings


class TerminalStatus(Enum):
    """Terminal connection status."""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    RUNNING = "running"
    PROCESS_DIED = "process_died"
    CONNECTION_REFUSED = "connection_refused"
    CONNECTION_TIMEOUT = "connection_timeout"
    AUTH_FAILED = "auth_failed"
    SESSION_EXPIRED = "session_expired"
    HTTP_ERROR = "http_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class TerminalDiagnostics:
    """Detailed diagnostics for terminal connection issues."""
    status: TerminalStatus
    message: str
    jar_exists: bool = False
    jar_path: Optional[str] = None
    creds_exists: bool = False
    creds_path: Optional[str] = None
    config_exists: bool = False
    config_path: Optional[str] = None
    process_running: bool = False
    process_exit_code: Optional[int] = None
    process_stdout: list[str] = field(default_factory=list)
    process_stderr: list[str] = field(default_factory=list)
    connection_error: Optional[str] = None
    http_status_code: Optional[int] = None
    http_response_body: Optional[str] = None
    java_available: bool = False
    java_version: Optional[str] = None

    def format_report(self) -> str:
        """Format diagnostics as a human-readable report."""
        lines = [
            f"Terminal Status: {self.status.value}",
            f"Message: {self.message}",
            "",
            "Environment:",
            f"  JAR exists: {self.jar_exists} ({self.jar_path})",
            f"  Credentials exist: {self.creds_exists} ({self.creds_path})",
            f"  Config exists: {self.config_exists} ({self.config_path})",
            f"  Java available: {self.java_available}",
        ]
        if self.java_version:
            lines.append(f"  Java version: {self.java_version}")

        lines.append("")
        lines.append("Process:")
        lines.append(f"  Running: {self.process_running}")
        if self.process_exit_code is not None:
            lines.append(f"  Exit code: {self.process_exit_code}")

        if self.process_stdout:
            lines.append("  Stdout (last 10 lines):")
            for line in self.process_stdout[-10:]:
                lines.append(f"    {line}")

        if self.process_stderr:
            lines.append("  Stderr (last 10 lines):")
            for line in self.process_stderr[-10:]:
                lines.append(f"    {line}")

        if self.connection_error:
            lines.append("")
            lines.append(f"Connection Error: {self.connection_error}")

        if self.http_status_code is not None:
            lines.append(f"HTTP Status: {self.http_status_code}")
        if self.http_response_body:
            lines.append(f"HTTP Response: {self.http_response_body[:500]}")

        return "\n".join(lines)


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
    _stdout_lines: list[str] = []
    _stderr_lines: list[str] = []
    _output_threads: list[threading.Thread] = []
    _last_connection_error: Optional[str] = None
    _last_http_status: Optional[int] = None
    _last_http_body: Optional[str] = None

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
        status, error = self._check_connection()
        return status == TerminalStatus.RUNNING

    def _check_connection(self) -> tuple[TerminalStatus, Optional[str]]:
        """
        Check terminal connection with detailed error info.

        Returns:
            Tuple of (status, error_message)
        """
        # Reset last error state
        ThetaTerminalManager._last_connection_error = None
        ThetaTerminalManager._last_http_status = None
        ThetaTerminalManager._last_http_body = None

        try:
            response = httpx.get(
                f"{self.base_url}/option/list/expirations",
                params={"symbol": "SPY", "format": "json"},
                timeout=5.0,
            )
            ThetaTerminalManager._last_http_status = response.status_code
            ThetaTerminalManager._last_http_body = response.text[:1000] if response.text else None

            if response.status_code == 200:
                return TerminalStatus.RUNNING, None
            elif response.status_code == 401:
                error = f"Authentication failed (HTTP 401): {response.text[:200]}"
                ThetaTerminalManager._last_connection_error = error
                return TerminalStatus.AUTH_FAILED, error
            elif response.status_code == 473:
                # Theta-specific: Invalid or expired session
                error = f"Session expired or invalid (HTTP 473): {response.text[:200]}"
                ThetaTerminalManager._last_connection_error = error
                return TerminalStatus.SESSION_EXPIRED, error
            else:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                ThetaTerminalManager._last_connection_error = error
                return TerminalStatus.HTTP_ERROR, error

        except httpx.ConnectError as e:
            error = f"Connection refused: {e}"
            ThetaTerminalManager._last_connection_error = error
            return TerminalStatus.CONNECTION_REFUSED, error
        except httpx.ReadTimeout as e:
            error = f"Connection timeout: {e}"
            ThetaTerminalManager._last_connection_error = error
            return TerminalStatus.CONNECTION_TIMEOUT, error
        except Exception as e:
            error = f"Unknown error: {type(e).__name__}: {e}"
            ThetaTerminalManager._last_connection_error = error
            logger.debug(f"Error checking terminal status: {e}")
            return TerminalStatus.UNKNOWN_ERROR, error

    @staticmethod
    def _stream_reader(stream, line_list: list[str], prefix: str) -> None:
        """Read lines from a stream and append to list."""
        try:
            for line in iter(stream.readline, b''):
                if line:
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    line_list.append(decoded)
                    logger.debug(f"{prefix}: {decoded}")
        except Exception as e:
            logger.debug(f"Stream reader error: {e}")
        finally:
            stream.close()

    def _start_output_capture(self) -> None:
        """Start background threads to capture process output."""
        if ThetaTerminalManager._process is None:
            return

        # Clear previous output
        ThetaTerminalManager._stdout_lines = []
        ThetaTerminalManager._stderr_lines = []
        ThetaTerminalManager._output_threads = []

        # Start stdout reader
        if ThetaTerminalManager._process.stdout:
            stdout_thread = threading.Thread(
                target=self._stream_reader,
                args=(ThetaTerminalManager._process.stdout,
                      ThetaTerminalManager._stdout_lines, "Terminal stdout"),
                daemon=True
            )
            stdout_thread.start()
            ThetaTerminalManager._output_threads.append(stdout_thread)

        # Start stderr reader
        if ThetaTerminalManager._process.stderr:
            stderr_thread = threading.Thread(
                target=self._stream_reader,
                args=(ThetaTerminalManager._process.stderr,
                      ThetaTerminalManager._stderr_lines, "Terminal stderr"),
                daemon=True
            )
            stderr_thread.start()
            ThetaTerminalManager._output_threads.append(stderr_thread)

    def _get_java_version(self) -> tuple[bool, Optional[str]]:
        """Check if Java is available and get version."""
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                timeout=5.0
            )
            # Java outputs version to stderr
            version_output = result.stderr.decode('utf-8', errors='replace')
            # Extract first line which typically contains version
            first_line = version_output.split('\n')[0] if version_output else None
            return True, first_line
        except FileNotFoundError:
            return False, None
        except Exception as e:
            return False, str(e)

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

            # Start background output capture
            self._start_output_capture()

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
        last_status = TerminalStatus.STARTING

        while time.time() - start_time < timeout:
            status, error = self._check_connection()
            if status == TerminalStatus.RUNNING:
                logger.info("Theta Terminal is ready!")
                return True

            last_status = status

            # Check if process died
            if ThetaTerminalManager._process and ThetaTerminalManager._process.poll() is not None:
                exit_code = ThetaTerminalManager._process.returncode
                logger.error(f"Theta Terminal process exited unexpectedly (exit code: {exit_code})")

                # Give threads a moment to capture final output
                time.sleep(0.2)

                # Log captured output
                if ThetaTerminalManager._stderr_lines:
                    logger.error("Terminal stderr output:")
                    for line in ThetaTerminalManager._stderr_lines[-20:]:
                        logger.error(f"  {line}")
                if ThetaTerminalManager._stdout_lines:
                    logger.info("Terminal stdout output:")
                    for line in ThetaTerminalManager._stdout_lines[-10:]:
                        logger.info(f"  {line}")

                return False

            time.sleep(0.5)

        # Timeout - log diagnostic info
        elapsed = time.time() - start_time
        logger.error(f"Theta Terminal did not become ready within {timeout}s")
        logger.error(f"Last connection status: {last_status.value}")
        if ThetaTerminalManager._last_connection_error:
            logger.error(f"Last connection error: {ThetaTerminalManager._last_connection_error}")

        # Log any captured output
        if ThetaTerminalManager._stderr_lines:
            logger.error("Terminal stderr (last 10 lines):")
            for line in ThetaTerminalManager._stderr_lines[-10:]:
                logger.error(f"  {line}")
        if ThetaTerminalManager._stdout_lines:
            logger.info("Terminal stdout (last 10 lines):")
            for line in ThetaTerminalManager._stdout_lines[-10:]:
                logger.info(f"  {line}")

        return False

    def get_diagnostics(self) -> TerminalDiagnostics:
        """
        Get comprehensive diagnostics about terminal status.

        Useful for troubleshooting connection issues.
        """
        # Check environment
        jar_path = self.jar_path
        creds_path = self.terminal_dir / "creds.txt"
        config_path = self.terminal_dir / "config.toml"
        java_available, java_version = self._get_java_version()

        # Check connection
        status, error = self._check_connection()

        # Check process state
        process_running = False
        exit_code = None
        if ThetaTerminalManager._process:
            exit_code = ThetaTerminalManager._process.poll()
            process_running = exit_code is None

        # Determine message
        if status == TerminalStatus.RUNNING:
            message = "Terminal is running and responding"
        elif not jar_path.exists():
            status = TerminalStatus.NOT_STARTED
            message = f"JAR file not found: {jar_path}"
        elif not creds_path.exists():
            status = TerminalStatus.NOT_STARTED
            message = f"Credentials file not found: {creds_path}"
        elif not java_available:
            status = TerminalStatus.NOT_STARTED
            message = "Java not found in PATH"
        elif exit_code is not None:
            status = TerminalStatus.PROCESS_DIED
            message = f"Process exited with code {exit_code}"
        elif error:
            message = error
        else:
            message = "Terminal not responding"

        return TerminalDiagnostics(
            status=status,
            message=message,
            jar_exists=jar_path.exists(),
            jar_path=str(jar_path),
            creds_exists=creds_path.exists(),
            creds_path=str(creds_path),
            config_exists=config_path.exists(),
            config_path=str(config_path),
            process_running=process_running,
            process_exit_code=exit_code,
            process_stdout=list(ThetaTerminalManager._stdout_lines),
            process_stderr=list(ThetaTerminalManager._stderr_lines),
            connection_error=ThetaTerminalManager._last_connection_error,
            http_status_code=ThetaTerminalManager._last_http_status,
            http_response_body=ThetaTerminalManager._last_http_body,
            java_available=java_available,
            java_version=java_version,
        )

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


def get_terminal_diagnostics(settings: Optional[Settings] = None) -> TerminalDiagnostics:
    """
    Get diagnostics about the terminal connection.

    Useful for troubleshooting when the terminal fails to connect.
    """
    manager = ThetaTerminalManager.get_instance(settings)
    return manager.get_diagnostics()
