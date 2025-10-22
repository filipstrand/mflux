"""
Centralized log path management for the debugger.

Defines the directory structure for all logs:
- Debugger server logs: logs/debugger/latest/
- Script execution logs: logs/runs/latest/{script}/
- Archived logs: logs/debugger/archive/ and logs/runs/archive/
"""

from pathlib import Path


def get_debugger_root() -> Path:
    """Get the root debugger directory (mflux_debugger/)."""
    # Find the repo root by looking for pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "mflux_debugger"

    # Fallback: use current working directory
    return Path.cwd() / "mflux_debugger"


def get_logs_root() -> Path:
    """Get the logs root directory."""
    logs_dir = get_debugger_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


# Debugger server logs
def get_debugger_logs_dir() -> Path:
    """Get debugger server logs base directory."""
    dir_path = get_logs_root() / "debugger"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_debugger_logs_latest_dir() -> Path:
    """Get latest debugger server logs directory."""
    dir_path = get_debugger_logs_dir() / "latest"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_debugger_log_path(framework: str) -> Path:
    """Get path to current debugger server log file."""
    return get_debugger_logs_latest_dir() / f"{framework}_debugger.log"


def get_debugger_logs_archive_dir() -> Path:
    """Get debugger server logs archive directory."""
    dir_path = get_debugger_logs_dir() / "archive"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# Script execution runs
def get_runs_dir() -> Path:
    """Get script runs base directory."""
    dir_path = get_logs_root() / "runs"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_runs_latest_dir() -> Path:
    """Get latest runs directory."""
    dir_path = get_runs_dir() / "latest"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_runs_archive_dir() -> Path:
    """Get archived runs directory."""
    dir_path = get_runs_dir() / "archive"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_run_session_dir(script_name: str, timestamp: str) -> Path:
    """
    Get session directory for a script run.

    Returns: logs/runs/latest/{script_name}_{timestamp}/
    """
    session_dir = get_runs_latest_dir() / f"{script_name}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_run_checkpoints_dir(script_name: str, timestamp: str) -> Path:
    """Get checkpoints directory for a run session."""
    checkpoints_dir = get_run_session_dir(script_name, timestamp) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_run_output_log_path(script_name: str, timestamp: str) -> Path:
    """Get script output log file path."""
    return get_run_session_dir(script_name, timestamp) / "script_output.log"
