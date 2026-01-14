"""
Centralized log path management for the debugger.

Defines the directory structure for all logs:
- Debugger server logs: logs/debugger/latest/
- Script execution logs: logs/runs/latest/{script}/
- Archived logs: logs/debugger/archive/ and logs/runs/archive/
"""

from pathlib import Path


def get_debugger_root() -> Path:
    """
    Get the root debugger directory (mflux_debugger/).

    This function is designed to work from anywhere - it finds the mflux repository
    root by looking for pyproject.toml, then returns mflux_debugger/ directory.

    The function is robust and will work whether called from:
    - Within mflux repository code
    - From external repositories (like diffusers) that import mflux_debugger
    - From scripts in any location

    Returns:
        Path to mflux_debugger/ directory (always relative to mflux repo root)
    """
    # Strategy: Find mflux_debugger by locating this file, then find repo root
    # This file is at: src/mflux_debugger/log_paths.py
    # So we go up 2 levels to get to repo root
    current_file = Path(__file__).resolve()

    # Method 1: Go up from this file to find repo root (most reliable)
    # log_paths.py -> mflux_debugger -> src -> mflux (repo root)
    repo_root = current_file.parent.parent.parent
    if (repo_root / "pyproject.toml").exists():
        debugger_dir = repo_root / "mflux_debugger"
        debugger_dir.mkdir(parents=True, exist_ok=True)
        return debugger_dir

    # Method 2: Walk up from current file looking for pyproject.toml
    for parent in [current_file] + list(current_file.parents):
        if (parent / "pyproject.toml").exists():
            debugger_dir = parent / "mflux_debugger"
            debugger_dir.mkdir(parents=True, exist_ok=True)
            return debugger_dir

    # Method 3: Try to find mflux_debugger relative to common locations
    # Check if we're in a known location (e.g., diffusers repo)
    # Look for mflux_debugger in common parent directories
    for parent in current_file.parents:
        # Check if there's a sibling directory called "mflux" with mflux_debugger
        potential_mflux = parent.parent / "mflux" / "mflux_debugger"
        if potential_mflux.exists():
            return potential_mflux

    # Fallback: use current working directory (last resort)
    # This handles edge cases but should rarely be needed
    fallback_dir = Path.cwd() / "mflux_debugger"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir


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


def get_run_session_dir(script_name: str, timestamp: str, ab_run_id: str | None = None) -> Path:
    """
    Get session directory for a script run.

    Returns:
        logs/runs/latest/{script_name}_{timestamp}/              (no ab_run_id)
        logs/runs/latest/{ab_run_id}__{script_name}_{timestamp}/ (when ab_run_id is set)
    """
    base_name = f"{script_name}_{timestamp}"
    if ab_run_id:
        base_name = f"{ab_run_id}__{base_name}"

    session_dir = get_runs_latest_dir() / base_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_run_checkpoints_dir(script_name: str, timestamp: str, ab_run_id: str | None = None) -> Path:
    """Get checkpoints directory for a run session."""
    checkpoints_dir = get_run_session_dir(script_name, timestamp, ab_run_id=ab_run_id) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_run_output_log_path(script_name: str, timestamp: str, ab_run_id: str | None = None) -> Path:
    """Get script output log file path."""
    return get_run_session_dir(script_name, timestamp, ab_run_id=ab_run_id) / "script_output.log"


# Coverage reports
def get_coverage_dir() -> Path:
    """Get coverage base directory."""
    dir_path = get_debugger_root() / "coverage"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_coverage_latest_dir() -> Path:
    """Get latest coverage directory."""
    dir_path = get_coverage_dir() / "latest"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_coverage_archive_dir() -> Path:
    """Get coverage archive directory."""
    dir_path = get_coverage_dir() / "archive"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_coverage_session_dir(script_name: str, timestamp: str) -> Path:
    """
    Get session directory for a coverage run.

    Returns: coverage/latest/{script_name}_{timestamp}/
    """
    session_dir = get_coverage_latest_dir() / f"{script_name}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
