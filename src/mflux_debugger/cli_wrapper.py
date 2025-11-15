"""
CLI wrapper for the ML debugger with automatic session management.

Provides native terminal commands:
- mflux-debug-mlx: Debug MLX/MFLUX code (port 8000)
- mflux-debug-pytorch: Debug PyTorch/Diffusers code (port 8001)

Features:
- Automatic server lifecycle management
- Automatic cleanup of previous sessions
- Fixed unique ports for MLX vs PyTorch
- Clear error messages
- Built-in state validation
- Automatic checkpoint logging to JSON files

Usage:
    # Interactive debugging
    mflux-debug-mlx start <script>
    mflux-debug-mlx break <file> <line>
    mflux-debug-mlx continue
    mflux-debug-mlx step [over|into|out]
    mflux-debug-mlx eval <expression>
    mflux-debug-mlx vars
    mflux-debug-mlx status
    mflux-debug-mlx location
    mflux-debug-mlx log [--lines N] [--follow]
    mflux-debug-mlx terminate
"""

import argparse
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil
import requests

from mflux_debugger.log_paths import get_debugger_log_path, get_debugger_logs_archive_dir, get_debugger_logs_dir

# Port assignments
MLX_PORT = 8000
PYTORCH_PORT = 8001

# Server state files (runtime state in home directory)
STATE_DIR = Path.home() / ".mflux_debugger"
STATE_DIR.mkdir(exist_ok=True)

# Debugger server log files (now in logs/debugger/)
LOG_DIR = get_debugger_logs_dir()
LOG_ARCHIVE_DIR = get_debugger_logs_archive_dir()


class DebuggerCLI:
    """CLI wrapper for the debugger with automatic session management."""

    def __init__(self, framework: str, port: int):
        """
        Initialize CLI wrapper.

        Args:
            framework: "mlx" or "pytorch"
            port: Port number for this framework
        """
        self.framework = framework
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.state_file = STATE_DIR / f"{framework}_server.json"
        self.pid_file = STATE_DIR / f"{framework}_server.pid"
        self.log_file = get_debugger_log_path(framework)

    def _archive_old_logs(self):
        """Archive old log files with timestamp before starting new session."""
        if not self.log_file.exists():
            return

        # Get file modification time or current time
        try:
            mtime = self.log_file.stat().st_mtime
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(mtime))
        except Exception:  # noqa: BLE001
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Move old log to archive
        archive_name = f"{self.framework}_debugger_{timestamp}.log"
        archive_path = LOG_ARCHIVE_DIR / archive_name

        try:
            self.log_file.rename(archive_path)
            print(f"📦 Archived previous log to: {archive_path.name}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            # If archiving fails, just delete the old log
            print(f"⚠️  Could not archive old log: {e}", file=sys.stderr)
            try:
                self.log_file.unlink()
            except Exception:  # noqa: BLE001
                pass

    def _check_editable_installs(self):
        """Check if transformers and diffusers are installed in editable mode."""
        import subprocess

        try:
            # Use pip to check for editable installs
            result = subprocess.run(
                ["uv", "pip", "list", "--editable"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print(f"⚠️  Warning: Could not check editable installs: {result.stderr}", file=sys.stderr)
                return True  # Don't block

            editable_packages = result.stdout.lower()
            issues = []

            # Check if diffusers is editable
            if "diffusers" not in editable_packages:
                issues.append("⚠️  diffusers is NOT installed in editable mode")

            # Check if transformers is editable
            if "transformers" not in editable_packages:
                issues.append("⚠️  transformers is NOT installed in editable mode")

            if issues:
                print("\n" + "=" * 70, file=sys.stderr)
                print("🚨 EDITABLE INSTALL CHECK FAILED", file=sys.stderr)
                print("=" * 70, file=sys.stderr)
                for issue in issues:
                    print(f"   {issue}", file=sys.stderr)
                print("\n💡 Why this matters:", file=sys.stderr)
                print("   Breakpoints in library code (diffusers/transformers) won't work", file=sys.stderr)
                print("   unless they're installed in editable mode.\n", file=sys.stderr)
                print("💡 To fix:", file=sys.stderr)
                print("   1. Clone libraries to a local directory:", file=sys.stderr)
                print("      cd ~/Desktop  # or any directory", file=sys.stderr)
                print("      git clone https://github.com/huggingface/transformers.git", file=sys.stderr)
                print("      git clone https://github.com/huggingface/diffusers.git", file=sys.stderr)
                print("   2. Install in editable mode:", file=sys.stderr)
                print("      cd /path/to/mflux", file=sys.stderr)
                print("      uv pip install -e /path/to/transformers", file=sys.stderr)
                print("      uv pip install -e /path/to/diffusers", file=sys.stderr)
                print("   3. Restart this debug session\n", file=sys.stderr)
                print("=" * 70, file=sys.stderr)
                return False

            # Success! Show where they're installed
            try:
                import diffusers
                import transformers

                print("✅ Editable installs verified:", file=sys.stderr)
                print(f"   diffusers: {diffusers.__file__}", file=sys.stderr)
                print(f"   transformers: {transformers.__file__}\n", file=sys.stderr)
            except ImportError:
                print("✅ Editable installs configured (libraries not imported yet)\n", file=sys.stderr)

            return True

        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Warning: Could not verify editable installs: {e}", file=sys.stderr)
            return True  # Don't block on unexpected errors

    def _save_server_state(self, pid: int):
        """Save server PID to state file."""
        state = {
            "pid": pid,
            "port": self.port,
            "framework": self.framework,
            "started_at": time.time(),
        }
        self.state_file.write_text(json.dumps(state, indent=2))
        self.pid_file.write_text(str(pid))

    def _load_server_state(self) -> Optional[dict]:
        """Load server state if it exists."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:  # noqa: BLE001
                return None
        return None

    def _is_server_running(self) -> bool:
        """Check if debugger server is running."""
        state = self._load_server_state()
        if not state:
            return False

        pid = state.get("pid")
        if not pid:
            return False

        # Check if process exists and is our server
        try:
            process = psutil.Process(pid)
            cmdline = " ".join(process.cmdline())
            return "uvicorn" in cmdline and "mflux_debugger" in cmdline and str(self.port) in cmdline
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _kill_server(self):
        """Kill the debugger server if running."""
        state = self._load_server_state()
        if not state:
            return

        pid = state.get("pid")
        if not pid:
            return

        try:
            process = psutil.Process(pid)
            # Kill the process group to ensure all children are killed
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Fallback to killing just the process
                process.terminate()

            # Wait for termination
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass  # Already dead
        except psutil.TimeoutExpired:
            # Force kill if still alive
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Error killing server: {e}", file=sys.stderr)

        # Clean up state files
        if self.state_file.exists():
            self.state_file.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _cleanup_stale_sessions(self):
        """Kill any stale debug sessions and ensure fresh start."""
        print(f"🧹 Cleaning up stale {self.framework.upper()} debug sessions...", file=sys.stderr)

        # Kill our server if it's running
        self._kill_server()

        # Also kill any rogue processes on our port
        killed_count = 0
        processes_to_kill = []

        # First pass: identify processes to kill
        # Note: try-except in loop is necessary here as processes can disappear during iteration
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "uvicorn" in cmdline and "mflux_debugger" in cmdline and str(self.port) in cmdline:
                    processes_to_kill.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # noqa: PERF203
                pass

        # Second pass: kill identified processes
        for proc in processes_to_kill:
            try:
                proc.terminate()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # noqa: PERF203
                pass

        if killed_count > 0:
            print(f"✅ Killed {killed_count} stale process(es)", file=sys.stderr)

        # Wait a moment for cleanup
        time.sleep(1)

    def _ensure_server_running(self):
        """Ensure debugger server is running, starting it if needed."""
        if self._is_server_running():
            return True

        # Archive old logs before starting new session
        self._archive_old_logs()

        # Server not running, start it
        print(f"🚀 Starting {self.framework.upper()} debugger server on port {self.port}...", file=sys.stderr)

        # Find the mflux project root
        debugger_path = Path(__file__).parent
        project_root = debugger_path.parent.parent

        # Start server in background
        cmd = [
            "uv",
            "run",
            "uvicorn",
            "mflux_debugger.fastapi_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]

        # Write session start marker with timestamp
        with self.log_file.open("w") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write("=" * 80 + "\n")
            f.write(f"🚀 NEW DEBUG SESSION STARTED: {timestamp}\n")
            f.write(f"Framework: {self.framework.upper()}\n")
            f.write(f"Port: {self.port}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            # Start in new process group so we can kill it cleanly
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        # Save PID
        self._save_server_state(process.pid)

        # Wait for server to be ready
        max_wait = 10
        for i in range(max_wait):
            time.sleep(1)
            try:
                response = requests.get(f"{self.base_url}/", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Server ready on port {self.port}", file=sys.stderr)
                    return True
            except requests.RequestException:
                pass

        print(f"❌ Server failed to start after {max_wait}s", file=sys.stderr)
        print(f"   Check logs: {self.log_file}", file=sys.stderr)
        return False

    def _api_call(self, method: str, endpoint: str, data: Optional[dict] = None, timeout: int = 5) -> dict:
        """
        Make an API call to the debugger server.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., "/debug/status")
            data: Request body (for POST)
            timeout: Request timeout in seconds

        Returns:
            Response JSON

        Raises:
            SystemExit: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data or {}, timeout=timeout)
            else:
                print(f"❌ Unsupported HTTP method: {method}", file=sys.stderr)
                sys.exit(1)

            return response.json()

        except requests.ConnectionError:
            print(f"❌ Cannot connect to debugger server on port {self.port}", file=sys.stderr)
            print(f"   Is the server running? Try: mflux-debug-{self.framework} start <script>", file=sys.stderr)
            sys.exit(1)
        except requests.Timeout:
            print(f"❌ Request timeout after {timeout}s", file=sys.stderr)
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"❌ Request failed: {e}", file=sys.stderr)
            sys.exit(1)

    def _format_response(self, response: dict):
        """Format and print API response."""
        success = response.get("success", False)
        message = response.get("message", "")
        data = response.get("data")
        error = response.get("error")

        if success:
            print(message)

            # Special handling for checkpoint info in location responses
            if data and "checkpoint" in data:
                checkpoint = data["checkpoint"]
                if checkpoint:
                    cp_name = checkpoint.get("name", "unknown")
                    cp_hit = checkpoint.get("hit_count", "?")
                    cp_context = checkpoint.get("context", {})
                    cp_vars = checkpoint.get("variables", {})

                    cp_verified = checkpoint.get("verified", False)
                    verified_marker = " ✅ VERIFIED" if cp_verified else ""
                    print(f"\n🎯 Checkpoint: '{cp_name}' (hit #{cp_hit}){verified_marker}")
                    if cp_context:
                        context_str = ", ".join([f"{k}={v}" for k, v in cp_context.items()])
                        print(f"   Context: {context_str}")

                    # Display checkpoint variables (focus on ACTUAL TENSOR VALUES, not statistics!)
                    if cp_vars:
                        print("\n📊 Checkpoint Values (actual tensor values):")
                        print("   ⚠️  TIP: Always compare actual values, not means/statistics!")
                        print("   💡 Use 'eval <tensor>' to inspect full tensor values\n")

                        # First pass: show shapes and samples (actual values)
                        for var_name, var_value in sorted(cp_vars.items()):
                            # Handle tensor shapes (usually end with _shape)
                            if var_name.endswith("_shape") and isinstance(var_value, (list, tuple)):
                                shape_str = " × ".join(str(d) for d in var_value)
                                print(f"   • {var_name}: [{shape_str}]")

                            # Handle tensor samples (usually end with _sample) - THIS IS WHAT WE WANT!
                            elif var_name.endswith("_sample") and isinstance(var_value, list):
                                self._print_tensor_sample(var_name, var_value)

                            # Handle integers and booleans (loop indexes, flags)
                            elif isinstance(var_value, (int, bool)) and not var_name.endswith(
                                ("_mean", "_std", "_min", "_max")
                            ):
                                print(f"   • {var_name}: {var_value}")

                            # Handle floats (scalars) - but skip statistics
                            elif isinstance(var_value, float) and not var_name.endswith(
                                ("_mean", "_std", "_min", "_max")
                            ):
                                if abs(var_value) < 0.001 and var_value != 0:
                                    print(f"   • {var_name}: {var_value:.6e}")
                                else:
                                    print(f"   • {var_name}: {var_value:.6f}")

                            # Handle lists/tuples (could be shapes or simple lists)
                            elif isinstance(var_value, (list, tuple)) and not var_name.endswith(
                                ("_mean", "_std", "_min", "_max")
                            ):
                                if len(var_value) <= 10:
                                    print(f"   • {var_name}: {var_value}")
                                else:
                                    preview = list(var_value[:5]) + ["..."] + list(var_value[-5:])
                                    print(f"   • {var_name}: {preview} ({len(var_value)} items)")

                            # Skip statistics (mean, std, min, max) - they hide real differences!
                            # Everything else (non-statistics)
                            elif not var_name.endswith(("_mean", "_std", "_min", "_max")):
                                print(f"   • {var_name}: {var_value}")

            if data:
                print("\n📊 Data:")
                print(json.dumps(data, indent=2))
        else:
            print(message, file=sys.stderr)
            if error:
                print(f"\n❌ Error: {error}", file=sys.stderr)
            sys.exit(1)

    def _print_tensor_sample(self, var_name: str, sample_data: list):
        """Print tensor sample values intelligently (first/last 10, per dimension)."""
        if not sample_data:
            print(f"   • {var_name}: []")
            return

        # Determine if it's nested (multi-dimensional) or flat
        is_nested = isinstance(sample_data[0], list) if sample_data else False

        if is_nested:
            # Multi-dimensional tensor
            print(f"   • {var_name}:")
            for i, row in enumerate(sample_data[:10]):  # First 10 rows
                if isinstance(row, list):
                    if len(row) <= 20:
                        row_str = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in row])
                    else:
                        first_10 = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in row[:10]])
                        last_10 = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in row[-10:]])
                        row_str = f"{first_10}, ..., {last_10} ({len(row)} values)"
                    print(f"      [{i}]: [{row_str}]")
                else:
                    print(f"      [{i}]: {row}")

            if len(sample_data) > 10:
                print(f"      ... ({len(sample_data)} total rows)")
                # Show last row too
                if len(sample_data) > 10:
                    last_row = sample_data[-1]
                    if isinstance(last_row, list):
                        if len(last_row) <= 20:
                            row_str = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in last_row])
                        else:
                            first_10 = ", ".join(
                                [f"{v:.6f}" if isinstance(v, float) else str(v) for v in last_row[:10]]
                            )
                            last_10 = ", ".join(
                                [f"{v:.6f}" if isinstance(v, float) else str(v) for v in last_row[-10:]]
                            )
                            row_str = f"{first_10}, ..., {last_10} ({len(last_row)} values)"
                        print(f"      [{len(sample_data) - 1}]: [{row_str}]")
        else:
            # 1D tensor - show first and last 10
            if len(sample_data) <= 20:
                values_str = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in sample_data])
                print(f"   • {var_name}: [{values_str}]")
            else:
                first_10 = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in sample_data[:10]])
                last_10 = ", ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in sample_data[-10:]])
                print(f"   • {var_name}: [{first_10}, ..., {last_10}] ({len(sample_data)} values)")

    def call_api(self, endpoint: str, data: Optional[dict] = None, method: str = "POST") -> dict:
        """
        Call API endpoint and return response (for use by joint CLI).

        Args:
            endpoint: API endpoint path
            data: Request data
            method: HTTP method (GET or POST)

        Returns:
            Response dictionary
        """
        return self._api_call(method, endpoint, data)

    # Command implementations

    def cmd_start(self, script_path: str, reset: bool = True, keep_tensors: bool = False):
        """
        Start a debugging session.

        Args:
            script_path: Path to script to debug
            reset: Whether to reset and cleanup previous sessions (default: True)
            keep_tensors: If True, keep saved tensors (disable default clearing for PyTorch)
        """
        if reset:
            # Always kill previous sessions for a fresh start
            self._cleanup_stale_sessions()

        # Ensure server is running
        if not self._ensure_server_running():
            sys.exit(1)

        # Check editable installs (only for pytorch framework, as it uses diffusers/transformers)
        if self.framework == "pytorch":
            if not self._check_editable_installs():
                print("\n❌ Cannot start debugging without proper editable installs", file=sys.stderr)
                print("   Breakpoints in library code will not work!", file=sys.stderr)
                sys.exit(1)

        # Check if another debug session is active
        status_response = self._api_call("GET", "/debug/status", timeout=2)
        current_state = status_response.get("data", {}).get("state")

        if current_state and current_state not in ["not_started", "finished", "failed"]:
            print(f"⚠️  WARNING: Another debug session is {current_state}!", file=sys.stderr)
            print(f"   You should terminate it first: mflux-debug-{self.framework} terminate", file=sys.stderr)
            print("   Or use --force to reset automatically", file=sys.stderr)
            sys.exit(1)

        # Determine clear_tensors behavior: None = default (clear for PyTorch), False = keep, True = force clear
        clear_tensors = None if not keep_tensors else False

        # Start new session
        response = self._api_call(
            "POST",
            "/debug/start",
            {"script_path": script_path, "framework": self.framework, "clear_tensors": clear_tensors},
        )
        self._format_response(response)

        # Store script path for checkpoint discovery
        self._store_current_script(script_path)

    def _store_current_script(self, script_path: str):
        """Store the current script path for checkpoint discovery."""
        script_state_file = STATE_DIR / f"{self.framework}_current_script.txt"
        try:
            with open(script_state_file, "w") as f:
                f.write(str(Path(script_path).resolve()))
        except Exception:  # noqa: BLE001
            pass

    def _get_current_script(self) -> Optional[Path]:
        """Get the currently debugging script path."""
        script_state_file = STATE_DIR / f"{self.framework}_current_script.txt"
        try:
            if script_state_file.exists():
                with open(script_state_file) as f:
                    return Path(f.read().strip())
        except Exception:  # noqa: BLE001
            pass
        return None

    def _get_relevant_directories(self, script_path: Path) -> list[Path]:
        """
        Get directories relevant to the script based on imports.

        Returns directories for imported packages and editable installs.
        """
        scan_dirs = [script_path.parent]  # Always include script directory
        editable_dirs = self._get_editable_install_dirs()

        # For PyTorch debugging: scan all editable installs
        # (diffusers uses transformers internally, so we can't rely on direct imports)
        for editable_dir in editable_dirs:
            # Check both src/package and package layouts
            for pkg_name in ["diffusers", "transformers"]:
                src_dir = editable_dir / "src" / pkg_name
                if src_dir.exists():
                    scan_dirs.append(src_dir)
                else:
                    pkg_dir = editable_dir / pkg_name
                    if pkg_dir.exists():
                        scan_dirs.append(pkg_dir)

        return scan_dirs

    def _get_editable_install_dirs(self) -> list[Path]:
        """
        Get editable install directories from pyproject.toml if using PyTorch debugger.

        Returns:
            List of Path objects for editable installs (transformers, diffusers)
        """
        if self.framework != "pytorch":
            return []

        # Find pyproject.toml in the current directory or parent directories
        current = Path.cwd()
        pyproject_path = None
        for parent in [current, *current.parents]:
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                pyproject_path = candidate
                break

        if not pyproject_path:
            return []

        try:
            # Try Python 3.11+ built-in tomllib first
            try:
                import tomllib  # noqa: PLC0415

                toml_lib = tomllib
            except ImportError:
                # Fallback to tomli for Python 3.10
                try:
                    import tomli  # noqa: PLC0415

                    toml_lib = tomli  # type: ignore[assignment]
                except ImportError:
                    # If neither available, parse manually
                    return self._parse_pyproject_manually(pyproject_path)

            with open(pyproject_path, "rb") as f:
                pyproject = toml_lib.load(f)

            # Look for editable dependencies
            deps = pyproject.get("project", {}).get("dependencies", [])
            editable_dirs = []

            for dep in deps:
                # Look for dict-style dependencies like {path = "...", editable = true}
                if isinstance(dep, dict) and dep.get("editable"):
                    path = dep.get("path")
                    if path:
                        abs_path = (pyproject_path.parent / path).resolve()
                        if abs_path.exists():
                            editable_dirs.append(abs_path)

            # Also check tool.uv.sources for editable installs
            sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
            for name, config in sources.items():
                if isinstance(config, dict) and config.get("editable") and config.get("path"):
                    abs_path = (pyproject_path.parent / config["path"]).resolve()
                    if abs_path.exists() and name in ["transformers", "diffusers"]:
                        editable_dirs.append(abs_path)

            return editable_dirs
        except Exception:  # noqa: BLE001
            return []

    def _parse_pyproject_manually(self, pyproject_path: Path) -> list[Path]:
        """Manually parse pyproject.toml for editable paths (fallback when tomllib/tomli not available)."""
        try:
            with open(pyproject_path, encoding="utf-8") as f:
                content = f.read()

            editable_dirs = []
            # Look for patterns like: transformers = { path = "/path/to/transformers", editable = true }
            # Simple regex-based extraction for [tool.uv.sources] section
            for line in content.split("\n"):
                for lib_name in ["transformers", "diffusers"]:
                    if f"{lib_name} =" in line and "path =" in line and "editable" in line:
                        # Extract path from the line
                        match = re.search(r'path\s*=\s*"([^"]+)"', line)
                        if match:
                            path_str = match.group(1)
                            abs_path = (pyproject_path.parent / path_str).resolve()
                            if abs_path.exists():
                                editable_dirs.append(abs_path)

            return editable_dirs
        except Exception:  # noqa: BLE001
            return []

    def _scan_for_checkpoints(self, root_dir: Path) -> list[tuple[str, int, str]]:
        """
        Scan Python files for debug_checkpoint() calls.

        Returns:
            List of (file_path, line_number, checkpoint_name) tuples
        """
        checkpoints = []
        # Pattern to match: debug_checkpoint( "name" or debug_checkpoint("name")
        # Handles multiline calls
        pattern = re.compile(r'debug_checkpoint\s*\(\s*["\']([^"\']+)["\']', re.MULTILINE)

        # Scan all Python files in the directory
        for py_file in root_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Find all matches in the file content
                for match in pattern.finditer(content):
                    checkpoint_name = match.group(1)
                    # Calculate line number by counting newlines before the match
                    line_num = content[: match.start()].count("\n") + 1
                    checkpoints.append((str(py_file), line_num, checkpoint_name))
            except Exception:  # noqa: BLE001, PERF203
                # Skip files that can't be read
                continue

        return checkpoints

    def cmd_break(self, file_path: str, line: int, condition: Optional[str] = None):
        """
        Set a *line* breakpoint (DISABLED).

        Line-based breakpoints are intentionally disabled in favor of semantic
        checkpoints (debug_checkpoint_* helpers and checkpoint-break).

        This CLI command now only prints an error and does NOT contact the
        debugger service.
        """
        print(
            "\n❌ Line breakpoints via CLI are disabled.\n"
            "   Use semantic checkpoints instead (debug_checkpoint_*/checkpoint-break).\n"
            "   Example for A/B mode:\n"
            "     mflux-debug-pytorch checkpoint-break pytorch_A\n"
            "     mflux-debug-pytorch checkpoint-break pytorch_B\n",
            file=sys.stderr,
        )

    def cmd_checkpoint_break(
        self,
        checkpoint_name: Optional[str] = None,
        context: Optional[str] = None,
        all_checkpoints: bool = False,
        pattern: Optional[str] = None,
        script_dir: Optional[Path] = None,
    ):
        """
        Set conditional breakpoints on semantic checkpoints.

        Args:
            checkpoint_name: Single checkpoint name to break on
            context: Context condition as key=value pairs
            all_checkpoints: If True, set breakpoints on all discovered checkpoints
            pattern: Glob pattern to match checkpoint names (e.g., "txt2img_*", "*rope*")
            script_dir: Directory to scan for checkpoints (default: current working directory)
        """
        # Warning: This is not the recommended approach
        print("\n⚠️  WARNING: Interactive checkpoint breakpoints are NOT recommended!", file=sys.stderr)
        print("📝 RECOMMENDED: Use debug_checkpoint(skip=False) directly in code instead.", file=sys.stderr)
        print("   This provides better maintainability and clarity.\n", file=sys.stderr)

        condition_dict = {}

        # Parse context string like "block=0,timestep=0,hit_count=1"
        if context:
            for pair in context.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to convert to int if possible
                    try:
                        condition_dict[key] = int(value)
                    except ValueError:
                        condition_dict[key] = value

        # Handle bulk checkpoint setting
        if all_checkpoints or pattern:
            if script_dir is None:
                script_dir = Path.cwd()

            # Get all directories to scan
            scan_dirs = [script_dir]
            editable_dirs = self._get_editable_install_dirs()
            scan_dirs.extend(editable_dirs)

            # Scan for checkpoints in all directories
            if editable_dirs:
                print("🔍 Scanning for checkpoints in:", file=sys.stderr)
                print(f"   • {script_dir} (main)", file=sys.stderr)
                for ed in editable_dirs:
                    print(f"   • {ed} (editable)", file=sys.stderr)
            else:
                print(f"🔍 Scanning for checkpoints in {script_dir}...", file=sys.stderr)

            checkpoints = []
            for scan_dir in scan_dirs:
                checkpoints.extend(self._scan_for_checkpoints(scan_dir))

            if not checkpoints:
                print("❌ No checkpoints found", file=sys.stderr)
                return

            # Filter by pattern if provided
            if pattern:
                checkpoint_names = [cp[2] for cp in checkpoints]
                matched_names = [name for name in checkpoint_names if fnmatch.fnmatch(name, pattern)]
                checkpoints = [cp for cp in checkpoints if cp[2] in matched_names]

                if not matched_names:
                    print(f"❌ No checkpoints match pattern: {pattern}", file=sys.stderr)
                    return

            # Get unique checkpoint names
            unique_names = sorted(set(cp[2] for cp in checkpoints))

            print(f"\n📋 Found {len(unique_names)} checkpoint(s):", file=sys.stderr)
            for name in unique_names:
                print(f"   • {name}", file=sys.stderr)

            # Set breakpoints for all
            print("\n⚙️  Setting breakpoints...", file=sys.stderr)
            success_count = 0
            for name in unique_names:
                try:
                    data = {"checkpoint_name": name, "condition": condition_dict}
                    response = self._api_call("POST", "/debug/checkpoint/break", data)
                    success_count += 1
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"   ⚠️  Failed to set breakpoint for '{name}': {e}", file=sys.stderr)

            print(f"\n✅ Set {success_count}/{len(unique_names)} checkpoint breakpoints", file=sys.stderr)
            return

        # Single checkpoint setting
        if not checkpoint_name:
            print("❌ Error: Must provide checkpoint_name, --all, or --pattern", file=sys.stderr)
            sys.exit(1)

        data = {"checkpoint_name": checkpoint_name, "condition": condition_dict}
        response = self._api_call("POST", "/debug/checkpoint/break", data)
        self._format_response(response)

    def cmd_checkpoint_list(self, script_dir: Optional[Path] = None):
        """List checkpoints relevant to the current debug session."""
        # Try to get the current script from the active session
        current_script = self._get_current_script()

        if current_script and current_script.exists():
            # Get relevant directories based on imports
            print(f"🔍 Analyzing imports from: {current_script.name}\n", file=sys.stderr)
            scan_dirs = self._get_relevant_directories(current_script)

            # Scan only relevant directories
            checkpoints = []
            for scan_dir in scan_dirs:
                checkpoints.extend(self._scan_for_checkpoints(scan_dir))
        else:
            # Fallback: scan the current workspace only
            if script_dir is None:
                script_dir = Path.cwd()

            print(f"🔍 Scanning workspace: {script_dir}\n", file=sys.stderr)
            print("💡 Tip: Start a debug session first for more accurate results\n", file=sys.stderr)
            checkpoints = self._scan_for_checkpoints(script_dir)

        if not checkpoints:
            print("❌ No checkpoints found", file=sys.stderr)
            return

        # Group by checkpoint name
        by_name = {}
        for file_path, line_num, name in checkpoints:
            if name not in by_name:
                by_name[name] = []
            by_name[name].append((file_path, line_num))

        print(f"📋 Found {len(by_name)} unique checkpoint(s):\n", file=sys.stderr)

        for name in sorted(by_name.keys()):
            locations = by_name[name]
            print(f"  • {name}", file=sys.stderr)
            for file_path, line_num in locations:
                # Show relative path
                try:
                    if current_script:
                        # Try relative to script dir or editable install dirs
                        for root in [current_script.parent, *self._get_editable_install_dirs()]:
                            try:
                                rel_path = Path(file_path).relative_to(root)
                                break
                            except ValueError:
                                continue
                        else:
                            rel_path = Path(file_path).name
                    else:
                        rel_path = Path(file_path).relative_to(Path.cwd())
                except ValueError:
                    rel_path = Path(file_path).name

                print(f"      {rel_path}:{line_num}", file=sys.stderr)

        print("\n💡 To set breakpoints:", file=sys.stderr)
        print(f"   Single: mflux-debug-{self.framework} checkpoint-break <name>", file=sys.stderr)
        print(f"   All:    mflux-debug-{self.framework} checkpoint-break --all", file=sys.stderr)
        print(f"   Pattern: mflux-debug-{self.framework} checkpoint-break --pattern 'rope_*'", file=sys.stderr)

    def cmd_remove_break(self, file_path: str, line: int):
        """Remove a breakpoint."""
        data = {"file_path": file_path, "line": line}
        response = self._api_call("POST", "/debug/breakpoint/remove", data)
        self._format_response(response)

    def cmd_list_breaks(self):
        """List all breakpoints."""
        response = self._api_call("GET", "/debug/breakpoints")
        self._format_response(response)

    def cmd_continue(self, max_wait: int = 120):
        """
        Continue execution.

        The command automatically polls every 2 seconds until execution pauses at a breakpoint,
        completes, or fails. This is the standard workflow - no need for manual status checks.

        Args:
            max_wait: Maximum seconds to poll (default: 120)
        """
        endpoint = "/debug/continue_async"
        response = self._api_call("POST", endpoint)
        self._format_response(response)

        # Always poll for completion
        if True:
            print(f"\n⏳ Polling for breakpoint (max {max_wait}s)...", file=sys.stderr)
            interval = 2
            iterations = max_wait // interval

            for i in range(iterations):
                time.sleep(interval)
                status_response = self._api_call("GET", "/debug/status", timeout=2)
                state = status_response.get("data", {}).get("state")
                print(f"[{i + 1}/{iterations}] Status: {state}", file=sys.stderr)

                if state == "paused":
                    print("✅ Hit breakpoint!", file=sys.stderr)
                    # Show location
                    loc_response = self._api_call("GET", "/debug/location")
                    self._format_response(loc_response)
                    break
                elif state == "failed":
                    print("❌ Script execution failed!", file=sys.stderr)
                    # Show error details automatically
                    self.cmd_status(verbose=False)
                    break
                elif state == "finished":
                    print("✅ Execution finished", file=sys.stderr)
                    break

    def cmd_eval(self, expression: str):
        """Evaluate an expression."""
        # Warn if evaluating statistics instead of actual values
        expr_lower = expression.lower().strip()
        if any(
            stat in expr_lower for stat in [".mean()", ".std()", ".min()", ".max()", "mean(", "std(", "min(", "max("]
        ):
            print("⚠️  WARNING: You're evaluating statistics (mean/std/min/max)!", file=sys.stderr)
            print("   💡 TIP: Compare ACTUAL tensor values instead - they reveal real differences!", file=sys.stderr)
            print("   💡 Example: eval 'tensor[0, 0, :10]' to see first 10 values\n", file=sys.stderr)

        response = self._api_call("POST", "/debug/evaluate", {"expression": expression})
        self._format_response(response)

    def cmd_vars(self):
        """List variables in current scope."""
        response = self._api_call("GET", "/debug/variables")
        self._format_response(response)

    def cmd_checkpoint_info(self):
        """Show current checkpoint information with captured values."""
        response = self._api_call("GET", "/debug/location")
        if response.get("success"):
            data = response.get("data", {})
            checkpoint = data.get("checkpoint")
            if checkpoint:
                # Use the enhanced display which already shows checkpoint values
                self._format_response(response)
            else:
                print("❌ Not currently at a checkpoint")
        else:
            print(response.get("message", "❌ Failed to get checkpoint info"))

    def cmd_status(self, verbose: bool = False):
        """Check debugger status with optional error details."""
        response = self._api_call("GET", "/debug/status")

        # Enhanced error display
        if not response.get("success") and response.get("data", {}).get("state") == "failed":
            print(f"❌ {response.get('message', 'Script failed')}", file=sys.stderr)

            error_info = response.get("data", {}).get("error", {})
            if error_info:
                print(f"\n🐛 Error Type: {error_info.get('type', 'Unknown')}", file=sys.stderr)
                print(f"💬 Message: {error_info.get('message', 'No message')}", file=sys.stderr)

                if verbose and error_info.get("traceback"):
                    print(f"\n📋 Traceback:\n{error_info['traceback']}", file=sys.stderr)
                elif error_info.get("traceback"):
                    # Show just the last few lines
                    traceback_lines = error_info["traceback"].split("\n")
                    relevant_lines = [line for line in traceback_lines if line.strip()][-10:]
                    print("\n📋 Traceback (last 10 lines):\n" + "\n".join(relevant_lines), file=sys.stderr)
                    print("\n💡 Use --verbose for full traceback", file=sys.stderr)
        else:
            self._format_response(response)

    def cmd_location(self):
        """Get current execution location."""
        response = self._api_call("GET", "/debug/location")
        self._format_response(response)

    def cmd_terminate(self):
        """Terminate debugging session."""
        response = self._api_call("POST", "/debug/terminate", timeout=10)
        self._format_response(response)

    def cmd_log(self, lines: int = 50, follow: bool = False):
        """View script output logs (what the script being debugged prints)."""
        from mflux_debugger.log_paths import get_runs_latest_dir

        # Find the most recent script_output.log file
        runs_latest_dir = get_runs_latest_dir()
        script_output_logs = list(runs_latest_dir.glob("*/script_output.log"))

        if not script_output_logs:
            print(f"❌ No script output log found in {runs_latest_dir}", file=sys.stderr)
            print(f"   Start a debug session first with: mflux-debug-{self.framework} start <script>", file=sys.stderr)
            return

        # Get the most recent one (by modification time)
        script_output_log = max(script_output_logs, key=lambda p: p.stat().st_mtime)

        if follow:
            # Follow logs in real-time (like tail -f)
            print("📋 Following script output logs (Ctrl+C to stop)...\n", file=sys.stderr)
            print(f"   File: {script_output_log}\n", file=sys.stderr)
            try:
                # Show last N lines first
                with script_output_log.open("r") as f:
                    all_lines = f.readlines()
                    for line in all_lines[-lines:]:
                        print(line, end="")

                # Then follow new lines
                with script_output_log.open("r") as f:
                    # Seek to end
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            print(line, end="")
                        else:
                            time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n✅ Stopped following logs", file=sys.stderr)
        else:
            # Show last N lines
            print(f"📋 Last {lines} lines of script output:\n", file=sys.stderr)
            print(f"   File: {script_output_log}\n", file=sys.stderr)
            with script_output_log.open("r") as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    print(line, end="")

            print("\n💡 Tip: Use --follow to watch logs in real-time", file=sys.stderr)
            print("💡 Tip: Use --lines N to show more/fewer lines", file=sys.stderr)

    # Debug Save/Load Commands
    def cmd_debug_list(self):
        """List all saved debug tensors."""
        response = self._api_call("GET", "/debug_tensors/list")

        if response.get("success"):
            data = response.get("data", {})
            tensors = data.get("tensors", [])

            if not tensors:
                print("📋 No saved debug tensors found")
                return

            print(f"📋 Saved debug tensors ({len(tensors)}):")
            for name in tensors:
                print(f"   • {name}")
        else:
            print(f"❌ {response.get('message', 'Failed to list tensors')}", file=sys.stderr)

    def cmd_debug_info(self, name: str):
        """Get information about a saved debug tensor."""
        response = self._api_call("GET", f"/debug_tensors/info/{name}")

        if response.get("success"):
            data = response.get("data", {})
            print(f"📊 Tensor: {data.get('name')}")
            print(f"   Shape: {data.get('shape')}")
            print(f"   Dtype: {data.get('dtype')}")
            print(f"   Size: {data.get('size_mb', 0):.2f} MB")
            print(f"   File: {data.get('file')}")
        else:
            print(f"❌ {response.get('message', 'Failed to get tensor info')}", file=sys.stderr)

    def cmd_debug_clear(self, name: Optional[str] = None):
        """Clear saved debug tensors."""
        data = {"name": name, "confirm": False}  # CLI handles confirmation separately

        # Ask for confirmation
        if name:
            response = input(f"⚠️  Delete debug tensor '{name}'? [y/N]: ").strip().lower()
        else:
            response = input("⚠️  Delete ALL saved debug tensors? [y/N]: ").strip().lower()

        if response != "y":
            print("❌ Cancelled")
            return

        api_response = self._api_call("POST", "/debug_tensors/clear", data)

        if api_response.get("success"):
            deleted = api_response.get("data", {}).get("deleted", 0)
            print(f"✅ Deleted {deleted} file(s)")
        else:
            print(f"❌ {api_response.get('message', 'Failed to clear tensors')}", file=sys.stderr)

    def cmd_coverage(self, script: str, output: Optional[str] = None, include_dirs: Optional[list[str]] = None):
        """Run script with coverage tracking to find dead code.

        Args:
            script: Path to script to analyze
            output: Optional output path for coverage report
            include_dirs: Optional list of additional directories to include (default: src/mflux only)
        """
        from pathlib import Path

        from mflux_debugger.coverage_report import generate_marked_up_file

        script_path = Path(script).resolve()
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}", file=sys.stderr)
            sys.exit(1)

        print("=" * 70, file=sys.stderr)
        print("🔍 COVERAGE ANALYSIS", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Script: {script_path}", file=sys.stderr)
        print("Running script with coverage tracking...\n", file=sys.stderr)

        # CRITICAL: Kill all previous debug sessions first (like cmd_start does)
        print("🧹 Cleaning up previous debug sessions...", file=sys.stderr)
        self._cleanup_stale_sessions()

        # Ensure server is running
        if not self._ensure_server_running():
            sys.exit(1)

        # Start session with coverage enabled
        data = {"script_path": str(script_path), "coverage_mode": True}
        response = self._api_call("POST", "/debug/start", data)

        if not response.get("success"):
            print(f"❌ Failed to start session: {response.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)

        # CRITICAL: Disable ALL checkpoint breaking in coverage mode BEFORE starting execution
        # This ensures checkpoints never pause execution
        print("⚙️  Disabling checkpoint breaks for full execution...", file=sys.stderr)
        self._api_call("POST", "/debug/checkpoint/break-all", {"enabled": False})

        # Run script to completion
        print("⏳ Running script (no breakpoints will pause execution)...", file=sys.stderr)
        self._api_call("POST", "/debug/continue_async")

        # Poll until finished
        max_wait = 300  # 5 minutes max
        interval = 2
        iterations = max_wait // interval

        print(
            "⏳ Waiting for script to complete (this may take a while for model loading/inference)...", file=sys.stderr
        )

        for i in range(iterations):
            time.sleep(interval)
            status_response = self._api_call("GET", "/debug/status", timeout=2)
            state = status_response.get("data", {}).get("state")

            # Show progress every 10 seconds
            if (i + 1) % 5 == 0:
                elapsed = (i + 1) * interval
                print(f"   [{elapsed}s] Status: {state}...", file=sys.stderr)

            if state == "finished":
                print("✅ Script completed", file=sys.stderr)
                break
            elif state == "failed":
                print("❌ Script execution failed", file=sys.stderr)
                self.cmd_status(verbose=False)
                sys.exit(1)
        else:
            print("⚠️  Script did not complete within timeout", file=sys.stderr)
            sys.exit(1)

        # Get coverage data
        print("\n📊 Collecting coverage data...", file=sys.stderr)
        coverage_response = self._api_call("GET", "/debug/coverage")

        if not coverage_response.get("success"):
            print(f"❌ Failed to get coverage data: {coverage_response.get('error')}", file=sys.stderr)
            sys.exit(1)

        coverage_data = coverage_response.get("data", {}).get("coverage_data", {})

        if not coverage_data:
            print("⚠️  No coverage data collected", file=sys.stderr)
            sys.exit(1)

        # Convert to sets
        coverage_sets = {file: set(lines) for file, lines in coverage_data.items()}

        # Filter to only include mflux project files (exclude diffusers/transformers)
        # Find project root by looking for pyproject.toml
        project_root = script_path.parent
        while project_root != project_root.parent:
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent

        # Filter coverage data - default to src/mflux, but allow additional directories
        # Exclude: diffusers, transformers (external libraries)
        # Default include: src/mflux/
        # Optional include: additional directories via --include flag

        # Build list of directories to include
        # Default: src/mflux/ and src/mflux_debugger/examples/ (examples are allowed)
        include_patterns = ["src/mflux/", "src/mflux_debugger/examples/"]
        if include_dirs:
            # Normalize include directories (ensure they end with /)
            for include_dir in include_dirs:
                normalized = include_dir.replace("\\", "/")
                if not normalized.endswith("/"):
                    normalized += "/"
                include_patterns.append(normalized)

        mflux_coverage_sets = {}
        excluded_count = 0
        for file_path, lines in coverage_sets.items():
            # Normalize path for cross-platform compatibility
            path_normalized = file_path.replace("\\", "/")

            # Skip files from external libraries (always exclude)
            if "/diffusers/" in path_normalized or "/transformers/" in path_normalized:
                excluded_count += 1
                continue

            # Always exclude debugger implementation code (but allow examples)
            # Exclude src/mflux_debugger/ but allow src/mflux_debugger/examples/
            if "/mflux_debugger/" in path_normalized and "/mflux_debugger/examples/" not in path_normalized:
                excluded_count += 1
                continue

            # Check if file matches any include pattern
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.is_absolute():
                    # Check if file is under project root
                    try:
                        rel_path = file_path_obj.relative_to(project_root)
                        rel_path_str = str(rel_path).replace("\\", "/")

                        # Check if file matches any include pattern
                        matches_pattern = False
                        for pattern in include_patterns:
                            if rel_path_str.startswith(pattern):
                                matches_pattern = True
                                break

                        if matches_pattern:
                            mflux_coverage_sets[file_path] = lines
                        else:
                            excluded_count += 1
                            continue
                    except ValueError:
                        # File is not under project root, skip it
                        excluded_count += 1
                        continue
                else:
                    # Relative path - check if it matches any include pattern
                    rel_path_str = file_path.replace("\\", "/")
                    matches_pattern = False
                    for pattern in include_patterns:
                        if rel_path_str.startswith(pattern):
                            matches_pattern = True
                            break

                    if matches_pattern:
                        mflux_coverage_sets[file_path] = lines
                    else:
                        excluded_count += 1
                        continue
            except Exception:  # noqa: BLE001
                # Skip files we can't parse
                excluded_count += 1
                continue

        # Build description of what's being analyzed
        if include_dirs:
            analyzed_dirs = ", ".join(include_patterns)
            print(f"📋 Filtered out {excluded_count} files (analyzing: {analyzed_dirs})", file=sys.stderr)
        else:
            print(f"📋 Filtered out {excluded_count} files (only analyzing src/mflux code)", file=sys.stderr)

        # Create coverage session directory
        import shutil
        from datetime import datetime

        from mflux_debugger.log_paths import get_coverage_archive_dir, get_coverage_latest_dir, get_coverage_session_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = script_path.stem

        # Archive old coverage sessions for this script
        latest_dir = get_coverage_latest_dir()
        if latest_dir.exists():
            pattern = f"{script_name}_*"
            old_sessions = list(latest_dir.glob(pattern))
            if old_sessions:
                archive_dir = get_coverage_archive_dir()
                archived_count = 0
                for session_dir in old_sessions:
                    if session_dir.is_dir():
                        archive_path = archive_dir / session_dir.name
                        try:
                            if archive_path.exists():
                                # If already archived, remove the old archive
                                shutil.rmtree(archive_path)
                            shutil.move(str(session_dir), str(archive_path))
                            archived_count += 1
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  Failed to archive {session_dir.name}: {e}", file=sys.stderr)
                if archived_count > 0:
                    print(f"📦 Archived {archived_count} old coverage session(s)", file=sys.stderr)

        coverage_dir = get_coverage_session_dir(script_name, timestamp)

        print(f"📁 Creating coverage directory: {coverage_dir}", file=sys.stderr)

        # Get watch files from filtered coverage
        watch_files = set(mflux_coverage_sets.keys())

        # Generate marked-up copies of all files
        print("📝 Generating marked-up file copies...", file=sys.stderr)
        files_generated = 0
        for file_path in watch_files:
            executed_lines = mflux_coverage_sets.get(file_path, set())

            # Create relative path structure in coverage directory
            # Preserve directory structure from project root
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.is_absolute():
                    # Try to get relative path from project root
                    try:
                        rel_path = file_path_obj.relative_to(project_root)
                    except ValueError:
                        # File not under project root, use filename only
                        rel_path = Path(file_path_obj.name)
                else:
                    rel_path = Path(file_path)

                # Create output path preserving directory structure
                output_file_path = coverage_dir / rel_path
                generate_marked_up_file(file_path, executed_lines, output_file_path)
                files_generated += 1
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Failed to generate marked-up file for {file_path}: {e}", file=sys.stderr)
                continue

        print(f"✅ Generated {files_generated} marked-up file(s)", file=sys.stderr)

        # Print simple summary
        total_executed = sum(len(lines) for lines in mflux_coverage_sets.values())
        total_dead = 0
        for file_path in watch_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    total_lines = len(
                        [line for line in f.readlines() if line.strip() and not line.strip().startswith("#")]
                    )
                executed_count = len(mflux_coverage_sets.get(file_path, set()))
                total_dead += total_lines - executed_count
            except Exception:  # noqa: BLE001, PERF203
                pass

        print("\n📊 Coverage Summary:", file=sys.stderr)
        print(f"  Files analyzed: {len(watch_files)}", file=sys.stderr)
        print(f"  Lines executed: {total_executed}", file=sys.stderr)
        print(f"  Dead lines: {total_dead}", file=sys.stderr)
        print(f"\n📁 Coverage files saved to: {coverage_dir}", file=sys.stderr)
        print("   Browse marked-up files to see ✅ (executed), ❌ (dead), ⚪ (non-executable)", file=sys.stderr)

        # Terminate session
        self._api_call("POST", "/debug/terminate")

    def cmd_coverage_multi(
        self, scripts: list[str], output: Optional[str] = None, include_dirs: Optional[list[str]] = None
    ):
        """Run coverage on multiple scripts and combine results into a multi-run report.

        Args:
            scripts: List of script paths to analyze
            output: Optional output path for coverage report
            include_dirs: Optional list of additional directories to include (default: src/mflux only)
        """
        import shutil
        from datetime import datetime
        from pathlib import Path

        from mflux_debugger.coverage_report import generate_marked_up_file
        from mflux_debugger.log_paths import get_coverage_archive_dir, get_coverage_latest_dir, get_coverage_session_dir

        if len(scripts) < 2:
            print("❌ Need at least 2 scripts for multi-run coverage", file=sys.stderr)
            sys.exit(1)

        print("=" * 70, file=sys.stderr)
        print("🔍 MULTI-RUN COVERAGE ANALYSIS", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Scripts: {', '.join(scripts)}\n", file=sys.stderr)

        # Find project root
        first_script_path = Path(scripts[0]).resolve()
        project_root = first_script_path.parent
        while project_root != project_root.parent:
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent

        # Build include patterns
        include_patterns = ["src/mflux/", "src/mflux_debugger/examples/"]
        if include_dirs:
            for include_dir in include_dirs:
                normalized = include_dir.replace("\\", "/")
                if not normalized.endswith("/"):
                    normalized += "/"
                include_patterns.append(normalized)

        # Collect coverage from each script
        all_runs_coverage = []  # List of dicts: [{file: set(lines)}, {file: set(lines)}, ...]

        for script_idx, script in enumerate(scripts, 1):
            script_path = Path(script).resolve()
            if not script_path.exists():
                print(f"❌ Script {script_idx} not found: {script_path}", file=sys.stderr)
                sys.exit(1)

            print(f"\n{'=' * 70}", file=sys.stderr)
            print(f"📊 Run {script_idx}/{len(scripts)}: {script_path.name}", file=sys.stderr)
            print(f"{'=' * 70}", file=sys.stderr)

            # Clean up previous sessions
            print("🧹 Cleaning up previous debug sessions...", file=sys.stderr)
            self._cleanup_stale_sessions()

            # Ensure server is running
            if not self._ensure_server_running():
                sys.exit(1)

            # Start session with coverage enabled
            data = {"script_path": str(script_path), "coverage_mode": True}
            response = self._api_call("POST", "/debug/start", data)

            if not response.get("success"):
                print(f"❌ Failed to start session: {response.get('error', 'Unknown error')}", file=sys.stderr)
                sys.exit(1)

            # Disable checkpoint breaks
            print("⚙️  Disabling checkpoint breaks for full execution...", file=sys.stderr)
            self._api_call("POST", "/debug/checkpoint/break-all", {"enabled": False})

            # Run script to completion
            print("⏳ Running script...", file=sys.stderr)
            self._api_call("POST", "/debug/continue_async")

            # Poll until finished
            max_wait = 300
            interval = 2
            iterations = max_wait // interval

            for i in range(iterations):
                time.sleep(interval)
                status_response = self._api_call("GET", "/debug/status", timeout=2)
                state = status_response.get("data", {}).get("state")

                if (i + 1) % 5 == 0:
                    elapsed = (i + 1) * interval
                    print(f"   [{elapsed}s] Status: {state}...", file=sys.stderr)

                if state == "finished":
                    print("✅ Script completed", file=sys.stderr)
                    break
                elif state == "failed":
                    print("❌ Script execution failed", file=sys.stderr)
                    self.cmd_status(verbose=False)
                    sys.exit(1)
            else:
                print("⚠️  Script did not complete within timeout", file=sys.stderr)
                sys.exit(1)

            # Get coverage data
            print("📊 Collecting coverage data...", file=sys.stderr)
            coverage_response = self._api_call("GET", "/debug/coverage")

            if not coverage_response.get("success"):
                print(f"❌ Failed to get coverage data: {coverage_response.get('error')}", file=sys.stderr)
                sys.exit(1)

            coverage_data = coverage_response.get("data", {}).get("coverage_data", {})

            if not coverage_data:
                print("⚠️  No coverage data collected", file=sys.stderr)
                sys.exit(1)

            # Convert to sets and filter
            coverage_sets = {file: set(lines) for file, lines in coverage_data.items()}
            mflux_coverage_sets = {}

            for file_path, lines in coverage_sets.items():
                path_normalized = file_path.replace("\\", "/")

                # Skip external libraries
                if "/diffusers/" in path_normalized or "/transformers/" in path_normalized:
                    continue
                if "/mflux_debugger/" in path_normalized and "/mflux_debugger/examples/" not in path_normalized:
                    continue

                # Check if file matches include patterns
                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.is_absolute():
                        try:
                            rel_path = file_path_obj.relative_to(project_root)
                            rel_path_str = str(rel_path).replace("\\", "/")
                            matches_pattern = any(rel_path_str.startswith(p) for p in include_patterns)
                            if matches_pattern:
                                mflux_coverage_sets[file_path] = lines
                        except ValueError:
                            continue
                    else:
                        rel_path_str = file_path.replace("\\", "/")
                        matches_pattern = any(rel_path_str.startswith(p) for p in include_patterns)
                        if matches_pattern:
                            mflux_coverage_sets[file_path] = lines
                except Exception:  # noqa: BLE001
                    continue

            all_runs_coverage.append(mflux_coverage_sets)

            # Terminate session
            self._api_call("POST", "/debug/terminate")

        # Combine coverage data from all runs
        print(f"\n{'=' * 70}", file=sys.stderr)
        print("📝 Combining coverage from all runs...", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)

        # Get union of all files across all runs
        all_files = set()
        for run_coverage in all_runs_coverage:
            all_files.update(run_coverage.keys())

        # Archive old coverage sessions for this multi-run script combination
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_names = "_".join([Path(s).stem for s in scripts])
        pattern = f"multi_{script_names}_*"

        latest_dir = get_coverage_latest_dir()
        if latest_dir.exists():
            old_sessions = list(latest_dir.glob(pattern))
            if old_sessions:
                archive_dir = get_coverage_archive_dir()
                archived_count = 0
                for session_dir in old_sessions:
                    if session_dir.is_dir():
                        archive_path = archive_dir / session_dir.name
                        try:
                            if archive_path.exists():
                                # If already archived, remove the old archive
                                shutil.rmtree(archive_path)
                            shutil.move(str(session_dir), str(archive_path))
                            archived_count += 1
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  Failed to archive {session_dir.name}: {e}", file=sys.stderr)
                if archived_count > 0:
                    print(f"📦 Archived {archived_count} old multi-run coverage session(s)", file=sys.stderr)

        # Create coverage directory
        coverage_dir = get_coverage_session_dir(f"multi_{script_names}", timestamp)

        print(f"📁 Creating coverage directory: {coverage_dir}", file=sys.stderr)

        # Generate marked-up files with multi-run markers
        print("📝 Generating marked-up file copies (multi-run)...", file=sys.stderr)
        files_generated = 0
        for file_path in all_files:
            # Collect executed lines for each run
            executed_lines_per_run = [run_coverage.get(file_path, set()) for run_coverage in all_runs_coverage]

            # Create output path
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.is_absolute():
                    try:
                        rel_path = file_path_obj.relative_to(project_root)
                    except ValueError:
                        rel_path = Path(file_path_obj.name)
                else:
                    rel_path = Path(file_path)

                output_file_path = coverage_dir / rel_path
                generate_marked_up_file(file_path, executed_lines_per_run, output_file_path)
                files_generated += 1
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Failed to generate marked-up file for {file_path}: {e}", file=sys.stderr)
                continue

        print(f"✅ Generated {files_generated} marked-up file(s)", file=sys.stderr)

        # Print summary
        total_executed_per_run = [sum(len(lines) for lines in run.values()) for run in all_runs_coverage]
        print("\n📊 Multi-Run Coverage Summary:", file=sys.stderr)
        print(f"  Files analyzed: {len(all_files)}", file=sys.stderr)
        for i, total in enumerate(total_executed_per_run, 1):
            print(f"  Run {i} lines executed: {total}", file=sys.stderr)
        print(f"\n📁 Coverage files saved to: {coverage_dir}", file=sys.stderr)
        print("   Each line shows markers for each run: ✅ (executed), ❌ (dead), ⚪ (non-executable)", file=sys.stderr)

    def cmd_tutorial(self, lesson: str = "basic"):
        """Provide interactive guidance for learning debugger usage - agent executes commands."""
        from pathlib import Path

        # Find tutorial script
        tutorial_dir = Path(__file__).parent / "examples"
        tutorial_script = tutorial_dir / f"tutorial_{lesson}_{self.framework}.py"

        if not tutorial_script.exists():
            print(f"❌ Tutorial '{lesson}' not found at: {tutorial_script}", file=sys.stderr)
            print(f"\n💡 Available tutorials for {self.framework}: basic, ab", file=sys.stderr)
            return

        # Define tutorial steps
        if lesson == "basic":
            self._show_basic_tutorial_guide(tutorial_script)
        elif lesson == "ab":
            self._show_ab_tutorial_guide(tutorial_script)
        else:
            print(f"❌ Unknown tutorial: {lesson}", file=sys.stderr)

    def _show_basic_tutorial_guide(self, script_path: Path):
        """Show step-by-step guide for basic breakpoints tutorial - agent follows along."""
        print("🎓 Interactive Debugger Tutorial - Basic Breakpoints")
        print("=" * 70)

        if self.framework == "pytorch":
            print("\n📚 What you'll learn (PyTorch):")
            print("  1. Starting a debugging session")
            print("  2. Using debug_checkpoint() for semantic breakpoints")
            print("  3. Checking debugger status and location")
            print("  4. Viewing script output during debugging")
            print("  5. Listing all variables in scope")
            print("  6. Inspecting variables at breakpoints")
            print("  7. Conditional breakpoints in loops (skip on condition)")
            print("  8. Using debug_save() to save PyTorch tensors")
            print("  9. Preparing tensors for MLX comparison")
            print("  10. Using coverage analysis to find dead code paths")
            print("\n✅ RECOMMENDED: Complete this PyTorch tutorial FIRST")
            print("   Then run 'mflux-debug-mlx tutorial' to compare the saved tensors in MLX.")
            print("\n🎯 Tutorial Script Location:")
            print(f"  {script_path}")
            print("\n📝 What the script does:")
            print("  • Creates a 2x3 input tensor in PyTorch")
            print("  • Scales it by 2")
            print("  • Loops 5 times with conditional skip (pauses only on iteration 3)")
            print("  • Computes sum along dim 1")
            print("  • Calls debug_checkpoint() with metadata (semantic breakpoints)")
            print("  • Demonstrates skip=True and conditional skip=expression")
            print("  • Calls debug_save() at each step to save tensors")
            print("  • Saved tensors can be loaded in MLX for comparison")
        else:  # mlx
            print("\n📚 What you'll learn (MLX):")
            print("  1. Starting a debugging session")
            print("  2. Using debug_checkpoint() for semantic breakpoints")
            print("  3. Adding metadata to checkpoints for better context")
            print("  4. Skipping checkpoints with skip=True")
            print("  5. Conditional breakpoints in loops (skip on condition)")
            print("  6. Inspecting variables at breakpoints")
            print("  7. Using debug_load() to compare with PyTorch tensors")
            print("  8. Using coverage analysis to find dead code paths")
            print("\n💡 NOTE: For the full cross-framework comparison experience:")
            print("   Run 'mflux-debug-pytorch tutorial' FIRST (if you haven't already)")
            print("   to save PyTorch tensors. This tutorial can then load and compare them.")
            print("   However, this tutorial works standalone too - just skip debug_load() steps.")
            print("\n🎯 Tutorial Script Location:")
            print(f"  {script_path}")
            print("\n📝 What the script does:")
            print("  • Creates a 2x3 input tensor in MLX")
            print("  • Scales it by 2")
            print("  • Loops 5 times with conditional skip (pauses only on iteration 3)")
            print("  • Computes sum along axis")
            print("  • Calls debug_checkpoint() with metadata (semantic breakpoints)")
            print("  • Demonstrates skip=True and conditional skip=expression")
            print("  • Optionally loads PyTorch tensors for comparison (if available)")
        print("\n" + "=" * 70)

    def _show_ab_tutorial_guide(self, script_path: Path):
        """Show step-by-step guide for A/B attention debugging tutorial."""
        print("🎓 Interactive Debugger Tutorial - A/B Attention Comparison")
        print("=" * 70)

        if self.framework == "pytorch":
            print("\n📚 What you'll learn (PyTorch, A/B mode):")
            print("  1. Using the dedicated A/B helpers: debug_checkpoint_pytorch_A/B")
            print("  2. Bracketing the attention core with a single A and B checkpoint")
            print("  3. Inspecting Q/K/V tensors at A")
            print("  4. Inspecting attention outputs at B")
            print("  5. Coordinating with the MLX A/B tutorial for cross-framework comparison")
            print("\n🎯 Tutorial Script Location (PyTorch A/B):")
            print(f"  {script_path}")
            print("\n📝 What the script does:")
            print("  • Builds tiny deterministic Q/K/V tensors with shape [B, S, H, D]")
            print("  • Calls debug_checkpoint_pytorch_A(...) just before the attention core")
            print("  • Runs a small scaled dot-product attention core")
            print("  • Calls debug_checkpoint_pytorch_B(...) right after the attention core")
            print("\n🚀 How to run this tutorial:")
            print("  1. Start the PyTorch session:")
            print(f"     mflux-debug-pytorch start {script_path}")
            print("  2. Set checkpoint breakpoints on A and B:")
            print("     mflux-debug-pytorch checkpoint-break pytorch_A")
            print("     mflux-debug-pytorch checkpoint-break pytorch_B")
            print("  3. Run until A, inspect Q/K/V:")
            print("     mflux-debug-pytorch continue")
            print('     mflux-debug-pytorch vars   # or eval "query.shape"')
            print("  4. Continue to B and inspect attention output:")
            print("     mflux-debug-pytorch continue")
            print('     mflux-debug-pytorch eval "attn_output.shape"')
            print("\n💡 Next step:")
            print("  Run the MLX A/B tutorial to compare checkpoints across frameworks:")
            print("     mflux-debug-mlx tutorial ab")
        else:  # mlx
            print("\n📚 What you'll learn (MLX, A/B mode):")
            print("  1. Using the dedicated A/B helpers: debug_checkpoint_mlx_A/B")
            print("  2. Matching the PyTorch attention core with MLX math")
            print("  3. Verifying that Q/K/V match at A")
            print("  4. Comparing attention outputs at B")
            print("\n🎯 Tutorial Script Location (MLX A/B):")
            print(f"  {script_path}")
            print("\n📝 What the script does:")
            print("  • Builds the same deterministic Q/K/V tensors as the PyTorch tutorial")
            print("  • Calls debug_checkpoint_mlx_A(...) before the attention core")
            print("  • Runs the equivalent scaled dot-product attention core in MLX")
            print("  • Calls debug_checkpoint_mlx_B(...) after the attention core")
            print("\n🚀 How to run this tutorial:")
            print("  1. Start the MLX session:")
            print(f"     mflux-debug-mlx start {script_path}")
            print("  2. Set checkpoint breakpoints on A and B:")
            print("     mflux-debug-mlx checkpoint-break mlx_A")
            print("     mflux-debug-mlx checkpoint-break mlx_B")
            print("  3. Run until A, inspect Q/K/V:")
            print("     mflux-debug-mlx continue")
            print('     mflux-debug-mlx vars   # or eval "query.shape"')
            print("  4. Continue to B and inspect attention output:")
            print("     mflux-debug-mlx continue")
            print('     mflux-debug-mlx eval "attn_output.shape"')
            print("\n💡 Recommended workflow:")
            print("  1. Run the PyTorch A/B tutorial first:")
            print("       mflux-debug-pytorch tutorial ab")
            print("  2. Then run this MLX A/B tutorial and compare checkpoints A/B")
        print("\n" + "=" * 70)
        print("\n💡 BEFORE YOU START:")
        print(f"   Run 'mflux-debug-{self.framework} --help' to see all available commands.")
        print("   This tutorial will teach you the most important ones interactively.\n")
        print("=" * 70)
        print("\n📋 STEP-BY-STEP INSTRUCTIONS:")
        print("   Execute each command below and observe the output.\n")

        if self.framework == "pytorch":
            self._show_pytorch_tutorial_steps(script_path)
        else:
            self._show_mlx_tutorial_steps(script_path)

    def _show_pytorch_tutorial_steps(self, script_path: Path):
        """Show PyTorch-specific tutorial steps."""
        print("┌─ STEP 1: Start the debugging session")
        print(f"│  Command: mflux-debug-{self.framework} start {script_path}")
        print("│  Purpose: Load the script and prepare for debugging")
        print("│  Expected: Server starts, session initialized")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 2: Run and hit semantic breakpoint (debug_checkpoint)")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Execute until first debug_checkpoint('after_input_creation')")
        print("│  Expected: Automatically pauses at semantic breakpoint")
        print("│  Observe: Checkpoint name, hit count, location (file:line), variable preview")
        print("│  ✨ NEW: Checkpoint values are automatically displayed!")
        print("│         Shows tensor shapes, sample values (first 10), and statistics (mean/std)")
        print("│  Note: No line numbers needed - debug_checkpoint() is in the code!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 2b: Check debugger status and location")
        print(f"│  Command: mflux-debug-{self.framework} status")
        print("│  Purpose: See current debugger state (paused/running/finished)")
        print(f"│  Then: mflux-debug-{self.framework} location")
        print("│  Purpose: Get detailed location info (file, line, function, checkpoint)")
        print("│  Expected: Shows 'paused' state and checkpoint details")
        print("│  ✨ NEW: Automatically displays checkpoint values:")
        print("│         • Tensor shapes (e.g., [2 × 3])")
        print("│         • Sample values (first 10, or first/last 10 for longer lists)")
        print("│         • Statistics (mean, std, min, max)")
        print("│         • Integers/booleans (loop indexes, flags)")
        print("│  Note: All checkpoint variables captured by debug_checkpoint() are shown!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 2c: View script output (optional)")
        print(f"│  Command: mflux-debug-{self.framework} log --lines 20")
        print("│  Purpose: See the script's print statements and checkpoint messages")
        print("│  Expected: Shows script output including checkpoint hit messages")
        print("│  Note: This shows what the script printed up to the current breakpoint")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 3: Inspect the input tensor")
        print(f'│  Command: mflux-debug-{self.framework} eval "x"')
        print("│  Purpose: Get detailed tensor information with statistics")
        print("│  Expected: Shape (2, 3), dtype, sample values, min/max/mean")
        print("│  Note: debug_save() already saved this tensor as 'input_tensor' for MLX!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 3b: Verify tensor was saved")
        print(f"│  Command: mflux-debug-{self.framework} tensors-list")
        print("│  Purpose: Confirm debug_save() stored the tensor")
        print("│  Expected: Shows 'input_tensor' in the list")
        print("│  Note: This tensor can now be loaded in MLX with debug_load('input_tensor')")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 3c: List all variables in current scope")
        print(f"│  Command: mflux-debug-{self.framework} vars")
        print("│  Purpose: See all variables available at the current breakpoint")
        print("│  Expected: Shows 'x' (and other variables if any)")
        print("│  Note: Useful to discover what variables are available for inspection!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 4: Continue to loop checkpoint")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Run to next checkpoint (skips 'after_scaling', hits 'loop_iteration')")
        print("│  Expected: Pauses at 'loop_iteration' checkpoint (iteration 3)")
        print("│  Observe: Notice how iterations 0-2 were skipped (logged but not paused)")
        print("│  Note: Loop iterations 0-2 and 4 are skipped via conditional skip=(i != 3)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 4b: Inspect variables at loop checkpoint")
        print(f'│  Command: mflux-debug-{self.framework} eval "i"')
        print("│  Purpose: Check the loop iteration number")
        print("│  Expected: Returns 3 (the iteration that triggered the breakpoint)")
        print(f'│  Then: mflux-debug-{self.framework} eval "temp"')
        print("│  Purpose: Inspect the computed tensor at this iteration")
        print("│  Expected: Shape (2, 3), values scaled by (i+1) = 4")
        print("│  Note: This demonstrates inspecting variables at intermediate checkpoints!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 5: Continue to final checkpoint")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Run to final checkpoint ('after_sum')")
        print("│  Expected: Pauses at 'after_sum' checkpoint")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 6: Inspect final result")
        print(f'│  Command: mflux-debug-{self.framework} eval "result"')
        print("│  Purpose: Check the computed sum")
        print("│  Expected: Returns tensor([12., 30.])")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 7: Check all saved tensors for MLX comparison")
        print(f"│  Command: mflux-debug-{self.framework} tensors-list")
        print("│  Purpose: See all tensors saved by debug_save() throughout execution")
        print("│  Expected: 3 tensors (input_tensor, scaled_tensor, sum_result)")
        print("│  Note: These are ready to be loaded in MLX with debug_load()!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 8: View tensor details")
        print(f"│  Command: mflux-debug-{self.framework} tensors-info sum_result")
        print("│  Purpose: Get metadata about the saved tensor")
        print("│  Expected: Shape (2,), dtype float32, size info")
        print("│  Note: MLX will use this exact file with debug_load('sum_result')")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 9: Continue to completion")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Let the script finish")
        print("│  Expected: Script completes, session becomes idle")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 10: Terminate the session")
        print(f"│  Command: mflux-debug-{self.framework} terminate")
        print("│  Purpose: Clean up and end the debugging session")
        print("│  Expected: Session terminated")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 11: Check the script output log")
        print("│  Command: cat mflux_debugger/logs/runs/latest/tutorial_basic_pytorch_*/script_output.log")
        print("│  Purpose: See the script's stdout/stderr with checkpoint references")
        print("│  Expected: Script output with checkpoint hits and JSON file paths")
        print("│  Note: This log persists after termination - includes all debug_checkpoint() hits!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 12: Check automatic checkpoint logs")
        print("│  Command: ls -lh mflux_debugger/logs/runs/latest/tutorial_basic_pytorch_*/checkpoints/")
        print("│  Purpose: See the JSON files automatically created by debug_checkpoint()")
        print("│  Expected: JSON files for each checkpoint with full state captured")
        print("│  Note: These persist after termination - perfect for offline analysis!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 13: Inspect a checkpoint file")
        print(
            "│  Command: cat mflux_debugger/logs/runs/latest/tutorial_basic_pytorch_*/checkpoints/checkpoint_after_input_creation*.json | jq '.variables'"
        )
        print("│  Purpose: See what debug_checkpoint() captured automatically")
        print("│  Expected: Auto-captured variables (x, scaled, temp, etc.)")
        print("│  Note: Variables are auto-captured from local scope when not explicitly passed!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 14: Run coverage analysis (NEW!)")
        print(f"│  Command: mflux-debug-{self.framework} coverage {script_path}")
        print("│  Purpose: Analyze code coverage to find dead code paths")
        print("│  Expected: Script runs to completion, generates marked-up files")
        print("│  Note: Coverage mode runs without pausing at checkpoints")
        print("│  Note: Default analyzes src/mflux/ code only")
        print("│  Note: Debugger implementation code is always excluded, but examples are allowed")
        print("│  Output: Coverage folder with marked-up files")
        print("│  ✨ NEW: Creates marked-up copies of all files showing:")
        print("│         ✅ (green) = line was executed")
        print("│         ❌ (red) = line exists but wasn't executed (dead code)")
        print("│         ⚪ (white) = line is not executable (blank, comment, etc.)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 14b: View marked-up coverage files")
        print("│  Command: ls -la mflux_debugger/coverage/latest/tutorial_basic_pytorch_*/")
        print("│  Purpose: See the coverage directory structure")
        print("│  Expected: Shows coverage folder with src/ subdirectory")
        print("│  Then: find mflux_debugger/coverage/latest/tutorial_basic_pytorch_*/ -type f | head -5")
        print("│  Purpose: List all generated marked-up files")
        print("│  Expected: Shows Python files from src/mflux/ that were executed")
        print("│  Then: head -30 mflux_debugger/coverage/latest/tutorial_basic_pytorch_*/src/mflux/...")
        print("│  Purpose: View a marked-up file showing execution status")
        print("│  Expected: Shows file with ✅/❌/⚪ markers for each line")
        print("│  Note: All files are full copies - easy to browse and see what was executed!")
        print("│  Note: Files preserve the same directory structure as your source code")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 14c: Multi-run coverage (compare multiple scripts)")
        print(f"│  Command: mflux-debug-{self.framework} coverage-multi script1.py script2.py")
        print("│  Purpose: Run coverage on multiple scripts and compare side-by-side")
        print("│  Expected: Each script runs sequentially, then combined report is generated")
        print("│  Output: Coverage folder with multi-run marked-up files")
        print("│  ✨ NEW: Each line shows markers for EACH run:")
        print("│         ✅ ✅ = executed in both runs")
        print("│         ✅ ❌ = executed in run 1 only")
        print("│         ❌ ✅ = executed in run 2 only")
        print("│         ❌ ❌ = dead code (not executed in any run)")
        print("│         ⚪ ⚪ = non-executable (blank, comment, etc.)")
        print("│  Example: Compare txt2img vs edit model to find code paths used by each")
        print("│  Example: mflux-debug-pytorch coverage-multi debug_txt2img.py debug_edit.py")
        print("│  Note: Perfect for finding dead code that's truly unused across all workflows")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 15: Use cleanup command")
        print("│  Command: mflux-debug-clean --dry-run")
        print("│  Purpose: Preview what would be cleaned")
        print("│  Expected: Shows size of mflux_debugger/ directory and what would be removed")
        print("│  Then try: mflux-debug-clean --yes")
        print("│  Note: Removes everything in mflux_debugger/ - use --dry-run first!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 16: Clean up all debugger processes (FINAL)")
        print("│  Command: mflux-debug-kill-all --dry-run")
        print("│  Purpose: Preview all running debugger processes")
        print("│  Expected: Shows PyTorch/MLX debugger processes")
        print("│  Then: mflux-debug-kill-all")
        print("│  Note: This kills ALL debugger servers - get a clean slate!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("=" * 70)
        print("✨ After completing these steps, you'll understand:")
        print("  ✓ How to start/stop debugging sessions")
        print("  ✓ How to use debug_checkpoint() for semantic breakpoints")
        print("  ✓ How checkpoint values are automatically displayed (shapes, samples, statistics)")
        print("  ✓ How to check debugger status and location")
        print("  ✓ How to view script output during debugging")
        print("  ✓ How to list all variables in scope with 'vars' command")
        print("  ✓ How debug_checkpoint() auto-captures local variables")
        print("  ✓ How debug_checkpoint() ALWAYS logs to JSON (even without debugger!)")
        print("  ✓ How to inspect variables at intermediate checkpoints")
        print("  ✓ How to use debug_save() to save PyTorch tensors")
        print("  ✓ How to inspect variables and saved tensors")
        print("  ✓ How to read script output logs with checkpoint references")
        print("  ✓ How to analyze checkpoint logs offline with jq")
        print("  ✓ How automatic tensor archiving works on session start")
        print("  ✓ How to use cleanup commands to manage debug artifacts")
        print("  ✓ How to prepare tensors for MLX comparison")
        print("  ✓ How to use coverage analysis to find dead code paths")
        print("  ✓ How to browse marked-up coverage files showing executed vs dead code")
        print("  ✓ How to use multi-run coverage to compare multiple scripts side-by-side")
        print("\n💡 Pro Tips:")
        print("  • ✨ NEW: Checkpoint values are automatically displayed when paused!")
        print("           Shows tensor shapes, sample values (first 10), and statistics")
        print("  • Use 'status' and 'location' commands to check where you are during debugging")
        print("  • Use 'vars' command to discover available variables at breakpoints")
        print("  • Use 'log' command to see script output during execution")
        print("  • debug_checkpoint() auto-captures tensors/arrays from local scope!")
        print("  • debug_checkpoint() ALWAYS logs - run scripts normally to collect state!")
        print("  • Check script_output.log for a full record of execution with checkpoint hits")
        print("  • Use jq to analyze checkpoint JSON files")
        print("  • Checkpoint logs persist after sessions - great for CI/CD debugging")
        print("  • Tensors are automatically archived when starting a new session")
        print("  • Use --dry-run with cleanup commands to preview what will be deleted")
        print("  • Use debug_checkpoint() with metadata to hint which variables to capture")
        print("  • Use debug_save() at key computation steps")
        print("  • Use descriptive names for tensors (e.g., 'hidden_states_block_0')")
        print("  • ✨ NEW: Use 'coverage' command to find dead code paths in your codebase!")
        print("           Coverage creates marked-up file copies showing ✅/❌/⚪ for each line")
        print("           Browse files in mflux_debugger/coverage/latest/ to see what was executed")
        print("  • ✨ NEW: Use 'coverage-multi' to compare multiple scripts side-by-side!")
        print("           Shows which lines are executed in which runs: ✅ ✅ (both), ✅ ❌ (run 1), ❌ ✅ (run 2)")
        print("\n🚀 Next Step:")
        print("  ✅ Run 'mflux-debug-mlx tutorial' NEXT to load and compare these PyTorch tensors in MLX!")
        print("     The MLX tutorial will use debug_load() to compare implementations.")
        print("\n💡 Cleanup Reminder:")
        print("  • Use 'mflux-debug-kill-all' to kill all debugger servers when done")
        print("  • Use 'mflux-debug-clean --yes' to clean up all debug artifacts")
        print("=" * 70)
        print()

    def _show_mlx_tutorial_steps(self, script_path: Path):
        """Show MLX-specific tutorial steps."""
        print("┌─ STEP 1: Start the debugging session")
        print(f"│  Command: mflux-debug-{self.framework} start {script_path}")
        print("│  Purpose: Load the script and prepare for debugging")
        print("│  Expected: Server starts, session initialized")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 2: Run and hit semantic breakpoint (debug_checkpoint)")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Execute until first debug_checkpoint('after_input_creation')")
        print("│  Expected: Automatically pauses at semantic breakpoint")
        print("│  ✨ NEW: Checkpoint values are automatically displayed!")
        print("│         Shows tensor shapes, sample values (first 10), and statistics")
        print("│  Note: No line numbers needed - debug_checkpoint() is in the code!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 3: Check current location")
        print(f"│  Command: mflux-debug-{self.framework} location")
        print("│  Purpose: Verify where execution is paused")
        print("│  Expected: Shows checkpoint 'after_input_creation' with metadata")
        print("│  ✨ NEW: Automatically displays checkpoint values:")
        print("│         • Tensor shapes (e.g., [2 × 3])")
        print("│         • Sample values (first 10, or first/last 10 for longer lists)")
        print("│         • Statistics (mean, std, min, max)")
        print("│         • Integers/booleans (loop indexes, flags)")
        print("│  Note: All checkpoint variables captured by debug_checkpoint() are shown!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 4: List available variables")
        print(f"│  Command: mflux-debug-{self.framework} vars")
        print("│  Purpose: See what variables exist at this point")
        print("│  Expected: Shows 'x' (the input tensor)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 5: Evaluate an expression")
        print(f'│  Command: mflux-debug-{self.framework} eval "x.shape"')
        print("│  Purpose: Check the shape of tensor x")
        print("│  Expected: Returns (2, 3)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 6: Inspect tensor details")
        print(f'│  Command: mflux-debug-{self.framework} eval "x"')
        print("│  Purpose: Get detailed tensor information")
        print("│  Expected: Shape, dtype, sample values")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 7: Load and compare PyTorch tensor (if available)")
        print(f'│  Command: mflux-debug-{self.framework} eval "pytorch_x"')
        print("│  Purpose: Check if PyTorch tensor was loaded from previous tutorial")
        print("│  Expected: Shows PyTorch tensor array([1,2,3],[4,5,6]) or None")
        print("│  Note: Will be None if you haven't run pytorch tutorial first")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 8: Verify cross-framework equivalence")
        print(
            f'│  Command: mflux-debug-{self.framework} eval "mx.allclose(x, pytorch_x, atol=1e-5) if pytorch_x is not None else None"'
        )
        print("│  Purpose: Verify MLX and PyTorch tensors match")
        print("│  Expected: Returns True if tensors loaded, None otherwise")
        print("│  Note: This is the KEY cross-framework verification step!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 9: Continue to loop checkpoint")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Run to next checkpoint (skips 'after_scaling', hits 'loop_iteration')")
        print("│  Expected: Pauses at 'loop_iteration' checkpoint (iteration 3)")
        print("│  Note: Loop iterations 0-2 and 4 are skipped via conditional skip=(i != 3)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 10: Continue to final checkpoint")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Run to final checkpoint ('after_sum')")
        print("│  Expected: Pauses at 'after_sum' checkpoint")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 11: Inspect final result")
        print(f'│  Command: mflux-debug-{self.framework} eval "result"')
        print("│  Purpose: Check the computed sum")
        print("│  Expected: Returns sum along axis 1: array([12., 30.])")
        print("│  Note: Values are scaled by 2, so [1,2,3]*2=[2,4,6]→12, [4,5,6]*2=[8,10,12]→30")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 12: Compare final results with PyTorch")
        print(f'│  Command: mflux-debug-{self.framework} eval "pytorch_result"')
        print("│  Purpose: View the PyTorch result tensor (if loaded)")
        print("│  Expected: Shows tensor([12., 30.]) or None")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 13: Verify final result equivalence")
        print(
            f'│  Command: mflux-debug-{self.framework} eval "mx.allclose(result, pytorch_result, atol=1e-5) if pytorch_result is not None else None"'
        )
        print("│  Purpose: Verify MLX and PyTorch final results match")
        print("│  Expected: Returns True - proves cross-framework equivalence!")
        print("│  Note: This confirms your MLX implementation produces identical results!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 14: Continue to completion")
        print(f"│  Command: mflux-debug-{self.framework} continue")
        print("│  Purpose: Let the script finish")
        print("│  Expected: Script completes, session becomes idle")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 15: Terminate the session")
        print(f"│  Command: mflux-debug-{self.framework} terminate")
        print("│  Purpose: Clean up and end the debugging session")
        print("│  Expected: Session terminated")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 16: Check the script output log")
        print("│  Command: ls mflux_debugger/logs/runs/latest/")
        print("│  Purpose: Find the timestamped directory for your run")
        print("│  Expected: Shows directories like tutorial_basic_mlx_YYYYMMDD_HHMMSS")
        print("│  Then: cat mflux_debugger/logs/runs/latest/tutorial_basic_mlx_*/script_output.log")
        print("│  Purpose: See the script's stdout/stderr with checkpoint references")
        print("│  Expected: Script output with ALL checkpoint hits (including skipped ones!) and JSON paths")
        print("│  Note: skip=True checkpoints are logged here with their JSON file paths!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 17: Check automatic checkpoint logs")
        print("│  Command: ls -lh mflux_debugger/logs/runs/latest/tutorial_basic_mlx_*/checkpoints/")
        print("│  Purpose: See the JSON files automatically created by debug_checkpoint()")
        print("│  Expected: JSON files for each checkpoint (INCLUDING skipped ones!)")
        print("│  Note: skip=True means 'log but don't pause' - JSON is still saved!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 18: Compare checkpoint logs with PyTorch")
        print(
            "│  Command: cat mflux_debugger/logs/runs/latest/tutorial_basic_mlx_*/checkpoints/checkpoint_after_sum*.json | jq '.'"
        )
        print("│  Purpose: See MLX checkpoint data captured automatically")
        print(
            "│  Then: cat mflux_debugger/logs/runs/latest/tutorial_basic_pytorch_*/checkpoints/checkpoint_after_sum*.json | jq '.'"
        )
        print("│  Expected: Both show the same metadata and structure - easy offline comparison!")
        print("│  Note: No need to rerun scripts - checkpoint logs persist!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 19: Run coverage analysis (NEW!)")
        print(f"│  Command: mflux-debug-{self.framework} coverage {script_path}")
        print("│  Purpose: Analyze code coverage to find dead code paths")
        print("│  Expected: Script runs to completion, generates marked-up files")
        print("│  Note: Coverage mode runs without pausing at checkpoints")
        print("│  Note: Default analyzes src/mflux/ code only")
        print("│  Note: Debugger implementation code is always excluded, but examples are allowed")
        print("│  Output: Coverage folder with marked-up files")
        print("│  ✨ NEW: Creates marked-up copies of all files showing:")
        print("│         ✅ (green) = line was executed")
        print("│         ❌ (red) = line exists but wasn't executed (dead code)")
        print("│         ⚪ (white) = line is not executable (blank, comment, etc.)")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 19b: View marked-up coverage files")
        print("│  Command: ls -la mflux_debugger/coverage/latest/tutorial_basic_mlx_*/")
        print("│  Purpose: See the coverage directory structure")
        print("│  Expected: Shows coverage folder with src/ subdirectory")
        print("│  Then: find mflux_debugger/coverage/latest/tutorial_basic_mlx_*/ -type f | head -5")
        print("│  Purpose: List all generated marked-up files")
        print("│  Expected: Shows Python files from src/mflux/ that were executed")
        print("│  Then: head -30 mflux_debugger/coverage/latest/tutorial_basic_mlx_*/src/mflux/...")
        print("│  Purpose: View a marked-up file showing execution status")
        print("│  Expected: Shows file with ✅/❌/⚪ markers for each line")
        print("│  Note: All files are full copies - easy to browse and see what was executed!")
        print("│  Note: Files preserve the same directory structure as your source code")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 19c: Multi-run coverage (compare multiple scripts)")
        print(f"│  Command: mflux-debug-{self.framework} coverage-multi script1.py script2.py")
        print("│  Purpose: Run coverage on multiple scripts and compare side-by-side")
        print("│  Expected: Each script runs sequentially, then combined report is generated")
        print("│  Output: Coverage folder with multi-run marked-up files")
        print("│  ✨ NEW: Each line shows markers for EACH run:")
        print("│         ✅ ✅ = executed in both runs")
        print("│         ✅ ❌ = executed in run 1 only")
        print("│         ❌ ✅ = executed in run 2 only")
        print("│         ❌ ❌ = dead code (not executed in any run)")
        print("│         ⚪ ⚪ = non-executable (blank, comment, etc.)")
        print("│  Example: Compare txt2img vs edit model to find code paths used by each")
        print("│  Example: mflux-debug-mlx coverage-multi debug_txt2img.py debug_edit.py")
        print("│  Note: Perfect for finding dead code that's truly unused across all workflows")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 20: Use cleanup command")
        print("│  Command: mflux-debug-clean --dry-run")
        print("│  Purpose: Preview what would be cleaned")
        print("│  Expected: Shows size of mflux_debugger/ directory and what would be removed")
        print("│  Then try: mflux-debug-clean --yes")
        print("│  Note: Removes everything in mflux_debugger/ - use --dry-run first!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("┌─ STEP 21: Clean up all debugger processes (FINAL)")
        print("│  Command: mflux-debug-kill-all --dry-run")
        print("│  Purpose: Preview all running debugger processes")
        print("│  Expected: Shows PyTorch/MLX debugger processes")
        print("│  Then: mflux-debug-kill-all")
        print("│  Note: This kills ALL debugger servers - get a clean slate!")
        print("└─────────────────────────────────────────────────────────────────\n")

        print("=" * 70)
        print("✨ After completing these steps, you'll understand:")
        print("  ✓ How to start/stop debugging sessions")
        print("  ✓ How to use debug_checkpoint() for semantic breakpoints")
        print("  ✓ How checkpoint values are automatically displayed (shapes, samples, statistics)")
        print("  ✓ How debug_checkpoint() ALWAYS logs to JSON (even without debugger!)")
        print("  ✓ How skip=True means 'log but don't pause'")
        print("  ✓ How to add metadata to checkpoints for better context")
        print("  ✓ How to skip checkpoints with skip=True")
        print("  ✓ How to inspect variables and evaluate expressions")
        print("  ✓ How to use debug_load() to compare with PyTorch tensors")
        print("  ✓ How to read script output logs with checkpoint references")
        print("  ✓ How to analyze checkpoint logs offline with jq")
        print("  ✓ How to compare MLX vs PyTorch using checkpoint JSON files")
        print("  ✓ How to use cleanup commands to manage debug artifacts")
        print("  ✓ How to use coverage analysis to find dead code paths")
        print("  ✓ How to browse marked-up coverage files showing executed vs dead code")
        print("  ✓ How to use multi-run coverage to compare multiple scripts side-by-side")
        print("\n💡 Pro Tips:")
        print("  • ✨ NEW: Checkpoint values are automatically displayed when paused!")
        print("           Shows tensor shapes, sample values (first 10), and statistics")
        print("  • debug_checkpoint() ALWAYS logs - run scripts without debugger to collect state!")
        print("  • skip=True logs to JSON but doesn't pause (perfect for loops)")
        print("  • Check script_output.log for a full record of execution with checkpoint hits")
        print("  • Use jq to compare checkpoint JSON files between MLX and PyTorch")
        print("  • Checkpoint logs persist after sessions - great for CI/CD debugging")
        print("  • Use --dry-run with cleanup commands to preview what will be deleted")
        print("  • Use debug_checkpoint() with metadata - more maintainable than line numbers!")
        print("  • ✨ NEW: Use 'coverage' command to find dead code paths in your codebase!")
        print("           Coverage creates marked-up file copies showing ✅/❌/⚪ for each line")
        print("           Browse files in mflux_debugger/coverage/latest/ to see what was executed")
        print("  • ✨ NEW: Use 'coverage-multi' to compare multiple scripts side-by-side!")
        print("           Shows which lines are executed in which runs: ✅ ✅ (both), ✅ ❌ (run 1), ❌ ✅ (run 2)")
        print("\n🚀 What's Next:")
        print("  • You've completed the MLX tutorial! 🎉")
        print("  • If you ran 'mflux-debug-pytorch tutorial' first, you practiced cross-framework comparison")
        print("  • Try line-based breakpoints: mflux-debug-mlx break <file> <line>")
        print('  • Try conditional breakpoints: --condition "x > 10"')
        print("  • Use coverage analysis to find dead code in your models")
        print("  • Apply these techniques to your own MFLUX model porting work!")
        print("\n💡 Cleanup Reminder:")
        print("  • Use 'mflux-debug-kill-all' to kill all debugger servers when done")
        print("  • Use 'mflux-debug-clean --yes' to clean up all debug artifacts")
        print("=" * 70)
        print()


def create_parser(framework: str) -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog=f"mflux-debug-{framework}",
        description=f"Debug {framework.upper()} code with breakpoints and inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # 🎯 RECOMMENDED: Code-first debugging with debug_checkpoint()
  # Place debug_checkpoint() calls directly in code, then:
  mflux-debug-{framework} start src/my_script.py
  mflux-debug-{framework} continue
  # Script automatically pauses at debug_checkpoint() locations
  mflux-debug-{framework} eval "prompt_embeds.shape"
  mflux-debug-{framework} eval "prompt_embeds"
  mflux-debug-{framework} continue
  mflux-debug-{framework} terminate

  # Traditional line-based debugging (also supported)
  mflux-debug-{framework} start src/my_script.py
  mflux-debug-{framework} break /path/to/file.py 123
  mflux-debug-{framework} continue
  mflux-debug-{framework} terminate

  # Debug save/load utilities
  mflux-debug-{framework} tensors-list
  mflux-debug-{framework} tensors-info tensor_name
  mflux-debug-{framework} tensors-clear

  # Server management
  mflux-debug-{framework} status
  mflux-debug-{framework} terminate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start debugging session")
    start_parser.add_argument("script", help="Path to script to debug")
    start_parser.add_argument("--no-reset", action="store_true", help="Don't reset previous sessions (not recommended)")
    start_parser.add_argument(
        "--keep-tensors", action="store_true", help="Keep saved tensors (disable default clearing for PyTorch)"
    )

    # Breakpoint commands
    break_parser = subparsers.add_parser("break", help="Set breakpoint")
    break_parser.add_argument("file", help="File path")
    break_parser.add_argument("line", type=int, help="Line number")
    break_parser.add_argument("--condition", help="Conditional breakpoint expression")

    # Conditional checkpoint breakpoint
    checkpoint_break_parser = subparsers.add_parser(
        "checkpoint-break", help="Set conditional breakpoint on semantic checkpoint"
    )
    checkpoint_break_parser.add_argument(
        "checkpoint_name",
        nargs="?",  # Make optional when using --all or --pattern
        help="Name of checkpoint (e.g., 'sdpa_05_output')",
    )
    checkpoint_break_parser.add_argument(
        "--context", help="Context condition as key=value pairs (e.g., 'block=0,timestep=0,hit_count=1')"
    )
    checkpoint_break_parser.add_argument(
        "--all", action="store_true", dest="all_checkpoints", help="Set breakpoints on all discovered checkpoints"
    )
    checkpoint_break_parser.add_argument(
        "--pattern", help="Glob pattern to match checkpoint names (e.g., 'rope_*', '*mlx_*')"
    )
    checkpoint_break_parser.add_argument(
        "--dir", type=Path, dest="script_dir", help="Directory to scan for checkpoints (default: current directory)"
    )

    # Checkpoint list command
    checkpoint_list_parser = subparsers.add_parser("checkpoint-list", help="List all checkpoints found in codebase")
    checkpoint_list_parser.add_argument(
        "--dir", type=Path, dest="script_dir", help="Directory to scan for checkpoints (default: current directory)"
    )

    remove_parser = subparsers.add_parser("remove", help="Remove breakpoint")
    remove_parser.add_argument("file", help="File path")
    remove_parser.add_argument("line", type=int, help="Line number")

    subparsers.add_parser("breaks", help="List all breakpoints")

    # Execution commands
    continue_parser = subparsers.add_parser("continue", help="Continue execution (polls automatically)")
    continue_parser.add_argument("--max-wait", type=int, default=120, help="Max seconds to poll (default: 120)")

    # Inspection commands
    eval_parser = subparsers.add_parser("eval", help="Evaluate expression")
    eval_parser.add_argument("expression", help="Python expression to evaluate")

    subparsers.add_parser("vars", help="List variables")

    # Status commands
    status_parser = subparsers.add_parser("status", help="Check debugger status")
    status_parser.add_argument("--verbose", action="store_true", help="Show full traceback for errors")

    subparsers.add_parser("location", help="Get current location")
    subparsers.add_parser("terminate", help="Terminate session")

    # Log command
    log_parser = subparsers.add_parser("log", help="View script output logs (what the script being debugged prints)")
    log_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show (default: 50)")
    log_parser.add_argument("--follow", action="store_true", help="Follow logs in real-time (like tail -f)")

    # Debug Save/Load commands
    subparsers.add_parser("tensors-list", help="List all saved debug tensors")

    tensors_info_parser = subparsers.add_parser("tensors-info", help="Get information about a saved debug tensor")
    tensors_info_parser.add_argument("name", help="Tensor name")

    tensors_clear_parser = subparsers.add_parser("tensors-clear", help="Clear saved debug tensors")
    tensors_clear_parser.add_argument("--name", help="Specific tensor name to clear (omit to clear all)")

    # Tutorial command
    tutorial_parser = subparsers.add_parser("tutorial", help="Interactive tutorial - learn by executing commands")
    tutorial_parser.add_argument(
        "lesson", nargs="?", default="basic", choices=["basic"], help="Tutorial lesson (default: basic)"
    )

    # Coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Run script with coverage tracking to find dead code")
    coverage_parser.add_argument("script", help="Path to script to analyze")
    coverage_parser.add_argument("--output", help="Output path for coverage report (default: COVERAGE_REPORT_*.md)")
    coverage_parser.add_argument(
        "--include",
        action="append",
        dest="include_dirs",
        help="Additional directories to include in coverage (default: src/mflux only). Can be used multiple times. Example: --include src/mflux_debugger",
    )

    # Coverage-multi command
    coverage_multi_parser = subparsers.add_parser(
        "coverage-multi", help="Run multiple scripts with coverage tracking and combine results"
    )
    coverage_multi_parser.add_argument("scripts", nargs="+", help="Paths to scripts to analyze (at least 2)")
    coverage_multi_parser.add_argument("--output", help="Output path for coverage report")
    coverage_multi_parser.add_argument(
        "--include",
        action="append",
        dest="include_dirs",
        help="Additional directories to include in coverage (default: src/mflux only). Can be used multiple times.",
    )

    return parser


def _dispatch_command(cli: DebuggerCLI, args, parser):
    """Common command dispatch logic for both MLX and PyTorch CLIs."""
    if args.command == "start":
        cli.cmd_start(
            args.script,
            reset=not args.no_reset,
            keep_tensors=args.keep_tensors if hasattr(args, "keep_tensors") else False,
        )
    elif args.command == "break":
        cli.cmd_break(args.file, args.line, args.condition)
    elif args.command == "checkpoint-break":
        cli.cmd_checkpoint_break(
            checkpoint_name=args.checkpoint_name,
            context=args.context,
            all_checkpoints=args.all_checkpoints,
            pattern=args.pattern,
            script_dir=args.script_dir,
        )
    elif args.command == "checkpoint-list":
        cli.cmd_checkpoint_list(script_dir=args.script_dir)
    elif args.command == "remove":
        cli.cmd_remove_break(args.file, args.line)
    elif args.command == "breaks":
        cli.cmd_list_breaks()
    elif args.command == "continue":
        cli.cmd_continue(args.max_wait)
    elif args.command == "eval":
        cli.cmd_eval(args.expression)
    elif args.command == "vars":
        cli.cmd_vars()
    elif args.command == "status":
        cli.cmd_status(verbose=args.verbose)
    elif args.command == "location":
        cli.cmd_location()
    elif args.command == "terminate":
        cli.cmd_terminate()
    elif args.command == "log":
        cli.cmd_log(args.lines, args.follow)
    elif args.command == "tensors-list":
        cli.cmd_debug_list()
    elif args.command == "tensors-info":
        cli.cmd_debug_info(args.name)
    elif args.command == "tensors-clear":
        cli.cmd_debug_clear(args.name if hasattr(args, "name") and args.name else None)
    elif args.command == "tutorial":
        cli.cmd_tutorial(args.lesson)
    elif args.command == "coverage":
        cli.cmd_coverage(
            args.script, args.output, include_dirs=args.include_dirs if hasattr(args, "include_dirs") else None
        )
    elif args.command == "coverage-multi":
        cli.cmd_coverage_multi(
            args.scripts, args.output, include_dirs=args.include_dirs if hasattr(args, "include_dirs") else None
        )
    else:
        parser.print_help()
        sys.exit(1)


def main_mlx():
    """Entry point for mflux-debug-mlx."""
    parser = create_parser("mlx")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = DebuggerCLI("mlx", MLX_PORT)
    _dispatch_command(cli, args, parser)


def main_pytorch():
    """Entry point for mflux-debug-pytorch."""
    parser = create_parser("pytorch")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = DebuggerCLI("pytorch", PYTORCH_PORT)
    _dispatch_command(cli, args, parser)


if __name__ == "__main__":
    # Determine which command was called
    prog_name = Path(sys.argv[0]).name
    if "mlx" in prog_name:
        main_mlx()
    else:
        main_pytorch()
