"""
Debugger for PyTorch and MLX.

Minimal overhead tracing that only instruments user code,
skipping heavy ML libraries entirely.
"""

import logging
import os
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
TRACE_DEBUG_PATTERN = os.environ.get("MFLUX_DEBUGGER_TRACE_PATTERN")
TRACE_DEBUG_MAX_EVENTS = int(os.environ.get("MFLUX_DEBUGGER_TRACE_MAX", "200"))

# Global registry for active debugger instance (for semantic checkpoints)
_ACTIVE_DEBUGGER: Optional["Debugger"] = None


class DebugState(Enum):
    """Current state of the debugger."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"  # Script execution failed with an error


@dataclass
class Breakpoint:
    """Represents a breakpoint."""

    file_path: str
    line: int
    condition: Optional[str] = None
    enabled: bool = True


@dataclass
class StackFrame:
    """Represents a stack frame."""

    file_path: str
    line: int
    function_name: str
    locals: Dict[str, Any]
    globals: Dict[str, Any]


def set_active_debugger(debugger: Optional["Debugger"]) -> None:
    """Set the global active debugger instance."""
    global _ACTIVE_DEBUGGER
    _ACTIVE_DEBUGGER = debugger


def get_active_debugger() -> Optional["Debugger"]:
    """Get the global active debugger instance."""
    return _ACTIVE_DEBUGGER


class Debugger:
    """
    Debugger optimized for ML workloads.

    Only traces user-specified files, completely skipping ML libraries
    to minimize overhead during heavy computations.
    """

    # Libraries to completely skip (never trace)
    SKIP_LIBRARIES = {
        "torch",
        "transformers",
        "diffusers",
        "mlx",
        "numpy",
        "huggingface_hub",
        "safetensors",
        "tokenizers",
        "site-packages",
        "dist-packages",
    }

    def __init__(self):
        """Initialize the debugger."""
        self.state = DebugState.NOT_STARTED
        self.breakpoints: Dict[Tuple[str, int], Breakpoint] = {}
        self.watch_files: Set[str] = set()
        self.watch_dirs: Set[str] = set()
        self.current_frame: Optional[Any] = None
        self.step_mode: Optional[str] = None  # None, 'over', 'into', 'out'
        self.step_depth: int = 0
        self.paused_event = threading.Event()
        self.continue_event = threading.Event()
        self.script_path: Optional[str] = None
        self.script_process: Optional[subprocess.Popen] = None
        self._trace_debug_counts: Dict[str, int] = {}

        # Coverage tracking (zero overhead when disabled)
        self.coverage_mode: bool = False  # Opt-in flag - no overhead when False
        self.coverage_data: Optional[Dict[str, Set[int]]] = None  # file -> set of executed line numbers

        # Pause state cache for thread-safe async execution
        self._pause_cache: Dict[str, Any] = {
            "location": None,  # (file, line, function)
            "variables": None,  # dict of locals
            "stack": None,  # list of StackFrame objects
        }
        self._cache_lock = threading.Lock()

        # Enhanced state tracking and logging
        self._trace_callback_count = 0
        self._last_trace_file = None
        self._execution_start_time = None
        self._state_lock = threading.Lock()  # Protect state transitions
        self._script_started = False

        # Exception and output tracking
        self._last_exception: Optional[Exception] = None
        self._exception_traceback: Optional[str] = None
        self._exit_code: int = 0  # 0 = success, non-zero = error
        self._stdout_capture: List[str] = []
        self._stderr_capture: List[str] = []

        # Semantic checkpoint tracking
        self.checkpoint_breakpoints: Dict[str, Breakpoint] = {}  # checkpoint_name -> synthetic breakpoint
        self.checkpoint_hits: List[Dict[str, Any]] = []  # History of checkpoint hits
        self.current_checkpoint: Optional[Dict[str, Any]] = None  # Currently paused checkpoint
        self.checkpoint_variables: Dict[str, Any] = {}  # Store checkpoint variables for evaluation
        self.record_all_checkpoints: bool = False  # If True, record all checkpoints even without breakpoints
        self.break_all_checkpoints: bool = False  # If True, break at ALL checkpoints for interactive debugging

        # Hit counter tracking (for verification)
        self.checkpoint_hit_counts: Dict[str, int] = {}  # checkpoint_name -> number of times hit
        self.checkpoint_order: List[str] = []  # Order in which checkpoints were hit (for validation)

        # Conditional checkpoint breakpoints (context-based)
        self.conditional_checkpoint_breakpoints: Dict[str, Dict[str, Any]] = {}  # checkpoint_name -> condition dict

    def should_trace_file(self, filename: str) -> bool:
        """
        Determine if we should trace this file.

        Returns True only for user code we're watching.
        """
        # COVERAGE MODE: Track all executed Python files (except stdlib and heavy ML libraries)
        if self.coverage_mode:
            # Skip standard library
            if "/lib/python" in filename or "/lib64/python" in filename:
                return False

            # Skip heavy ML libraries (to avoid overhead) but keep mflux code
            # Only skip if in site-packages (external installs), not if in project
            for lib in self.SKIP_LIBRARIES:
                if lib == "mlx":  # Skip MLX library code, but track mflux code
                    if f"/site-packages/{lib}/" in filename or f"/dist-packages/{lib}/" in filename:
                        return False
                else:
                    # Skip transformers, diffusers, torch, etc. even if editable
                    if f"/site-packages/{lib}/" in filename or f"/dist-packages/{lib}/" in filename:
                        return False

            # Track all other Python files (including mflux code, user code, etc.)
            # Skip frozen imports and non-file code objects
            if "<frozen" in filename or "<builtin" in filename:
                return False

            return True  # Track everything else in coverage mode

        # NORMAL DEBUGGING MODE: Only trace watched files (existing behavior)
        # FIRST: Check if we have explicit breakpoints in this file
        # If so, always trace it regardless of library skip rules
        if self.breakpoints:
            try:
                normalized = str(Path(filename).resolve())
                # Check if any breakpoint is in this file
                for (bp_file, _bp_line), bp in self.breakpoints.items():
                    if bp_file == normalized and bp.enabled:
                        # DEBUG: Log when we find a matching breakpoint in library code
                        if "site-packages" in filename or "/Desktop/" in filename:
                            print(f"✅ Tracing library file with breakpoint: {filename}", flush=True)
                        return True
                # DEBUG: Log when we have a library file but no breakpoint match
                # This helps identify which files are actually being executed
                if ("site-packages" in filename or "/Desktop/" in filename) and len(self.breakpoints) > 0:
                    # Only log occasionally to avoid spam
                    if not hasattr(self, "_logged_files"):
                        self._logged_files = set()
                    if normalized not in self._logged_files and len(self._logged_files) < 50:
                        self._logged_files.add(normalized)
                        print(f"ℹ️  Library file without breakpoint: {filename}", flush=True)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error in should_trace_file breakpoint check: {e}")

        # Skip standard library
        if "/lib/python" in filename or "/lib64/python" in filename:
            return False

        # Skip if it's in a library package directory we're ignoring
        # Check for /site-packages/<lib>/ or /dist-packages/<lib>/
        for lib in self.SKIP_LIBRARIES:
            if f"/site-packages/{lib}/" in filename or f"/dist-packages/{lib}/" in filename:
                return False
            if f"/site-packages\\{lib}\\" in filename or f"/dist-packages\\{lib}\\" in filename:
                return False

        # If we have watch files, only trace those (and their parent dirs)
        if self.watch_files:
            # Normalize the filename for comparison
            try:
                normalized = str(Path(filename).resolve())
            except Exception:  # noqa: BLE001
                normalized = filename

            if normalized in self.watch_files:
                return True

            # Also honor any directories we've explicitly watched
            for watch_dir in self.watch_dirs:
                if normalized.startswith(watch_dir):
                    return True

            return False

        # Otherwise, trace files in the script's directory
        if self.script_path:
            script_dir = str(Path(self.script_path).parent)
            return filename.startswith(script_dir)

        return False

    def add_watch_file(self, file_path: str):
        """Add a file to watch (trace execution in this file)."""
        abs_path = str(Path(file_path).resolve())
        self.watch_files.add(abs_path)
        # Also watch the parent directory so new code objects / regenerated files still match
        parent_dir = str(Path(abs_path).parent.resolve())
        if not parent_dir.endswith(os.sep):
            parent_dir = parent_dir + os.sep
        self.watch_dirs.add(parent_dir)
        # Also ensure we watch the qwen/models tree to catch dynamically imported blocks
        qwen_models_dir = Path(__file__).resolve().parents[2] / "models"
        qwen_models_str = str(qwen_models_dir.resolve())
        if not qwen_models_str.endswith(os.sep):
            qwen_models_str += os.sep
        self.watch_dirs.add(qwen_models_str)
        logger.info(f"Watching file: {abs_path}")

    def set_breakpoint(self, file_path: str, line: int, condition: Optional[str] = None) -> Breakpoint:
        """Set a breakpoint at the specified location."""
        abs_path = str(Path(file_path).resolve())

        # Validate the breakpoint location
        self._validate_breakpoint_line(abs_path, line)

        bp = Breakpoint(abs_path, line, condition)
        self.breakpoints[(abs_path, line)] = bp

        # Automatically watch this file
        self.add_watch_file(abs_path)

        logger.info(f"Breakpoint set: {abs_path}:{line}")
        return bp

    def _validate_breakpoint_line(self, file_path: str, line: int) -> None:
        """
        Validate that a breakpoint line is executable.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If line is empty, comment-only, or non-executable
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot set breakpoint: File not found: {file_path}")
        except OSError as e:  # noqa: BLE001
            logger.warning(f"Could not validate breakpoint line: {e}")
            return  # Don't block breakpoint setting on read errors

        # Check if line number is valid
        if line < 1 or line > len(lines):
            raise ValueError(
                f"Cannot set breakpoint: Line {line} is out of range (file has {len(lines)} lines): {file_path}"
            )

        # Get the line content (0-indexed)
        line_content = lines[line - 1]
        stripped = line_content.strip()

        # Check for empty lines - HARD REJECT
        if not stripped:
            # Find nearby executable lines to suggest
            suggestions = []
            for offset in range(1, 6):
                if line + offset <= len(lines):
                    next_line = lines[line + offset - 1].strip()
                    if next_line and not next_line.startswith("#"):
                        suggestions.append(f"  → Line {line + offset}: {next_line[:60]}")
                        break

            error_msg = (
                f"❌ INVALID BREAKPOINT: Cannot set on empty line {line}\n"
                f"   File: {file_path}\n"
                f"   Line {line} is blank - empty lines are NEVER executed.\n"
            )
            if suggestions:
                error_msg += "\n💡 Suggested executable lines nearby:\n" + "\n".join(suggestions)

            raise ValueError(error_msg)

        # Check for comment-only lines - ALSO HARD REJECT
        if stripped.startswith("#"):
            # Find nearby executable lines
            suggestions = []
            for offset in [1, 2, 3, -1, -2]:
                check_line = line + offset
                if 1 <= check_line <= len(lines):
                    check_content = lines[check_line - 1].strip()
                    if check_content and not check_content.startswith("#"):
                        suggestions.append(f"  → Line {check_line}: {check_content[:60]}")
                        if len(suggestions) >= 2:
                            break

            error_msg = (
                f"❌ INVALID BREAKPOINT: Cannot set on comment-only line {line}\n"
                f"   File: {file_path}\n"
                f"   Line {line}: {stripped}\n"
                f"   Comment lines are NEVER executed.\n"
            )
            if suggestions:
                error_msg += "\n💡 Suggested executable lines nearby:\n" + "\n".join(suggestions)

            raise ValueError(error_msg)

        # Check for closing brackets/parens only - HARD REJECT
        if stripped in [")", "]", "}", "),", "],", "},", "),"]:
            # Find nearby executable lines
            suggestions = []
            for offset in [-1, -2, 1, 2, 3]:
                check_line = line + offset
                if 1 <= check_line <= len(lines):
                    check_content = lines[check_line - 1].strip()
                    if check_content and not check_content.startswith("#") and check_content not in [")", "]", "}"]:
                        suggestions.append(f"  → Line {check_line}: {check_content[:60]}")
                        if len(suggestions) >= 2:
                            break

            error_msg = (
                f"❌ INVALID BREAKPOINT: Cannot set on closing bracket/paren line {line}\n"
                f"   File: {file_path}\n"
                f"   Line {line}: {stripped}\n"
                f"   Closing brackets are part of multi-line statements and won't trigger breakpoints.\n"
            )
            if suggestions:
                error_msg += "\n💡 Suggested executable lines nearby:\n" + "\n".join(suggestions)

            raise ValueError(error_msg)

        # Check for common non-executable lines
        non_executable_patterns = [
            '"""',  # Docstring
            "'''",  # Docstring
            "pass",  # Pass statement (may or may not be hit)
        ]

        for pattern in non_executable_patterns:
            if stripped == pattern or (
                pattern in ['"""', "'''"] and stripped.startswith(pattern) and stripped.endswith(pattern)
            ):
                logger.warning(
                    f"Setting breakpoint on potentially non-executable line {line}: {file_path}\n"
                    f"Line {line}: {stripped}\n"
                    f"This line may not trigger the breakpoint as expected."
                )

    def remove_breakpoint(self, file_path: str, line: int):
        """Remove a breakpoint."""
        abs_path = str(Path(file_path).resolve())
        if (abs_path, line) in self.breakpoints:
            del self.breakpoints[(abs_path, line)]
            logger.info(f"Breakpoint removed: {abs_path}:{line}")

    def list_breakpoints(self) -> List[Breakpoint]:
        """List all breakpoints."""
        return list(self.breakpoints.values())

    def check_breakpoint(self, frame: Any) -> bool:
        """Check if current location has a breakpoint."""
        # Normalize path to absolute to match how breakpoints are stored
        filename = str(Path(frame.f_code.co_filename).resolve())
        lineno = frame.f_lineno

        bp = self.breakpoints.get((filename, lineno))
        if not bp or not bp.enabled:
            return False

        # Check condition if present
        if bp.condition:
            try:
                result = eval(bp.condition, frame.f_globals, frame.f_locals)
                return bool(result)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Breakpoint condition failed: {e}")
                return False

        return True

    def trace_function(self, frame: Any, event: str, arg: Any) -> Optional[Callable]:
        """
        Main trace function called by Python for each event.

        This is the critical path - must be fast!
        """
        filename = frame.f_code.co_filename

        # Fast path: skip files we're not watching
        if not self.should_trace_file(filename):
            return None  # Disable tracing for this frame tree

        # Track trace callbacks
        self._trace_callback_count += 1

        # Log when script actually starts executing (first line in main script)
        if not self._script_started and event == "line" and filename == self.script_path:
            self._script_started = True
            logger.info(f"📝 Script execution started: {Path(filename).name}:{frame.f_lineno}")

        # Track file changes
        if filename != self._last_trace_file:
            self._last_trace_file = filename
            logger.debug(f"  → Tracing {Path(filename).name}")

        if TRACE_DEBUG_PATTERN and TRACE_DEBUG_PATTERN in filename and event == "line":
            count = self._trace_debug_counts.get(filename, 0)
            if count < TRACE_DEBUG_MAX_EVENTS:
                logger.info(f"[TraceDebug] {filename}:{frame.f_lineno} ({event})")
                self._trace_debug_counts[filename] = count + 1

        # Handle different events
        if event == "call":
            return self._handle_call(frame)
        elif event == "line":
            return self._handle_line(frame)
        elif event == "return":
            return self._handle_return(frame, arg)
        elif event == "exception":
            return self._handle_exception(frame, arg)

        return self.trace_function

    def _handle_call(self, frame: Any) -> Optional[Callable]:
        """Handle function call event."""
        if self.step_mode == "into":
            self.step_depth += 1

        return self.trace_function

    def _handle_line(self, frame: Any) -> Optional[Callable]:
        """Handle line execution event."""
        # Coverage tracking (zero overhead when disabled - single if check)
        if self.coverage_mode:
            self._track_line_coverage(frame)

        # Check for breakpoint
        if self.check_breakpoint(frame):
            self._pause_at(frame, "breakpoint")
            return self.trace_function

        # Check for step mode
        if self.step_mode == "over":
            # Only pause if we're at the same depth or higher
            self._pause_at(frame, "step")
            self.step_mode = None
            return self.trace_function
        elif self.step_mode == "into":
            # Pause at any line
            self._pause_at(frame, "step")
            self.step_mode = None
            return self.trace_function

        return self.trace_function

    def _track_line_coverage(self, frame: Any) -> None:
        """Track executed line for coverage (minimal overhead - fast set addition)."""
        if not self.coverage_data:
            self.coverage_data = {}

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Fast set addition - O(1) average case
        if filename not in self.coverage_data:
            self.coverage_data[filename] = set()
        self.coverage_data[filename].add(lineno)

    def get_coverage_data(self) -> Optional[Dict[str, Set[int]]]:
        """Get coverage data (file -> set of executed line numbers)."""
        return self.coverage_data.copy() if self.coverage_data else None

    def _handle_return(self, frame: Any, return_value: Any) -> Optional[Callable]:
        """Handle function return event."""
        if self.step_mode == "out":
            # Pause after we return from current function
            self.step_mode = "over"  # Will pause on next line

        return self.trace_function

    def _handle_exception(self, frame: Any, exc_info: Tuple) -> Optional[Callable]:
        """Handle exception event."""
        exc_type, exc_value, exc_tb = exc_info
        logger.info(f"Exception in traced code: {exc_type.__name__}: {exc_value}")
        return self.trace_function

    def _pause_at(self, frame: Any, reason: str):
        """Pause execution at current frame."""
        logger.info(f"⏸️  Paused at {frame.f_code.co_filename}:{frame.f_lineno} ({reason})")

        # Capture state immediately for thread-safe access
        with self._cache_lock:
            # Cache location
            self._pause_cache["location"] = (
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_name,
            )

            # Cache variables (shallow copy is sufficient)
            self._pause_cache["variables"] = dict(frame.f_locals)

            # Cache stack trace
            stack_frames = []
            current = frame
            while current is not None:
                if self.should_trace_file(current.f_code.co_filename):
                    stack_frames.append(
                        StackFrame(
                            file_path=current.f_code.co_filename,
                            line=current.f_lineno,
                            function_name=current.f_code.co_name,
                            locals=dict(current.f_locals),
                            globals=current.f_globals,
                        )
                    )
                current = current.f_back
            self._pause_cache["stack"] = stack_frames

        # Set state and frame (with lock to ensure atomicity)
        with self._state_lock:
            self.state = DebugState.PAUSED
            self.current_frame = frame

        # Signal pause event (outside lock to avoid deadlock)
        self.paused_event.set()

        logger.info(f"Paused at {frame.f_code.co_filename}:{frame.f_lineno} ({reason})")

        # Wait for continue signal
        self.continue_event.wait()
        self.continue_event.clear()
        self.paused_event.clear()

        with self._state_lock:
            self.state = DebugState.RUNNING
        logger.debug("▶️  Resuming execution...")

    def start_script(self, script_path: str, args: List[str] = None) -> None:
        """
        Start debugging a script in the current process.

        Note: This runs the script in the same process with tracing enabled.
        """
        import time

        self.script_path = str(Path(script_path).resolve())

        with self._state_lock:
            self.state = DebugState.RUNNING
            self._execution_start_time = time.time()
            self._script_started = False

        # Register as active debugger for semantic checkpoints
        set_active_debugger(self)

        # Activate semantic checkpoint system
        try:
            from mflux_debugger.semantic_checkpoint import set_debugger_active

            set_debugger_active(True)
        except ImportError:
            pass  # semantic_checkpoint module may not be available

        # Automatically watch the main script
        self.add_watch_file(self.script_path)
        # Ensure any future threads inherit the tracer
        threading.settrace(self.trace_function)

        logger.info(f"🚀 Starting script with tracer: {self.script_path}")
        logger.info(f"   Breakpoints set: {len(self.breakpoints)}")
        logger.info(f"   Watch files: {len(self.watch_files)}")

        # Run the script in a thread so we can control it
        def run_script():
            # Capture stdout/stderr
            import io

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            # Get the script output log file from checkpoint writer (if exists)
            script_output_file = None
            try:
                from pathlib import Path

                from mflux_debugger.checkpoint_writer import get_checkpoint_writer

                script_name = Path(self.script_path).stem
                writer = get_checkpoint_writer(script_name)
                if writer and writer.output_log_file:
                    script_output_file = writer.output_log_file
            except Exception:  # noqa: BLE001, S110
                pass  # No checkpoint writer, just capture to memory

            try:
                # Redirect stdout/stderr to both buffer and file
                if script_output_file:
                    # Tee to both buffer and file
                    from mflux_debugger.tee_writer import TeeWriter

                    sys.stdout = TeeWriter(stdout_buffer, script_output_file)
                    sys.stderr = TeeWriter(stderr_buffer, script_output_file, prefix="[stderr] ")
                else:
                    # Just buffer
                    sys.stdout = stdout_buffer
                    sys.stderr = stderr_buffer

                # Register this debugger as active (for semantic checkpoints)
                set_active_debugger(self)
                from mflux_debugger.semantic_checkpoint import set_debugger_active

                set_debugger_active(True)

                # Enable tracing in this thread
                sys.settrace(self.trace_function)

                # Read and execute the script
                with open(self.script_path) as f:
                    code = compile(f.read(), self.script_path, "exec")

                # Set up globals for the script
                script_globals = {
                    "__name__": "__main__",
                    "__file__": self.script_path,
                }

                # Execute
                exec(code, script_globals)

                # Success - exit code 0
                self._exit_code = 0
                with self._state_lock:
                    self.state = DebugState.FINISHED
                logger.info(f"✅ Script finished successfully (total callbacks: {self._trace_callback_count})")

            except Exception as e:  # noqa: BLE001
                # Failure - capture exception and set exit code
                self._last_exception = e
                self._exception_traceback = traceback.format_exc()
                self._exit_code = 1

                with self._state_lock:
                    self.state = DebugState.FAILED

                logger.error(f"❌ Script execution failed: {e}")
                logger.error(self._exception_traceback)

            finally:
                # Restore stdout/stderr and capture output
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                self._stdout_capture = stdout_buffer.getvalue().splitlines()
                self._stderr_capture = stderr_buffer.getvalue().splitlines()

                sys.settrace(None)
                threading.settrace(None)
                self.paused_event.set()  # Wake up any waiting threads

        self.script_thread = threading.Thread(target=run_script, daemon=False)
        self.script_thread.start()

        # Wait for first pause or completion
        self.paused_event.wait(timeout=300.0)

    def _clear_pause_cache(self):
        """Clear the pause cache when resuming execution."""
        with self._cache_lock:
            self._pause_cache = {
                "location": None,
                "variables": None,
                "stack": None,
            }

    def continue_execution(self) -> bool:
        """
        Continue execution until next breakpoint.

        Returns True if paused at a location, False if finished.
        """
        if self.state != DebugState.PAUSED:
            logger.warning("Cannot continue - not paused")
            return False

        self.step_mode = None
        self.paused_event.clear()  # Clear the event before continuing
        self.continue_event.set()

        # Wait for next pause or finish with timeout detection (Fix 3)
        self.paused_event.wait(timeout=60.0)

        # Fix 1: Don't clear cache prematurely - if we timeout, keep old location
        # Fix 2: Verify cache is populated if we paused
        if self.state == DebugState.PAUSED:
            with self._cache_lock:
                has_location = self._pause_cache.get("location") is not None

            if not has_location:
                logger.warning("Paused but location cache not populated")

        return self.state == DebugState.PAUSED

    def step_over(self) -> bool:
        """
        Execute current line and pause at next line (don't step into functions).

        Returns True if paused at a location, False if finished.
        """
        if self.state != DebugState.PAUSED:
            logger.warning("Cannot step - not paused")
            return False

        self.step_mode = "over"
        self.paused_event.clear()  # Clear the event before continuing
        self.continue_event.set()

        # Wait for next pause with timeout detection (Fix 3)
        timed_out = not self.paused_event.wait(timeout=60.0)

        # Fix 1 & 2: Only clear old cache if we successfully paused at new location
        # (cache will be repopulated by _pause_at if successful)
        if timed_out:
            logger.warning("Step over timed out - debugger may be hung")
            return False

        # Fix 2: Verify cache is populated after step
        with self._cache_lock:
            has_location = self._pause_cache.get("location") is not None

        if not has_location and self.state == DebugState.PAUSED:
            logger.warning("Step completed but location cache not populated")

        return self.state == DebugState.PAUSED

    def step_into(self) -> bool:
        """
        Execute current line and pause at next line (step into functions).

        Returns True if paused at a location, False if finished.
        """
        if self.state != DebugState.PAUSED:
            logger.warning("Cannot step - not paused")
            return False

        self.step_mode = "into"
        self.paused_event.clear()  # Clear the event before continuing
        self.continue_event.set()

        # Wait for next pause with timeout detection (Fix 3)
        paused = self.paused_event.wait(timeout=60.0)

        # Fix 1 & 2: Only clear old cache if we successfully paused at new location
        if not paused:
            logger.warning("Step into timed out - debugger may be hung")
            return False

        # Fix 2: Verify cache is populated after step
        with self._cache_lock:
            has_location = self._pause_cache.get("location") is not None

        if not has_location and self.state == DebugState.PAUSED:
            logger.warning("Step completed but location cache not populated")

        return self.state == DebugState.PAUSED

    def step_out(self) -> bool:
        """
        Continue until current function returns.

        Returns True if paused at a location, False if finished.
        """
        if self.state != DebugState.PAUSED:
            logger.warning("Cannot step - not paused")
            return False

        self.step_mode = "out"
        self.paused_event.clear()  # Clear the event before continuing
        self.continue_event.set()

        # Wait for next pause with timeout detection (Fix 3)
        paused = self.paused_event.wait(timeout=60.0)

        # Fix 1 & 2: Only clear old cache if we successfully paused at new location
        if not paused:
            logger.warning("Step out timed out - debugger may be hung")
            return False

        # Fix 2: Verify cache is populated after step
        with self._cache_lock:
            has_location = self._pause_cache.get("location") is not None

        if not has_location and self.state == DebugState.PAUSED:
            logger.warning("Step completed but location cache not populated")

        return self.state == DebugState.PAUSED

    def get_location(self) -> Optional[Tuple[str, int, str]]:
        """
        Get current execution location.

        Returns (file_path, line_number, function_name) or None.

        Thread-safe: Uses cached location when paused for async execution.
        """
        # Use cache if paused (thread-safe for async execution)
        if self.state == DebugState.PAUSED:
            with self._cache_lock:
                return self._pause_cache.get("location")

        # Fallback to current frame for sync execution
        if not self.current_frame:
            return None

        frame = self.current_frame
        return (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)

    def get_stack_trace(self) -> List[StackFrame]:
        """
        Get the current call stack.

        Thread-safe: Uses cached stack when paused for async execution.
        """
        # Use cache if paused (thread-safe for async execution)
        if self.state == DebugState.PAUSED:
            with self._cache_lock:
                cached_stack = self._pause_cache.get("stack")
                return cached_stack if cached_stack is not None else []

        # Fallback to current frame for sync execution
        if not self.current_frame:
            return []

        frames = []
        frame = self.current_frame

        while frame is not None:
            # Only include frames from files we're watching
            if self.should_trace_file(frame.f_code.co_filename):
                stack_frame = StackFrame(
                    file_path=frame.f_code.co_filename,
                    line=frame.f_lineno,
                    function_name=frame.f_code.co_name,
                    locals=dict(frame.f_locals),
                    globals=frame.f_globals,
                )
                frames.append(stack_frame)

            frame = frame.f_back

        return frames

    def _auto_eval_mlx_in_scope(self, variables: Dict[str, Any]) -> None:
        """
        Automatically evaluate MLX arrays in the given variable dict.

        This handles MLX's lazy evaluation by detecting and evaluating
        MLX arrays before inspection, preventing crashes from unevaluated arrays.

        Args:
            variables: Dictionary of variables to check and evaluate (modified in-place)
        """
        try:
            # Try to import MLX (may not be available)
            import mlx.core as mx  # noqa: PLC0415

            # Collect all MLX arrays to evaluate in one batch
            mlx_arrays = []
            mlx_keys = []

            for name, value in list(variables.items()):
                try:  # noqa: PERF203
                    # Check if this is an MLX array
                    if isinstance(value, mx.array):
                        mlx_arrays.append(value)
                        mlx_keys.append(name)
                    # Also check nested structures (dicts, lists)
                    elif isinstance(value, dict):
                        self._eval_mlx_in_dict(value, mx)
                    elif isinstance(value, (list, tuple)):
                        self._eval_mlx_in_sequence(value, mx)
                except Exception:  # noqa: BLE001, PERF203
                    # Skip variables that can't be checked safely
                    continue

            # Batch evaluate all found MLX arrays (with Metal error protection)
            if mlx_arrays:
                try:
                    mx.eval(*mlx_arrays)
                    logger.debug(f"Auto-evaluated {len(mlx_arrays)} MLX arrays: {mlx_keys}")
                except (RuntimeError, AssertionError) as e:
                    # Metal/GPU errors during MLX evaluation
                    error_msg = str(e)
                    if "MTL" in error_msg or "Metal" in error_msg or "Completed handler" in error_msg:
                        logger.warning(
                            f"MLX/Metal error during auto-evaluation (this is expected when pausing mid-MLX-operation): {e}"
                        )
                        logger.warning(
                            f"Affected variables: {mlx_keys}. "
                            "These arrays may show as unevaluated. "
                            "Consider setting breakpoints between MLX operations, not during them."
                        )
                    else:
                        # Re-raise non-Metal errors
                        raise

        except ImportError:
            # MLX not available, skip
            pass
        except Exception as e:  # noqa: BLE001
            # Log but don't crash if evaluation fails
            logger.warning(f"MLX auto-evaluation failed: {e}")

    def _eval_mlx_in_dict(self, d: dict, mx) -> None:
        """Recursively evaluate MLX arrays in a dictionary."""
        mlx_arrays = []
        for value in d.values():
            try:  # noqa: PERF203
                if isinstance(value, mx.array):
                    mlx_arrays.append(value)
                elif isinstance(value, dict):
                    self._eval_mlx_in_dict(value, mx)
                elif isinstance(value, (list, tuple)):
                    self._eval_mlx_in_sequence(value, mx)
            except Exception:  # noqa: BLE001, PERF203
                continue
        if mlx_arrays:
            mx.eval(*mlx_arrays)

    def _eval_mlx_in_sequence(self, seq, mx) -> None:
        """Recursively evaluate MLX arrays in a list/tuple."""
        mlx_arrays = []
        for item in seq:
            try:  # noqa: PERF203
                if isinstance(item, mx.array):
                    mlx_arrays.append(item)
                elif isinstance(item, dict):
                    self._eval_mlx_in_dict(item, mx)
                elif isinstance(item, (list, tuple)):
                    self._eval_mlx_in_sequence(item, mx)
            except Exception:  # noqa: BLE001, PERF203
                continue
        if mlx_arrays:
            mx.eval(*mlx_arrays)

    def list_variables(self, include_globals: bool = False, auto_eval_mlx: bool = True) -> Dict[str, Any]:
        """
        List variables in current scope.

        Args:
            include_globals: Whether to include global variables
            auto_eval_mlx: Automatically evaluate MLX arrays before returning

        Returns:
            Dictionary of variable names to values

        Thread-safe: Uses cached variables when paused for async execution.
        """
        # Use cache if paused (thread-safe for async execution)
        if self.state == DebugState.PAUSED:
            with self._cache_lock:
                cached_vars = self._pause_cache.get("variables")
                if cached_vars is not None:
                    variables = dict(cached_vars)  # Return a copy
                    # Auto-evaluate MLX arrays if requested
                    if auto_eval_mlx:
                        self._auto_eval_mlx_in_scope(variables)
                    return variables

        # Fallback to current frame for sync execution
        if not self.current_frame:
            return {}

        variables = dict(self.current_frame.f_locals)

        if include_globals:
            # Add globals that aren't builtins
            for name, value in self.current_frame.f_globals.items():
                if not name.startswith("__") and name not in variables:
                    variables[name] = value

        # Auto-evaluate MLX arrays if requested
        if auto_eval_mlx:
            self._auto_eval_mlx_in_scope(variables)

        return variables

    def inspect_variable(self, name: str) -> Optional[Any]:
        """
        Get the value of a specific variable.

        Args:
            name: Variable name (can be dotted like 'obj.attr')

        Returns:
            Variable value or None if not found
        """
        if not self.current_frame:
            return None

        try:
            # Try locals first
            if "." not in name:
                if name in self.current_frame.f_locals:
                    return self.current_frame.f_locals[name]
                if name in self.current_frame.f_globals:
                    return self.current_frame.f_globals[name]

            # For dotted names, evaluate
            return eval(name, self.current_frame.f_globals, self.current_frame.f_locals)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to inspect variable '{name}': {e}")
            return None

    def set_checkpoint_breakpoint(self, checkpoint_name: str, description: str = "") -> None:
        """
        Enable breaking at a semantic checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint to break at
            description: Optional description of this checkpoint
        """
        # Create a synthetic breakpoint for this checkpoint
        # We don't have a specific file/line, so we use the checkpoint name as the key
        self.checkpoint_breakpoints[checkpoint_name] = Breakpoint(
            file_path=f"<checkpoint:{checkpoint_name}>",
            line=0,
            condition=None,
            enabled=True,
        )
        logger.info(f"Checkpoint breakpoint set: {checkpoint_name}")

    def remove_checkpoint_breakpoint(self, checkpoint_name: str) -> None:
        """Remove a checkpoint breakpoint."""
        if checkpoint_name in self.checkpoint_breakpoints:
            del self.checkpoint_breakpoints[checkpoint_name]
            logger.info(f"Checkpoint breakpoint removed: {checkpoint_name}")

    def list_checkpoint_breakpoints(self) -> Dict[str, Breakpoint]:
        """List all checkpoint breakpoints."""
        return self.checkpoint_breakpoints.copy()

    def set_record_all_checkpoints(self, enabled: bool) -> None:
        """
        Enable or disable recording of ALL semantic checkpoints.

        When enabled, all checkpoints will be recorded in checkpoint_hits
        even if they don't have explicit breakpoints. This is useful for
        comparison and debugging without having to set breakpoints on every checkpoint.

        Args:
            enabled: True to record all checkpoints, False to only record breakpointed ones
        """
        self.record_all_checkpoints = enabled
        logger.info(f"Record all checkpoints: {'enabled' if enabled else 'disabled'}")

    def set_break_all_checkpoints(self, enabled: bool) -> None:
        """
        Enable or disable breaking at ALL semantic checkpoints.

        When enabled, execution will pause at EVERY checkpoint hit for interactive
        inspection. This is useful for step-by-step debugging through model execution.

        Note: This also enables recording (same as record_all_checkpoints).

        Args:
            enabled: True to break at all checkpoints, False to only break at explicit ones
        """
        self.break_all_checkpoints = enabled
        if enabled:
            # Also enable recording when breaking at all
            self.record_all_checkpoints = True
        logger.info(f"Break at all checkpoints: {'enabled' if enabled else 'disabled'}")

    def set_conditional_checkpoint_breakpoint(self, checkpoint_name: str, condition: Dict[str, Any]) -> None:
        """
        Set a conditional breakpoint that triggers when checkpoint context matches.

        Args:
            checkpoint_name: Name of the checkpoint to break at
            condition: Dictionary of context conditions to match (e.g., {"block": 0, "timestep": 0})
        """
        self.conditional_checkpoint_breakpoints[checkpoint_name] = condition
        self.record_all_checkpoints = True  # Enable recording to catch the checkpoint
        logger.info(f"Set conditional breakpoint at '{checkpoint_name}' with condition: {condition}")

    def _context_matches_condition(self, context: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """
        Check if execution context matches the breakpoint condition.

        Args:
            context: Current execution context (e.g., {"block": 0, "timestep": 0})
            condition: Required context values (e.g., {"block": 0})

        Returns:
            True if all condition keys match context values
        """
        for key, value in condition.items():
            if key not in context or context[key] != value:
                return False
        return True

    def handle_checkpoint(
        self,
        checkpoint_name: str,
        variables: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        frame: Any,
        verified: bool = False,
    ) -> None:
        """
        Handle a checkpoint hit during execution.

        This is called by the debug_checkpoint() function when code execution
        reaches a semantic checkpoint marker.

        Args:
            checkpoint_name: Name of the checkpoint that was hit (may include stream prefix like "img:name")
            variables: Variables captured at the checkpoint
            context: Execution context (block, timestep, etc.) for synchronization
            frame: The frame where the checkpoint was hit
        """
        # Check if we should break at this checkpoint
        # Support both exact matches and stream-prefixed matches
        # e.g., "before_rope" should match both "before_rope" and "img:before_rope"
        matching_breakpoint = None

        if checkpoint_name in self.checkpoint_breakpoints:
            # Exact match
            matching_breakpoint = self.checkpoint_breakpoints[checkpoint_name]
        elif ":" in checkpoint_name:
            # Check if the base name (without stream prefix) has a breakpoint
            base_name = checkpoint_name.split(":", 1)[1]
            if base_name in self.checkpoint_breakpoints:
                matching_breakpoint = self.checkpoint_breakpoints[base_name]
        else:
            # Check if there's a stream-prefixed version with a breakpoint
            for bp_name in self.checkpoint_breakpoints:
                if ":" in bp_name and bp_name.split(":", 1)[1] == checkpoint_name:
                    matching_breakpoint = self.checkpoint_breakpoints[bp_name]
                    break

        # Determine if we should record/break at this checkpoint
        # NEW BEHAVIOR: Checkpoints break by default unless explicitly configured otherwise
        should_break = True  # Break by default (code-first philosophy)
        should_record = True  # Always record

        # CRITICAL: In coverage mode, never break at checkpoints (we want full execution)
        if self.coverage_mode:
            should_break = False
            logger.debug(f"Checkpoint '{checkpoint_name}' not breaking: coverage_mode=True")

        # Override: Don't break if there's an explicit matching breakpoint that's disabled
        if matching_breakpoint is not None and not matching_breakpoint.enabled:
            should_break = False
            logger.debug(f"Checkpoint '{checkpoint_name}' not breaking: explicit breakpoint disabled")

        # Legacy support: break_all_checkpoints flag (for backward compatibility)
        if self.break_all_checkpoints:
            should_break = True
            should_record = True

        # Check conditional breakpoints (context-based)
        if checkpoint_name in self.conditional_checkpoint_breakpoints:
            condition = self.conditional_checkpoint_breakpoints[checkpoint_name]
            should_record = True  # Always record when there's a conditional breakpoint

            # Check if context matches the condition
            if context and self._context_matches_condition(context, condition):
                should_break = True
                logger.info(f"🎯 Conditional breakpoint matched: {checkpoint_name} with context {context}")

        if not should_record:
            # Not monitoring this checkpoint at all
            logger.debug(f"Checkpoint '{checkpoint_name}' not recorded: should_record=False")
            return

        # Log if checkpoint should break but won't (for debugging)
        if not should_break:
            logger.warning(
                f"⚠️ Checkpoint '{checkpoint_name}' hit but will NOT break "
                f"(coverage_mode={self.coverage_mode}, "
                f"break_all_checkpoints={self.break_all_checkpoints}, "
                f"matching_breakpoint={'disabled' if matching_breakpoint and not matching_breakpoint.enabled else 'none'})"
            )

        # Update hit counter
        self.checkpoint_hit_counts[checkpoint_name] = self.checkpoint_hit_counts.get(checkpoint_name, 0) + 1
        hit_count = self.checkpoint_hit_counts[checkpoint_name]

        # Track checkpoint order
        self.checkpoint_order.append(checkpoint_name)

        # Record the checkpoint hit (don't store actual variable values - they may not be serializable)
        checkpoint_info = {
            "name": checkpoint_name,
            "variable_names": list(variables.keys()),  # Just store names, not values
            "context": context or {},  # Execution context for synchronization
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "function": frame.f_code.co_name,
            "hit_count": hit_count,  # Include hit count in checkpoint info
            "verified": verified,  # Include verification status
        }

        self.checkpoint_hits.append(checkpoint_info)
        self.current_checkpoint = checkpoint_info

        # Display context and hit count if available
        context_parts = []
        if context and (context.get("block") is not None or context.get("timestep") is not None):
            context_parts.append(" ".join([f"{k}={v}" for k, v in context.items()]))
        context_parts.append(f"hit#{hit_count}")
        if verified:
            context_parts.append("✅ VERIFIED")
        context_str = ", ".join(context_parts)
        logger.info(f"📍 Checkpoint '{checkpoint_name}' ({context_str})")

        # Store checkpoint variables for evaluation (replaces broken frame.f_locals.update)
        self.checkpoint_variables = variables.copy()

        # Only pause execution if we should break (not just record)
        if should_break:
            # Pause execution at this checkpoint
            self.current_frame = frame
            with self._state_lock:
                self.state = DebugState.PAUSED

            # Cache the pause state
            with self._cache_lock:
                self._pause_cache["location"] = (
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    frame.f_code.co_name,
                )
                self._pause_cache["variables"] = frame.f_locals.copy()
                self._pause_cache["checkpoint"] = checkpoint_info
                self._pause_cache["checkpoint_variables"] = variables

            # Wait for continue signal
            self.paused_event.set()
            self.continue_event.clear()
            self.continue_event.wait()

        # Clear checkpoint info when continuing
        self.current_checkpoint = None
        if not should_break:
            # Only clear variables if we didn't pause (still may need them for next checkpoint)
            self.checkpoint_variables = {}

    def get_current_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about the current checkpoint if paused at one."""
        return self.current_checkpoint

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get history of all checkpoint hits in this session."""
        return self.checkpoint_hits.copy()

    def evaluate(self, expression: str, auto_eval_mlx: bool = True) -> Any:
        """
        Evaluate an expression or execute multi-line code in the current context.

        Supports:
        - Single-line expressions: "latents.shape"
        - Multi-line code blocks: "import torch\nresult = torch.randn(2, 3)"
        - Code with imports: "from pathlib import Path\npath = Path('/tmp')"

        Args:
            expression: Python expression or multi-line code to evaluate/execute
            auto_eval_mlx: Automatically evaluate if result is an MLX array

        Returns:
            Result of evaluation (for expressions) or the last assigned variable (for statements)

        Raises:
            Exception if evaluation fails
        """
        if not self.current_frame:
            raise RuntimeError("No current frame - not paused")

        # Build evaluation namespace with checkpoint variables
        eval_namespace = {
            **self.current_frame.f_locals,
            **self.checkpoint_variables,  # Add checkpoint variables
        }

        # First, auto-evaluate any MLX arrays in scope IN-PLACE to prevent evaluation errors
        if auto_eval_mlx:
            self._auto_eval_mlx_in_scope(eval_namespace)

        # Detect if this is multi-line code or contains import statements
        is_multiline = "\n" in expression or "import " in expression or "from " in expression

        if is_multiline:
            # Multi-line code or imports - use exec()
            logger.debug(f"Executing multi-line code block: {expression[:50]}...")

            # Create a new namespace for execution that includes current context
            exec_namespace = {
                **self.current_frame.f_globals,
                **eval_namespace,
            }

            # Execute the code block
            exec(expression, exec_namespace, exec_namespace)

            # Try to find the result - look for common result variable names
            result = None
            result_candidates = ["result", "_result", "_", "output", "_output"]

            for candidate in result_candidates:
                if candidate in exec_namespace and candidate not in eval_namespace:
                    result = exec_namespace[candidate]
                    logger.debug(f"Found result in variable: {candidate}")
                    break

            # If no standard result variable, return the last modified variable
            if result is None:
                # Find variables that were added or modified
                new_vars = {
                    k: v for k, v in exec_namespace.items() if k not in eval_namespace and not k.startswith("__")
                }

                if new_vars:
                    # Return the last modified variable (dict preserves insertion order in Python 3.7+)
                    last_var = list(new_vars.items())[-1]
                    result = last_var[1]
                    logger.debug(f"Returning last modified variable: {last_var[0]}")
                else:
                    # No new variables, return None (successful execution but no result)
                    result = None
                    logger.debug("Multi-line code executed successfully, no result variable found")
        else:
            # Single-line expression - use eval() as before
            result = eval(expression, self.current_frame.f_globals, eval_namespace)

        # If result is an MLX array, evaluate it eagerly (with Metal error protection)
        if auto_eval_mlx and result is not None:
            try:
                import mlx.core as mx  # noqa: PLC0415

                if isinstance(result, mx.array):
                    try:
                        mx.eval(result)
                        logger.debug(f"Auto-evaluated MLX result for expression: {expression[:50]}")
                    except (RuntimeError, AssertionError) as e:
                        error_msg = str(e)
                        if "MTL" in error_msg or "Metal" in error_msg or "Completed handler" in error_msg:
                            logger.warning(f"MLX/Metal error during result evaluation: {e}")
                            logger.warning(
                                "Result may be unevaluated. Consider setting breakpoints between MLX operations."
                            )
                        else:
                            raise
                # Also handle lists/tuples of MLX arrays
                elif isinstance(result, (list, tuple)):
                    mlx_arrays = [x for x in result if isinstance(x, mx.array)]
                    if mlx_arrays:
                        try:
                            mx.eval(*mlx_arrays)
                            logger.debug(f"Auto-evaluated {len(mlx_arrays)} MLX arrays in result sequence")
                        except (RuntimeError, AssertionError) as e:
                            error_msg = str(e)
                            if "MTL" in error_msg or "Metal" in error_msg or "Completed handler" in error_msg:
                                logger.warning(f"MLX/Metal error during sequence evaluation: {e}")
                                logger.warning("Some results may be unevaluated.")
                            else:
                                raise
            except (ImportError, Exception):  # noqa: BLE001
                pass

        return result

    def terminate(self):
        """Terminate the debug session."""
        # Only set to FINISHED if not already in FAILED state
        if self.state != DebugState.FAILED:
            self.state = DebugState.FINISHED
        sys.settrace(None)
        threading.settrace(None)

        # Unregister as active debugger
        if get_active_debugger() is self:
            set_active_debugger(None)

        # Deactivate semantic checkpoint system
        try:
            from mflux_debugger.semantic_checkpoint import set_debugger_active

            set_debugger_active(False)
        except ImportError:
            pass

        # Wake up any waiting threads
        self.continue_event.set()
        self.paused_event.set()

        logger.info(f"Debug session terminated (final state: {self.state.value})")
