"""
Lightweight debugger for ML workloads.

Minimal overhead tracing that only instruments user code,
skipping heavy ML libraries entirely.
"""

import logging
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DebugState(Enum):
    """Current state of the debugger."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


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


class LightweightDebugger:
    """
    Lightweight debugger optimized for ML workloads.

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
        """Initialize the lightweight debugger."""
        self.state = DebugState.NOT_STARTED
        self.breakpoints: Dict[Tuple[str, int], Breakpoint] = {}
        self.watch_files: Set[str] = set()
        self.current_frame: Optional[Any] = None
        self.step_mode: Optional[str] = None  # None, 'over', 'into', 'out'
        self.step_depth: int = 0
        self.paused_event = threading.Event()
        self.continue_event = threading.Event()
        self.script_path: Optional[str] = None
        self.script_process: Optional[subprocess.Popen] = None

        # Pause state cache for thread-safe async execution
        self._pause_cache: Dict[str, Any] = {
            "location": None,  # (file, line, function)
            "variables": None,  # dict of locals
            "stack": None,  # list of StackFrame objects
        }
        self._cache_lock = threading.Lock()

    def should_trace_file(self, filename: str) -> bool:
        """
        Determine if we should trace this file.

        Returns True only for user code we're watching.
        """
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

        # If we have watch files, only trace those
        if self.watch_files:
            # Normalize the filename for comparison
            try:
                normalized = str(Path(filename).resolve())
                return normalized in self.watch_files
            except Exception:  # noqa: BLE001
                return filename in self.watch_files

        # Otherwise, trace files in the script's directory
        if self.script_path:
            script_dir = str(Path(self.script_path).parent)
            return filename.startswith(script_dir)

        return False

    def add_watch_file(self, file_path: str):
        """Add a file to watch (trace execution in this file)."""
        abs_path = str(Path(file_path).resolve())
        self.watch_files.add(abs_path)
        logger.info(f"Watching file: {abs_path}")

    def set_breakpoint(self, file_path: str, line: int, condition: Optional[str] = None) -> Breakpoint:
        """Set a breakpoint at the specified location."""
        abs_path = str(Path(file_path).resolve())
        bp = Breakpoint(abs_path, line, condition)
        self.breakpoints[(abs_path, line)] = bp

        # Automatically watch this file
        self.add_watch_file(abs_path)

        logger.info(f"Breakpoint set: {abs_path}:{line}")
        return bp

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

            # Set state INSIDE lock to ensure atomicity with cache
            self.state = DebugState.PAUSED
            self.current_frame = frame

        # Signal pause event (outside lock to avoid deadlock)
        self.paused_event.set()

        logger.info(f"Paused at {frame.f_code.co_filename}:{frame.f_lineno} ({reason})")

        # Wait for continue signal
        self.continue_event.wait()
        self.continue_event.clear()
        self.paused_event.clear()

        self.state = DebugState.RUNNING

    def start_script(self, script_path: str, args: List[str] = None) -> None:
        """
        Start debugging a script in the current process.

        Note: This runs the script in the same process with tracing enabled.
        """
        self.script_path = str(Path(script_path).resolve())
        self.state = DebugState.RUNNING

        # Automatically watch the main script
        self.add_watch_file(self.script_path)

        logger.info(f"Starting script with lightweight tracer: {self.script_path}")

        # Run the script in a thread so we can control it
        def run_script():
            try:
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

            except Exception as e:  # noqa: BLE001
                logger.error(f"Script execution failed: {e}")
                logger.error(traceback.format_exc())
            finally:
                self.state = DebugState.FINISHED
                sys.settrace(None)
                threading.settrace(None)
                self.paused_event.set()  # Wake up any waiting threads

        self.script_thread = threading.Thread(target=run_script, daemon=True)
        self.script_thread.start()

        # Wait for first pause or completion (long timeout for ML model loading)
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
        self._clear_pause_cache()  # Clear cache when resuming
        self.continue_event.set()

        # Wait for next pause or finish
        self.paused_event.wait(timeout=60.0)

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
        self._clear_pause_cache()  # Clear cache when resuming
        self.continue_event.set()

        # Wait for next pause
        self.paused_event.wait(timeout=60.0)

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
        self._clear_pause_cache()  # Clear cache when resuming
        self.continue_event.set()

        # Wait for next pause
        self.paused_event.wait(timeout=60.0)

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
        self._clear_pause_cache()  # Clear cache when resuming
        self.continue_event.set()

        # Wait for next pause
        self.paused_event.wait(timeout=60.0)

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

            # Batch evaluate all found MLX arrays
            if mlx_arrays:
                mx.eval(*mlx_arrays)
                logger.debug(f"Auto-evaluated {len(mlx_arrays)} MLX arrays: {mlx_keys}")

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

    def evaluate(self, expression: str, auto_eval_mlx: bool = True) -> Any:
        """
        Evaluate an expression in the current context.

        Args:
            expression: Python expression to evaluate
            auto_eval_mlx: Automatically evaluate if result is an MLX array

        Returns:
            Result of evaluation

        Raises:
            Exception if evaluation fails
        """
        if not self.current_frame:
            raise RuntimeError("No current frame - not paused")

        # First, auto-evaluate any MLX arrays in scope to prevent evaluation errors
        if auto_eval_mlx:
            variables = dict(self.current_frame.f_locals)
            self._auto_eval_mlx_in_scope(variables)

        result = eval(expression, self.current_frame.f_globals, self.current_frame.f_locals)

        # If result is an MLX array, evaluate it
        if auto_eval_mlx:
            try:
                import mlx.core as mx  # noqa: PLC0415

                if isinstance(result, mx.array):
                    mx.eval(result)
                    logger.debug(f"Auto-evaluated MLX result for expression: {expression}")
            except (ImportError, Exception):  # noqa: BLE001
                pass

        return result

    def terminate(self):
        """Terminate the debug session."""
        self.state = DebugState.FINISHED
        sys.settrace(None)
        threading.settrace(None)

        # Wake up any waiting threads
        self.continue_event.set()
        self.paused_event.set()

        logger.info("Debug session terminated")
