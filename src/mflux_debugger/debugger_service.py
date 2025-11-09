"""
Debugger service layer - transport-agnostic debugging operations.

This module provides a clean API for debugging operations that can be
used by different transport layers (MCP, FastAPI, gRPC, etc.).
"""

import logging
import re
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mflux_debugger.lightweight_debugger import DebugState, LightweightDebugger

logger = logging.getLogger(__name__)


@dataclass
class DebuggerResponse:
    """Standard response from debugger operations."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DebuggerService:
    """
    High-level debugger service providing transport-agnostic operations.

    This class encapsulates all debugging logic and can be used by any
    transport layer (MCP, FastAPI, CLI, etc.).
    """

    def __init__(self, enable_rich_context: bool = True):
        """Initialize the debugger service.

        Args:
            enable_rich_context: Enable rich context (code context, call stack, auto-preview)
        """
        self.debugger = LightweightDebugger()
        self.enable_rich_context = enable_rich_context
        self._execution_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._execution_start_time: Optional[float] = None
        self._execution_timeout: Optional[float] = None

    def _count_debug_checkpoints(self, script_path: str) -> Dict[str, int]:
        """
        Count approximate number of debug_checkpoint() calls in script and related modules.

        Scans:
        - The main script
        - Local files imported from same directory
        - mflux, transformers, diffusers packages (if editable install)

        Returns:
            Dict with counts per source: {"script": N, "mflux": N, "transformers": N, "diffusers": N}

        This is a rough estimate - doesn't account for conditional logic,
        but gives users an idea of how many checkpoints they have.
        """
        try:
            counts = {"script": 0, "mflux": 0, "transformers": 0, "diffusers": 0}
            scanned_files = set()

            # Pattern for debug_checkpoint calls
            checkpoint_pattern = r"debug_checkpoint\s*\("

            # Scan main script and local files
            script_path_obj = Path(script_path)
            counts["script"] += self._scan_file_for_checkpoints(script_path, checkpoint_pattern)
            scanned_files.add(script_path)

            # Scan local Python files in same directory
            if script_path_obj.parent.exists():
                for py_file in script_path_obj.parent.glob("*.py"):
                    if str(py_file) not in scanned_files:
                        counts["script"] += self._scan_file_for_checkpoints(str(py_file), checkpoint_pattern)
                        scanned_files.add(str(py_file))

            # Scan key packages (mflux, transformers, diffusers) if editable installs
            for package_name in ["mflux", "transformers", "diffusers"]:
                package_count = self._scan_package_for_checkpoints(package_name, checkpoint_pattern)
                if package_count > 0:
                    logger.debug(f"Found {package_count} checkpoints in {package_name}")
                    counts[package_name] = package_count

            return counts
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not scan for checkpoints: {e}")
            return {"script": 0, "mflux": 0, "transformers": 0, "diffusers": 0}

    def _scan_file_for_checkpoints(self, file_path: str, pattern: str) -> int:
        """Scan a single file for checkpoint calls."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            matches = re.findall(pattern, content)
            return len(matches)
        except Exception:  # noqa: BLE001
            return 0

    def _scan_package_for_checkpoints(self, package_name: str, pattern: str) -> int:
        """Scan a package for checkpoint calls (only if editable install)."""
        try:
            import importlib.util

            # Try to find the package
            spec = importlib.util.find_spec(package_name)
            if spec is None or spec.origin is None:
                return 0

            # Only scan if it's an editable install (local directory)
            package_path = Path(spec.origin).parent

            # Quick heuristic: if it's in site-packages, skip it (non-editable)
            if "site-packages" in str(package_path):
                return 0

            # Scan all Python files in the package
            count = 0
            for py_file in package_path.rglob("*.py"):
                # Skip __pycache__ and test files for performance
                if "__pycache__" in str(py_file) or "test" in str(py_file):
                    continue
                count += self._scan_file_for_checkpoints(str(py_file), pattern)
                # Limit scanning for performance (if we find many, that's enough info)
                if count > 100:
                    return count

            return count
        except Exception:  # noqa: BLE001
            return 0

    def start_session(
        self,
        script_path: str,
        framework: str | None = None,
        clear_tensors: bool | None = None,
        coverage_mode: bool = False,
    ) -> DebuggerResponse:
        """
        Start a debugging session.

        Note: This doesn't actually run the script yet - it just prepares the debugger.
        The script starts executing when continue_execution() is first called.

        IMPORTANT: This automatically terminates any existing debug session first
        to prevent memory issues and ensure clean state.

        Args:
            script_path: Path to the script to debug
            framework: Optional framework identifier ("pytorch" or "mlx").
                      If None, will attempt to auto-detect from script path/name.
            clear_tensors: Whether to clear saved tensors. Default behavior:
                         - None/False: Keep tensors (default, useful for cross-framework comparison)
                         - True: Clear tensors before starting session
        """
        try:
            # ALWAYS terminate existing session first to prevent memory issues
            cleaned_up = False
            if hasattr(self, "debugger") and self.debugger is not None:
                try:
                    logger.info("Terminating existing debug session before starting new one")
                    self.debugger.terminate()
                    # Wait for cleanup to complete
                    import time  # noqa: PLC0415

                    time.sleep(2)
                    cleaned_up = True
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Error during cleanup: {e}")

            # Auto-detect framework from script path if not provided
            if framework is None:
                script_name_lower = Path(script_path).stem.lower()
                if "pytorch" in script_name_lower or "torch" in script_name_lower:
                    framework = "pytorch"
                elif "mlx" in script_name_lower:
                    framework = "mlx"
                else:
                    # Default to not archiving if we can't determine
                    framework = "unknown"

            # Clear existing debug tensors - default is False (keep tensors for cross-framework comparison)
            # Users can explicitly clear with clear_tensors=True if needed
            cleared_count = 0
            should_clear = clear_tensors

            # Determine default behavior - default to False (keep tensors)
            if should_clear is None:
                should_clear = False

            if should_clear:
                from mflux_debugger.tensor_debug import debug_clear

                try:
                    cleared_count = debug_clear(confirm=False)
                    if cleared_count > 0:
                        logger.info(f"🗑️  Cleared {cleared_count} tensor(s) from previous session")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to clear tensors: {e}")
            else:
                logger.debug(f"Skipping tensor clearing for framework: {framework} (tensors preserved for MLX to load)")

            # Archive existing images for the current framework when starting any session
            from mflux_debugger.image_archive import archive_images

            if framework and framework != "unknown":
                try:
                    archived_images = archive_images(framework)
                    if archived_images > 0:
                        logger.info(f"📦 Archived {archived_images} image(s) from previous {framework} session")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to archive images: {e}")

            # Create a fresh debugger instance for this session
            self.debugger = LightweightDebugger()

            # Enable coverage mode if requested
            if coverage_mode:
                self.debugger.coverage_mode = True
                # CRITICAL: Disable checkpoint breaking in coverage mode
                # We want full execution without pauses
                self.debugger.break_all_checkpoints = False
                logger.info("📊 Coverage tracking enabled (checkpoints will not pause execution)")

            # Register this debugger as the active instance for semantic checkpoints
            from mflux_debugger.lightweight_debugger import set_active_debugger
            from mflux_debugger.semantic_checkpoint import set_debugger_active

            set_active_debugger(self.debugger)
            set_debugger_active(True)
            # Enable recording of all checkpoints for hit counting
            self.debugger.set_record_all_checkpoints(True)
            logger.info("✅ Semantic checkpoints enabled and recording")

            # Clear Python cache BEFORE loading any code
            logger.info("🧹 Clearing Python cache to ensure fresh code")
            self._clear_python_cache()

            # Store the script path but don't start yet
            # We'll start when the user calls continue for the first time
            resolved_path = str(Path(script_path).resolve())

            # Validate script exists
            if not Path(resolved_path).exists():
                raise FileNotFoundError(f"❌ Script not found: {script_path}\nResolved to: {resolved_path}")

            self.debugger.script_path = resolved_path
            self.debugger.add_watch_file(resolved_path)

            # Scan for debug_checkpoint calls
            checkpoint_counts = self._count_debug_checkpoints(resolved_path)
            total_count = sum(checkpoint_counts.values())

            message = f"✅ Debug session started for `{Path(script_path).name}`\n\n"
            if cleaned_up:
                message += "🧹 Cleaned up previous session\n\n"
            if cleared_count > 0:
                message += f"🗑️  Cleared {cleared_count} tensor(s) from previous session\n\n"
            message += "🧹 Cleared Python cache for fresh code\n\n"
            message += "Session is initialized and ready.\n\n"

            if total_count > 0:
                message += f"🎯 Found ~{total_count} debug_checkpoint() call(s) in code:\n"
                # Show breakdown by source
                for source, count in checkpoint_counts.items():
                    if count > 0:
                        message += f"   • {source}: ~{count}\n"
                message += "   (Note: Actual hits depend on code path, conditionals, and skip parameter)\n\n"
                message += "📋 **Next steps:**\n"
                message += "   1. Run: `continue_execution()` - will pause at first checkpoint\n"
                message += "   2. When paused: `list_variables()`, `evaluate(expression)`\n"
                message += "   3. Continue: `continue_execution()` to next checkpoint\n"
                message += "   4. Or set line breakpoints: `set_breakpoint(file_path, line)`"
            else:
                message += "📋 **Next steps:**\n"
                message += "   1. Set breakpoints: `set_breakpoint(file_path, line)`\n"
                message += "   2. Run: `continue_execution()` - will pause at breakpoints\n"
                message += "   3. When paused: `list_variables()`, `evaluate(expression)`\n"
                message += "   4. Continue: `continue_execution()`, `step_over()`, `step_into()`\n"
                message += "   💡 TIP: Use debug_checkpoint() in code for semantic breakpoints!"

            return DebuggerResponse(
                success=True,
                message=message,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start session: {e}", exc_info=True)
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def set_breakpoint(self, file_path: str, line: int, condition: Optional[str] = None) -> DebuggerResponse:
        """Set a breakpoint with validation."""
        try:
            # Resolve to absolute path for consistency
            abs_path = str(Path(file_path).resolve())

            # VALIDATE: File must exist
            if not Path(abs_path).exists():
                return DebuggerResponse(
                    success=False,
                    message=f"❌ Cannot set breakpoint: File not found\n"
                    f"   Path: {file_path}\n"
                    f"   Resolved to: {abs_path}\n"
                    f"   Please check the file path.",
                    error="FileNotFoundError",
                )

            # VALIDATE: Check for duplicate breakpoint
            if (abs_path, line) in self.debugger.breakpoints:
                return DebuggerResponse(
                    success=False,
                    message=f"❌ Breakpoint already exists at this location\n"
                    f"   File: {Path(abs_path).name} (line {line})\n"
                    f"   Tip: Use list_breakpoints() to see all breakpoints\n"
                    f"   Or remove_breakpoint() to replace it",
                    error="DuplicateBreakpoint",
                )

            # VALIDATE: Warn if file is NOT in watch list (might never be executed)
            if abs_path not in self.debugger.watch_files:
                logger.warning(
                    f"⚠️  Setting breakpoint in file NOT being watched: {abs_path}\n"
                    f"   This breakpoint may never be hit if the file isn't imported/executed."
                )

            self.debugger.set_breakpoint(file_path, line, condition)

            # Show the actual line content for verification
            try:
                with open(abs_path) as f:
                    lines = f.readlines()
                    if line <= len(lines):
                        line_content = lines[line - 1].strip()[:80]
                    else:
                        line_content = "(line out of range)"
            except Exception:  # noqa: BLE001
                line_content = "(unable to read line)"

            # Build success message with warnings if needed
            message = (
                f"✅ Breakpoint set and validated\n"
                f"📍 File: `{Path(abs_path).name}` (line {line})\n"
                f"📝 Code: {line_content}"
            )

            if abs_path not in self.debugger.watch_files:
                message += (
                    "\n\n⚠️  WARNING: This file is not in the watch list.\n"
                    "   The breakpoint will only hit if this file is imported/executed."
                )

            return DebuggerResponse(
                success=True,
                message=message,
                data={"file": abs_path, "line": line, "condition": condition, "code": line_content},
            )
        except (ValueError, FileNotFoundError) as e:
            # User errors with helpful messages - pass through
            return DebuggerResponse(success=False, message=str(e), error=str(e))
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Unexpected error: {e}", error=str(e))

    def remove_breakpoint(self, file_path: str, line: int) -> DebuggerResponse:
        """Remove a breakpoint."""
        try:
            self.debugger.remove_breakpoint(file_path, line)
            return DebuggerResponse(
                success=True,
                message=f"✅ Breakpoint removed\n📍 File: `{file_path}` (line {line})",
                data={"file": file_path, "line": line},
            )
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def list_breakpoints(self) -> DebuggerResponse:
        """List all breakpoints."""
        try:
            breakpoints = self.debugger.list_breakpoints()
            if not breakpoints:
                return DebuggerResponse(
                    success=True,
                    message="No breakpoints set",
                    data={"breakpoints": []},
                )

            bp_list = []
            message_lines = ["📍 Active Breakpoints:\n"]
            for bp in breakpoints:
                bp_dict = {
                    "file": bp.file_path,
                    "line": bp.line,
                    "enabled": bp.enabled,
                    "condition": bp.condition,
                }
                bp_list.append(bp_dict)
                status = "✓" if bp.enabled else "✗"
                cond_str = f" (if {bp.condition})" if bp.condition else ""
                message_lines.append(f"  {status} {bp.file_path}:{bp.line}{cond_str}")

            return DebuggerResponse(
                success=True,
                message="\n".join(message_lines),
                data={"breakpoints": bp_list, "count": len(bp_list)},
            )
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def clear_all_breakpoints(self) -> DebuggerResponse:
        """Remove all breakpoints."""
        try:
            breakpoints = self.debugger.list_breakpoints()
            count = len(breakpoints)
            for bp in breakpoints:
                self.debugger.remove_breakpoint(bp.file_path, bp.line)

            return DebuggerResponse(
                success=True,
                message=f"✅ Cleared {count} breakpoint(s)",
                data={"cleared": count},
            )
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def continue_execution(self) -> DebuggerResponse:
        """Continue execution until next breakpoint."""
        try:
            # VALIDATE: Check script is set
            if not hasattr(self.debugger, "script_path") or not self.debugger.script_path:
                return DebuggerResponse(
                    success=False,
                    message="❌ No script loaded\n   Call start_session(script_path) first to load a script.",
                    error="no_script",
                )

            # WARN: No breakpoints set (execution will run to completion)
            if len(self.debugger.breakpoints) == 0:
                logger.warning(
                    "⚠️  Continuing without any breakpoints set!\n"
                    "   Execution will run to completion without pausing.\n"
                    "   Set breakpoints first if you want to inspect state."
                )

            # If this is the first continue call and script hasn't started yet, start it now
            if self.debugger.state == DebugState.NOT_STARTED and hasattr(self.debugger, "script_path"):
                self.debugger.start_script(self.debugger.script_path)
            else:
                self.debugger.continue_execution()

            if self.debugger.state == DebugState.PAUSED:
                # Get current state
                location_tuple = self.debugger.get_location()
                variables = self.debugger.list_variables()

                if location_tuple:
                    file_path, line, func_name = location_tuple
                    location_dict = {"file": file_path, "line": line, "function": func_name}

                    # Build response data
                    response_data = {
                        "location": location_dict,
                        "state": "paused",
                    }

                    # Add rich context if enabled
                    if self.enable_rich_context:
                        response_data["code_context"] = self._get_code_context(file_path, line, context_lines=3)
                        response_data["call_stack"] = self._get_call_stack_info()
                        response_data["variable_preview"] = self._smart_variable_preview(variables, limit=5)
                    else:
                        # Fallback to just variables
                        response_data["variables"] = variables

                    # Format message
                    var_lines = []
                    preview = response_data.get("variable_preview", {})
                    for name, info in list(preview.items())[:3]:  # Top 3 for message
                        if "shape" in info:
                            var_lines.append(f"  • {name}: {info['type']} {info['shape']}")
                        elif "value" in info:
                            var_lines.append(f"  • {name} = {info['value']}")
                        else:
                            var_lines.append(f"  • {name}: {info['type']}")

                    message = (
                        f"🛑 Stopped at breakpoint\n\n"
                        f"📍 Location: `{file_path}:{line}` in `{func_name}()`\n\n"
                        f"**Key variables:**\n" + "\n".join(var_lines)
                        if var_lines
                        else ""
                    )

                    return DebuggerResponse(
                        success=True,
                        message=message,
                        data=response_data,
                    )
                else:
                    return DebuggerResponse(success=True, message="⏸️ Program is paused")
            elif self.debugger.state == DebugState.FINISHED:
                return DebuggerResponse(success=True, message="✅ Program finished")
            else:
                return DebuggerResponse(success=True, message="▶️ Program is running")

        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def continue_execution_async(self) -> DebuggerResponse:
        """
        Continue execution in background (non-blocking for long ML operations).

        This method starts execution in a background thread and returns immediately.
        Use check_status() or get_location() to poll for when breakpoint is hit.

        Perfect for ML workloads with:
        - Heavy model loading (30+ seconds)
        - Long inference operations
        - Risk of HTTP timeouts

        Returns immediately with "running" state.
        """
        try:
            with self._thread_lock:
                # Check if already running
                if self._execution_thread and self._execution_thread.is_alive():
                    return DebuggerResponse(
                        success=False,
                        message="⚠️ Execution already running in background",
                        data={"state": "running"},
                    )

                # Track start time for timeout detection
                import time

                start_time = time.time()
                timeout = 600.0  # 10 minute default timeout

                # Start execution in background thread
                def _execute():
                    try:
                        if self.debugger.state == DebugState.NOT_STARTED and hasattr(self.debugger, "script_path"):
                            logger.info("🚀 Starting script in background thread")
                            self.debugger.start_script(self.debugger.script_path)
                        else:
                            logger.info("▶️  Continuing execution in background thread")
                            self.debugger.continue_execution()

                    except Exception as e:  # noqa: BLE001
                        logger.error(f"❌ Error in background execution: {e}", exc_info=True)

                self._execution_thread = threading.Thread(target=_execute, daemon=True)
                self._execution_thread.start()
                self._execution_start_time = start_time
                self._execution_timeout = timeout

                return DebuggerResponse(
                    success=True,
                    message=f"▶️ Execution started in background (timeout: {timeout}s)\n\n"
                    "💡 Use `debug_status()` or `debug_location()` to check if paused.",
                    data={"state": "running", "timeout": timeout},
                )

        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def step_over(self) -> DebuggerResponse:
        """Step over the current line."""
        try:
            self.debugger.step_over()
            return self._get_step_response(step_type="step_over")
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def step_into(self) -> DebuggerResponse:
        """Step into a function call."""
        try:
            self.debugger.step_into()
            return self._get_step_response(step_type="step_into")
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def step_out(self) -> DebuggerResponse:
        """Step out of the current function."""
        try:
            self.debugger.step_out()
            return self._get_step_response(step_type="step_out")
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def list_variables(self) -> DebuggerResponse:
        """List all variables in the current scope."""
        try:
            # VALIDATE: Can only list variables when paused
            if self.debugger.state != DebugState.PAUSED:
                return DebuggerResponse(
                    success=False,
                    message=f"❌ Cannot list variables: Debugger is not paused\n"
                    f"   Current state: {self.debugger.state.value}\n"
                    f"   You can only inspect variables when paused at a breakpoint.",
                    error="not_paused",
                )

            # Synchronize MLX before inspection (prevents Metal errors)
            self._sync_mlx_if_available()

            variables = self.debugger.list_variables()

            if not variables:
                return DebuggerResponse(
                    success=True,
                    message="ℹ️  No variables in current scope\n"
                    "   Tip: You might be at module level or before variable assignments.",
                    data={"variables": {}},
                )

            # Smart filtering and sneak peek approach
            filtered_vars = self._filter_and_preview_variables(variables)

            if not filtered_vars:
                return DebuggerResponse(
                    success=True,
                    message="ℹ️  No relevant variables to display (internal/large objects filtered)",
                    data={"variables": {}},
                )

            lines = ["**Variables in current scope:**"]
            for name, preview in filtered_vars.items():
                if isinstance(preview, dict) and "preview" in preview:
                    # Metadata preview
                    lines.append(f"  • **{name}**: {preview['preview']}")
                else:
                    # Simple serialized value
                    formatted = self._format_value(preview, max_length=100)
                    lines.append(f"  • **{name}**: {formatted}")

            return DebuggerResponse(
                success=True, message="\n".join(lines), data={"variables": self._make_serializable(filtered_vars)}
            )
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def _get_total_elements(self, tensor: Any) -> Optional[int]:
        """
        Get total number of elements in a tensor.

        Handles both PyTorch (where size is a method) and NumPy/MLX (where size is a property).

        Args:
            tensor: Tensor to get size from

        Returns:
            Total number of elements, or None if not available
        """
        try:
            if hasattr(tensor, "size"):
                # PyTorch: tensor.size() is a method that returns shape
                # NumPy/MLX: tensor.size is a property that returns total elements
                size_attr = tensor.size
                if callable(size_attr):
                    # PyTorch case: size() returns shape, so we need numel() instead
                    if hasattr(tensor, "numel"):
                        return int(tensor.numel())
                    # Fallback: compute from shape
                    shape = size_attr()
                    result = 1
                    for dim in shape:
                        result *= dim
                    return result
                else:
                    # NumPy/MLX case: size is already the total elements
                    return int(size_attr)
            elif hasattr(tensor, "shape"):
                # Fallback: compute from shape
                result = 1
                for dim in tensor.shape:
                    result *= dim
                return result
            return None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to get total elements: {e}")
            return None

    def _get_tensor_edges(self, tensor: Any, num_elements: int = 10) -> Dict[str, Any]:
        """
        Get first and last N elements for each dimension of a tensor.

        This provides a comprehensive view of tensor structure, useful for detecting
        issues like concatenation errors or unexpected padding.

        Args:
            tensor: Tensor to inspect (PyTorch, MLX, or NumPy)
            num_elements: Number of elements to show from start and end of each dim

        Returns:
            Dict with dimension-wise first/last elements
        """
        try:
            shape = tensor.shape
            ndim = len(shape)

            edges = {"shape": list(shape), "ndim": ndim, "dimensions": []}

            # For each dimension, show first and last N elements along that axis
            for dim_idx in range(ndim):
                dim_size = shape[dim_idx]

                # Create slice to get first N elements along this dimension
                first_slice = [slice(None)] * ndim
                first_slice[dim_idx] = slice(0, min(num_elements, dim_size))
                first_slice = tuple(first_slice)

                # Create slice to get last N elements along this dimension
                last_slice = [slice(None)] * ndim
                last_slice[dim_idx] = slice(max(0, dim_size - num_elements), dim_size)
                last_slice = tuple(last_slice)

                # Extract and flatten to 1D for display
                try:
                    first_values = tensor[first_slice]
                    last_values = tensor[last_slice]

                    # Flatten completely for display
                    if hasattr(first_values, "flatten"):
                        first_flat = first_values.flatten()[:10]  # Show max 10 values
                        last_flat = last_values.flatten()[:10]
                    else:
                        first_flat = first_values[:10] if hasattr(first_values, "__getitem__") else [first_values]
                        last_flat = last_values[:10] if hasattr(last_values, "__getitem__") else [last_values]

                    # Convert to serializable format
                    first_list = self._make_serializable(first_flat)
                    last_list = self._make_serializable(last_flat)

                    edges["dimensions"].append(
                        {"index": dim_idx, "size": dim_size, "first_10": first_list, "last_10": last_list}
                    )

                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to extract edges for dimension {dim_idx}: {e}")
                    edges["dimensions"].append({"index": dim_idx, "size": dim_size, "error": str(e)})

            return edges

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get tensor edges: {e}")
            return {"error": str(e)}

    def evaluate(self, expression: str) -> DebuggerResponse:
        """Evaluate a Python expression in the current context."""
        try:
            # Validate state before evaluation
            if self.debugger.state != DebugState.PAUSED:
                logger.warning(f"Evaluate called when not paused (state={self.debugger.state.value})")
                return DebuggerResponse(
                    success=False,
                    message=f"⚠️ Cannot evaluate - debugger not paused (state: {self.debugger.state.value})",
                    error="not_paused",
                )

            if not self.debugger.current_frame:
                logger.error("Evaluate called but no current frame available")
                return DebuggerResponse(
                    success=False, message="⚠️ Cannot evaluate - no execution frame available", error="no_frame"
                )

            # Synchronize MLX before evaluation (prevents Metal errors)
            self._sync_mlx_if_available()

            result = self.debugger.evaluate(expression)

            # Convert to serializable format for API response
            serializable_result = self._make_serializable(result)
            formatted = self._format_value(result, max_length=500)

            message = f"**Expression:** `{expression}`\n\n**Result:** {formatted}"
            # Return both formatted message and serializable result
            return DebuggerResponse(
                success=True,
                message=message,
                data={"expression": expression, "result": serializable_result, "formatted": formatted},
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Evaluation failed for '{expression}': {e}")
            logger.error(traceback.format_exc())
            return DebuggerResponse(
                success=False, message=f"❌ Evaluation failed: {e}\n\nExpression: `{expression}`", error=str(e)
            )

    def get_location(self) -> DebuggerResponse:
        """Get the current execution location with optional rich context."""
        try:
            location_tuple = self.debugger.get_location()
            if location_tuple:
                file_path, line, func_name = location_tuple
                location_dict = {"file": file_path, "line": line, "function": func_name}

                # Build response data
                response_data = {"location": location_dict}

                # Add checkpoint information if at a checkpoint
                if self.debugger.current_checkpoint:
                    checkpoint_info = self.debugger.current_checkpoint
                    checkpoint_data = {
                        "name": checkpoint_info.get("name"),
                        "hit_count": checkpoint_info.get("hit_count"),
                        "context": checkpoint_info.get("context", {}),
                    }

                    # Try to load checkpoint variables from JSON file
                    try:
                        checkpoint_vars = self._load_checkpoint_variables(
                            checkpoint_info.get("name"),
                            checkpoint_info.get("hit_count"),
                        )
                        if checkpoint_vars:
                            checkpoint_data["variables"] = checkpoint_vars
                    except Exception:  # noqa: BLE001
                        # If loading fails, continue without variables
                        pass

                    response_data["checkpoint"] = checkpoint_data

                # Add rich context if enabled
                if self.enable_rich_context:
                    variables = self.debugger.list_variables()
                    response_data["code_context"] = self._get_code_context(file_path, line, context_lines=3)
                    response_data["call_stack"] = self._get_call_stack_info()
                    response_data["variable_preview"] = self._smart_variable_preview(variables, limit=5)

                message = f"📍 Current location: `{file_path}:{line}` in `{func_name}()`"

                # Add checkpoint info to message if available
                if self.debugger.current_checkpoint:
                    cp_name = self.debugger.current_checkpoint.get("name")
                    cp_hit = self.debugger.current_checkpoint.get("hit_count")
                    message += f"\n🎯 At checkpoint: '{cp_name}' (hit #{cp_hit})"

                return DebuggerResponse(success=True, message=message, data=response_data)
            else:
                return DebuggerResponse(success=False, message="❌ No current location")
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def _load_checkpoint_variables(self, checkpoint_name: str, hit_count: int) -> dict | None:
        """
        Load checkpoint variables from JSON file.

        Args:
            checkpoint_name: Name of the checkpoint
            hit_count: Hit count for the checkpoint

        Returns:
            Dictionary of checkpoint variables, or None if not found
        """
        try:
            import json

            from mflux_debugger.log_paths import get_runs_archive_dir, get_runs_latest_dir

            # Find the checkpoint JSON file
            # Pattern: checkpoint_{name}_hit{count:03d}.json
            safe_name = checkpoint_name.replace(":", "_").replace("/", "_")
            checkpoint_filename = f"checkpoint_{safe_name}_hit{hit_count:03d}.json"

            # Search in latest runs directory first
            # Checkpoints can be in any module, so search all recent sessions
            latest_dir = get_runs_latest_dir()
            if latest_dir.exists():
                # Get all session directories, sorted by modification time (most recent first)
                session_dirs = sorted(
                    [d for d in latest_dir.iterdir() if d.is_dir()],
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )

                # Search through recent sessions (check up to 5 most recent)
                for session_dir in session_dirs[:5]:
                    checkpoints_dir = session_dir / "checkpoints"
                    if not checkpoints_dir.exists():
                        continue
                    checkpoint_file = checkpoints_dir / checkpoint_filename
                    if checkpoint_file.exists():
                        with open(checkpoint_file) as f:
                            checkpoint_data = json.load(f)
                            return checkpoint_data.get("variables", {})

            # If not found, search archive directory (most recent session only)
            archive_dir = get_runs_archive_dir()
            if archive_dir.exists():
                session_dirs = sorted(
                    [d for d in archive_dir.iterdir() if d.is_dir()],
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )

                # Only check most recent archived session
                for session_dir in session_dirs[:1]:
                    checkpoints_dir = session_dir / "checkpoints"
                    if not checkpoints_dir.exists():
                        continue
                    checkpoint_file = checkpoints_dir / checkpoint_filename
                    if checkpoint_file.exists():
                        with open(checkpoint_file) as f:
                            checkpoint_data = json.load(f)
                            return checkpoint_data.get("variables", {})

            return None
        except Exception:  # noqa: BLE001
            return None

    def check_status(self) -> DebuggerResponse:
        """Check the current debugger status with timeout detection."""
        state = self.debugger.state
        state_messages = {
            DebugState.NOT_STARTED: "❌ Debug session not started",
            DebugState.RUNNING: "▶️ Program is running",
            DebugState.PAUSED: "⏸️ Program is stopped",
            DebugState.FINISHED: "✅ Program finished",
            DebugState.FAILED: "❌ Script failed",
        }
        message = state_messages.get(state, f"Unknown state: {state}")

        # Build data dict
        data = {"state": state.value}

        # Add timeout info for running state
        if state == DebugState.RUNNING and self._execution_start_time and self._execution_timeout:
            import time

            elapsed = time.time() - self._execution_start_time
            remaining = self._execution_timeout - elapsed
            data["elapsed_seconds"] = round(elapsed, 1)
            data["remaining_seconds"] = round(remaining, 1)

            if remaining < 0:
                message += f"\n⚠️  WARNING: Execution exceeded timeout by {abs(round(remaining, 1))}s - may be hung!"
                data["timeout_exceeded"] = True
            elif remaining < 60:
                message += f"\n⏱️  {round(remaining, 1)}s remaining before timeout"

        # Add error details for failed state
        if state == DebugState.FAILED:
            data["exit_code"] = self.debugger._exit_code

            if self.debugger._last_exception:
                error_type = type(self.debugger._last_exception).__name__
                error_msg = str(self.debugger._last_exception)
                data["error"] = {
                    "type": error_type,
                    "message": error_msg,
                    "traceback": self.debugger._exception_traceback,
                }
                message = f"❌ Script failed with error: {error_type}: {error_msg}"

            if self.debugger._stdout_capture:
                data["stdout_lines"] = len(self.debugger._stdout_capture)
            if self.debugger._stderr_capture:
                data["stderr_lines"] = len(self.debugger._stderr_capture)

            return DebuggerResponse(success=False, message=message, data=data)

        # Add output info for finished state
        if state == DebugState.FINISHED:
            data["exit_code"] = self.debugger._exit_code
            if self.debugger._stdout_capture:
                data["stdout_lines"] = len(self.debugger._stdout_capture)
            if self.debugger._stderr_capture:
                data["stderr_lines"] = len(self.debugger._stderr_capture)

        # Add checkpoint info if paused at a checkpoint
        if state == DebugState.PAUSED:
            checkpoint_info = self.debugger.get_current_checkpoint()
            if checkpoint_info:
                data["checkpoint"] = checkpoint_info

        return DebuggerResponse(success=True, message=message, data=data)

    def terminate(self) -> DebuggerResponse:
        """Terminate the debugging session and save trace."""
        try:
            self.debugger.terminate()

            # Disable semantic checkpoints and clear active debugger
            from mflux_debugger.lightweight_debugger import set_active_debugger
            from mflux_debugger.semantic_checkpoint import set_debugger_active

            set_active_debugger(None)
            set_debugger_active(False)

            # Build termination summary
            state = self.debugger.state
            message_parts = ["✅ Debug session terminated and subprocess cleaned up."]

            # Add state and exit code
            message_parts.append(f"\nState: {state.value}")
            message_parts.append(f"Exit Code: {self.debugger._exit_code}")

            # Note: Removed breakpoint hit tracking (was trace_recorder based)

            # Add error info if failed
            if state == DebugState.FAILED and self.debugger._last_exception:
                error_type = type(self.debugger._last_exception).__name__
                error_msg = str(self.debugger._last_exception)
                message_parts.append(f"\n❌ Error: {error_type}: {error_msg}")

            # Add output capture info
            if self.debugger._stdout_capture:
                message_parts.append(f"stdout: {len(self.debugger._stdout_capture)} lines")
            if self.debugger._stderr_capture:
                message_parts.append(f"stderr: {len(self.debugger._stderr_capture)} lines")

            # Clear Python cache to ensure fresh code on next debug session
            self._clear_python_cache()
            message_parts.append("🧹 Cleared Python cache for next debug session.")

            message = "\n".join(message_parts)

            return DebuggerResponse(success=True, message=message)
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def _clear_python_cache(self):
        """Clear Python __pycache__ directories and .pyc files."""
        try:
            import subprocess  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            # Get workspace root (assuming we're in src/mflux_debugger/)
            workspace_root = Path(__file__).parent.parent.parent

            # Clear __pycache__ directories
            subprocess.run(
                [
                    "find",
                    str(workspace_root / "src"),
                    "-type",
                    "d",
                    "-name",
                    "__pycache__",
                    "-exec",
                    "rm",
                    "-rf",
                    "{}",
                    "+",
                ],
                stderr=subprocess.DEVNULL,
                check=False,
            )

            # Clear .pyc files
            subprocess.run(
                ["find", str(workspace_root / "src"), "-name", "*.pyc", "-delete"],
                stderr=subprocess.DEVNULL,
                check=False,
            )

            logger.info("Cleared Python cache after debug session")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to clear Python cache: {e}")

    # Helper methods

    def _get_code_context(self, file_path: str, line: int, context_lines: int = 3) -> Dict[str, Any]:
        """Get code context around a specific line.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            context_lines: Number of lines to show before/after

        Returns:
            Dict with before, current, and after lines
        """
        try:
            with open(file_path) as f:
                lines = f.readlines()

            # Convert to 0-indexed
            idx = line - 1

            # Get context
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)

            before = [(i + 1, lines[i].rstrip()) for i in range(start, idx)]
            current = (line, lines[idx].rstrip()) if idx < len(lines) else (line, "")
            after = [(i + 1, lines[i].rstrip()) for i in range(idx + 1, end)]

            return {
                "before": before,
                "current": current,
                "after": after,
            }
        except Exception:  # noqa: BLE001
            return {"before": [], "current": (line, ""), "after": []}

    def _get_call_stack_info(self) -> list[Dict[str, Any]]:
        """Get formatted call stack information.

        Returns:
            List of stack frames with file, line, function
        """
        try:
            frames = self.debugger.get_stack_trace()
            return [
                {
                    "file": frame.file_path,
                    "line": frame.line,
                    "function": frame.function_name,
                }
                for frame in frames
            ]
        except Exception:  # noqa: BLE001
            return []

    def _filter_and_preview_variables(self, variables: Dict[str, Any], max_vars: int = 50) -> Dict[str, Any]:
        """
        Filter and create sneak peek previews of variables.

        Smart filtering:
        - Skip internal variables (_name)
        - Skip very large objects (>10MB estimated)
        - Skip modules, functions, classes
        - Prioritize ML arrays and important data

        Sneak peek approach:
        - Small values: serialize fully
        - Arrays: show metadata + sample values
        - Large objects: show type and size only

        Args:
            variables: Dict of variable name -> value
            max_vars: Maximum variables to return

        Returns:
            Dict of variable name -> preview (metadata or serialized value)
        """
        import sys

        filtered = {}

        for name, value in variables.items():
            # Skip if we've hit the limit
            if len(filtered) >= max_vars:
                break

            # Filter: Skip internal/dunder variables
            if name.startswith("_"):
                continue

            # Filter: Skip modules, functions, classes, types
            if isinstance(value, type) or callable(value):
                value_type = type(value).__name__
                if value_type in ("module", "function", "method", "type", "builtin_function_or_method"):
                    continue

            try:
                # Estimate size (rough)
                size_estimate = sys.getsizeof(value)

                # Check for tensor/array types
                is_tensor = hasattr(value, "shape") and hasattr(value, "dtype")

                if is_tensor:
                    # Tensor sneak peek: metadata + sample values
                    try:
                        shape = list(value.shape) if hasattr(value.shape, "__iter__") else [value.shape]
                        dtype = str(value.dtype) if hasattr(value, "dtype") else "unknown"
                        total_elements = 1
                        for dim in shape:
                            total_elements *= dim

                        # Small tensors: serialize first few elements
                        preview_str = f"{type(value).__name__}(shape={shape}, dtype={dtype})"
                        preview_dict = {
                            "type": type(value).__name__,
                            "shape": shape,
                            "dtype": dtype,
                            "size": total_elements,
                            "preview": preview_str,
                        }

                        # Add sample values for small arrays only
                        if total_elements <= 100:  # Small enough to sample
                            try:
                                # Try to get first few values
                                sample = value.flatten()[:5] if hasattr(value, "flatten") else value[:5]
                                sample_list = self._make_serializable(sample)
                                if sample_list and len(sample_list) > 0:
                                    preview_dict["sample_values"] = sample_list
                                    preview_dict["preview"] = preview_str + f", samples={sample_list[:3]}..."
                            except Exception:  # noqa: BLE001
                                pass  # Skip samples if they fail

                        filtered[name] = preview_dict

                    except Exception:  # noqa: BLE001
                        # Fallback to simple type
                        filtered[name] = {"type": type(value).__name__, "preview": type(value).__name__}

                # Small simple values: serialize fully
                elif isinstance(value, (int, float, str, bool, type(None))):
                    filtered[name] = value

                # Lists/tuples: show length + type
                elif isinstance(value, (list, tuple)):
                    list_len = len(value)
                    if list_len == 0:
                        filtered[name] = value  # Empty, just serialize
                    elif list_len <= 5:
                        # Small list, serialize it
                        filtered[name] = self._make_serializable(value)
                    else:
                        # Large list, show preview
                        filtered[name] = {
                            "type": type(value).__name__,
                            "length": list_len,
                            "preview": f"{type(value).__name__}(length={list_len})",
                        }

                # Dicts: show key count
                elif isinstance(value, dict):
                    dict_len = len(value)
                    if dict_len == 0:
                        filtered[name] = value
                    elif dict_len <= 3:
                        # Small dict, serialize it
                        filtered[name] = self._make_serializable(value)
                    else:
                        # Large dict, show preview
                        filtered[name] = {
                            "type": "dict",
                            "keys": list(value.keys())[:5],
                            "length": dict_len,
                            "preview": f"dict({dict_len} keys)",
                        }

                # Everything else: type name only
                else:
                    # Filter out very large objects
                    if size_estimate > 10_000_000:  # >10MB
                        continue

                    filtered[name] = {
                        "type": type(value).__name__,
                        "preview": type(value).__name__,
                    }

            except Exception:  # noqa: BLE001
                # If inspection fails, skip this variable
                continue

        return filtered

    def _smart_variable_preview(self, variables: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
        """Create smart preview of variables with focus on ML arrays.

        Args:
            variables: Dict of variable name -> value
            limit: Maximum number of variables to preview

        Returns:
            Dict of variable name -> preview info
        """
        preview = {}
        count = 0

        for name, value in variables.items():
            if count >= limit:
                break

            # Skip internal/dunder variables
            if name.startswith("_"):
                continue

            var_info = {"type": type(value).__name__}

            # Special handling for ML arrays
            try:
                # Check for MLX array
                if hasattr(value, "shape") and hasattr(value, "dtype"):
                    var_info["shape"] = list(value.shape) if hasattr(value.shape, "__iter__") else [value.shape]
                    var_info["dtype"] = str(value.dtype)

                    # Try to get basic stats and sample values (will trigger MLX eval if needed)
                    try:
                        import mlx.core as mx  # noqa: PLC0415

                        if isinstance(value, mx.array):
                            # Already evaluated by auto_eval_mlx
                            var_info["device"] = "mps"  # MLX always uses Metal

                            # Add sample values: first 10 values along each dimension
                            try:
                                shape = value.shape
                                if len(shape) > 0:
                                    # Get first 10 values along first dimension
                                    sample_size = min(10, shape[0])
                                    if len(shape) == 1:
                                        # 1D: first 10 values - slice first, then evaluate
                                        sample_slice = value[:sample_size]
                                        var_info["sample"] = mx.eval(sample_slice.astype(mx.float32)).tolist()
                                    elif len(shape) == 2:
                                        # 2D: first 10 rows, first 10 values of each
                                        sample_rows = min(10, shape[0])
                                        sample_cols = min(10, shape[1])
                                        sample_slice = value[:sample_rows, :sample_cols]
                                        var_info["sample"] = mx.eval(sample_slice.astype(mx.float32)).tolist()
                                    elif len(shape) == 3:
                                        # 3D: first element, first 10 rows, first 10 values
                                        sample_rows = min(10, shape[1])
                                        sample_cols = min(10, shape[2])
                                        sample_slice = value[0, :sample_rows, :sample_cols]
                                        var_info["sample"] = mx.eval(sample_slice.astype(mx.float32)).tolist()
                                    elif len(shape) >= 4:
                                        # 4D+: first element along all but last 2 dims, then sample
                                        sample_rows = min(10, shape[-2])
                                        sample_cols = min(10, shape[-1])
                                        # Flatten leading dimensions and take first element
                                        indices = tuple(
                                            [0] * (len(shape) - 2) + [slice(sample_rows), slice(sample_cols)]
                                        )
                                        sample_slice = value[indices]
                                        var_info["sample"] = mx.eval(sample_slice.astype(mx.float32)).tolist()
                            except Exception as e:  # noqa: BLE001
                                # If sampling fails, continue without sample
                                logger.debug(f"Failed to extract sample values: {e}")
                                pass
                    except ImportError:
                        pass

                    # Also handle PyTorch tensors
                    try:
                        import torch  # noqa: PLC0415

                        if isinstance(value, torch.Tensor):
                            var_info["device"] = str(value.device)

                            # Add sample values for PyTorch tensors
                            try:
                                shape = value.shape
                                if len(shape) > 0:
                                    sample_size = min(10, shape[0])
                                    if len(shape) == 1:
                                        var_info["sample"] = value[:sample_size].detach().cpu().tolist()
                                    elif len(shape) == 2:
                                        sample_rows = min(10, shape[0])
                                        sample_cols = min(10, shape[1])
                                        var_info["sample"] = value[:sample_rows, :sample_cols].detach().cpu().tolist()
                                    elif len(shape) == 3:
                                        sample_rows = min(10, shape[1])
                                        sample_cols = min(10, shape[2])
                                        var_info["sample"] = (
                                            value[0, :sample_rows, :sample_cols].detach().cpu().tolist()
                                        )
                                    elif len(shape) >= 4:
                                        sample_rows = min(10, shape[-2])
                                        sample_cols = min(10, shape[-1])
                                        indices = tuple(
                                            [0] * (len(shape) - 2) + [slice(sample_rows), slice(sample_cols)]
                                        )
                                        var_info["sample"] = value[indices].detach().cpu().tolist()
                            except Exception:  # noqa: BLE001
                                pass
                    except ImportError:
                        pass

                # For scalars, show the value
                elif isinstance(value, (int, float, str, bool)):
                    var_info["value"] = value

                # For sequences, show length
                elif isinstance(value, (list, tuple)):
                    var_info["length"] = len(value)

            except Exception:  # noqa: BLE001
                # If inspection fails, just keep the type
                pass

            preview[name] = var_info
            count += 1

        return preview

    def _get_step_response(self, step_type: str = "step") -> DebuggerResponse:
        """Get response after a step operation."""
        if self.debugger.state == DebugState.PAUSED:
            location_tuple = self.debugger.get_location()
            variables = self.debugger.list_variables()

            if location_tuple:
                file_path, line, func_name = location_tuple
                location_dict = {"file": file_path, "line": line, "function": func_name}

                # Build response data with rich context
                response_data = {
                    "location": location_dict,
                    "state": "paused",
                }

                if self.enable_rich_context:
                    response_data["code_context"] = self._get_code_context(file_path, line, context_lines=3)
                    response_data["call_stack"] = self._get_call_stack_info()
                    response_data["variable_preview"] = self._smart_variable_preview(variables, limit=5)
                else:
                    response_data["variables"] = variables

                message = f"👣 Stepped\n\n📍 Location: `{file_path}:{line}` in `{func_name}()`"

                return DebuggerResponse(
                    success=True,
                    message=message,
                    data=response_data,
                )
            else:
                return DebuggerResponse(success=True, message="⏸️ Program is paused")
        elif self.debugger.state == DebugState.FINISHED:
            return DebuggerResponse(success=True, message="✅ Program finished")
        else:
            return DebuggerResponse(success=True, message="▶️ Program is running")

    def _sync_mlx_if_available(self, timeout: float = 10.0) -> bool:
        """
        Synchronize MLX GPU operations before inspection.

        This prevents Metal command buffer errors by ensuring all GPU operations
        complete before we inspect tensors. The agent is willing to wait for this.

        Args:
            timeout: Maximum seconds to wait for sync (default: 10s)

        Returns:
            True if sync succeeded or MLX not available, False if timeout
        """
        try:
            import time  # noqa: PLC0415

            import mlx.core as mx  # noqa: PLC0415

            logger.info("Synchronizing MLX GPU operations before inspection...")
            start_time = time.time()

            # Wait for all GPU operations to complete
            mx.synchronize()

            # Small additional sleep to let Metal cleanup
            time.sleep(0.1)

            elapsed = time.time() - start_time
            logger.info(f"MLX sync completed in {elapsed:.3f}s")
            return True

        except ImportError:
            # MLX not available, nothing to sync
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MLX sync failed (continuing anyway): {e}")
            # Continue even if sync fails - better to try inspection than fail completely
            return False

    def _make_serializable(self, value: Any) -> Any:
        """
        Convert a value to a JSON-serializable format.

        Handles MLX arrays, PyTorch tensors, NumPy arrays, etc.
        """
        try:
            # Handle MLX arrays - convert to list with Metal error protection
            try:
                import mlx.core as mx  # noqa: PLC0415

                if isinstance(value, mx.array):
                    try:
                        # Ensure evaluated - wrap in try/catch for Metal errors
                        mx.eval(value)

                        # For large arrays, only serialize preview (at least 10 values)
                        shape = value.shape if hasattr(value, "shape") else None
                        total_elements = 1
                        if shape:
                            for dim in shape:
                                total_elements *= dim

                        # If array is large (>100 elements), serialize preview only
                        if total_elements > 100:
                            # Get first 10 and last 10 elements (flattened)
                            flat = value.flatten()
                            first_10 = flat[:10].tolist()
                            last_10 = flat[-10:].tolist() if len(flat) > 10 else []

                            return {
                                "type": "mlx.array",
                                "shape": list(shape) if shape else None,
                                "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
                                "preview_first": first_10,
                                "preview_last": last_10,
                                "total_elements": int(total_elements),
                            }

                        # Small arrays: convert fully
                        return value.tolist()
                    except (RuntimeError, AssertionError) as metal_error:
                        # Metal/GPU errors during MLX operations
                        error_msg = str(metal_error)
                        if "MTL" in error_msg or "Metal" in error_msg or "Completed handler" in error_msg:
                            logger.warning(f"MLX/Metal error during serialization: {metal_error}")
                            # Return shape info instead of crashing
                            return {
                                "type": "mlx.array",
                                "shape": list(value.shape) if hasattr(value, "shape") else "unknown",
                                "dtype": str(value.dtype) if hasattr(value, "dtype") else "unknown",
                                "error": "Metal GPU error - cannot serialize value",
                            }
                        else:
                            raise  # Re-raise if not a Metal error
            except (ImportError, AttributeError):
                pass

            # Handle PyTorch tensors
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                # Check if it's a complex dtype
                dtype_str = str(getattr(value, "dtype", ""))
                if "complex" in dtype_str:
                    # Convert complex tensor to dict with real/imag parts
                    cpu_val = value.detach().cpu()
                    arr_list = cpu_val.tolist()

                    # Handle nested lists (multi-dimensional tensors)
                    def serialize_complex_list(lst):
                        if isinstance(lst, list):
                            return [serialize_complex_list(item) for item in lst]
                        elif isinstance(lst, complex):
                            return {"real": lst.real, "imag": lst.imag}
                        else:
                            return lst

                    return {
                        "type": "complex_tensor",
                        "shape": list(getattr(cpu_val, "shape", [])),
                        "dtype": dtype_str,
                        "values": serialize_complex_list(arr_list)
                        if isinstance(arr_list, list) and len(arr_list) <= 10
                        else f"<{len(arr_list)} items>",  # Only serialize small tensors
                    }

                # For large tensors, only serialize preview (at least 10 values)
                cpu_val = value.detach().cpu()
                shape = list(cpu_val.shape) if hasattr(cpu_val, "shape") else None
                total_elements = 1
                if shape:
                    for dim in shape:
                        total_elements *= dim

                # If tensor is large (>100 elements), serialize preview only
                if total_elements > 100:
                    # Get first 10 and last 10 elements (flattened)
                    flat = cpu_val.flatten()
                    first_10 = flat[:10].tolist()
                    last_10 = flat[-10:].tolist() if len(flat) > 10 else []

                    return {
                        "type": "torch.Tensor",
                        "shape": shape,
                        "dtype": dtype_str,
                        "preview_first": first_10,
                        "preview_last": last_10,
                        "total_elements": int(total_elements),
                    }

                # Small tensors: convert fully
                return cpu_val.tolist()

            # Handle NumPy arrays
            if hasattr(value, "tolist") and hasattr(value, "dtype"):
                # Check if it's a complex dtype
                dtype_str = str(getattr(value, "dtype", ""))
                if "complex" in dtype_str:
                    # Convert complex array to dict with real/imag parts
                    arr_list = value.tolist()
                    return {
                        "type": "complex_array",
                        "shape": list(getattr(value, "shape", [])),
                        "dtype": dtype_str,
                        "values": [
                            {"real": v.real, "imag": v.imag} if isinstance(v, complex) else v
                            for v in (arr_list if isinstance(arr_list, list) else [arr_list])[:10]
                        ],  # Show first 10
                    }

                # For large arrays, only serialize preview (at least 10 values)
                shape = list(value.shape) if hasattr(value, "shape") else None
                total_elements = 1
                if shape:
                    for dim in shape:
                        total_elements *= dim

                # If array is large (>100 elements), serialize preview only
                if total_elements > 100:
                    # Get first 10 and last 10 elements (flattened)
                    flat = value.flatten()
                    first_10 = flat[:10].tolist()
                    last_10 = flat[-10:].tolist() if len(flat) > 10 else []

                    return {
                        "type": "numpy.ndarray",
                        "shape": shape,
                        "dtype": dtype_str,
                        "preview_first": first_10,
                        "preview_last": last_10,
                        "total_elements": int(total_elements),
                    }

                # Small arrays: convert fully
                return value.tolist()

            # Handle lists/tuples recursively
            if isinstance(value, (list, tuple)):
                return [self._make_serializable(x) for x in value]

            # Handle dicts recursively
            if isinstance(value, dict):
                return {k: self._make_serializable(v) for k, v in value.items()}

            # Primitives are already serializable
            if isinstance(value, (int, float, str, bool, type(None))):
                return value

            # Handle complex numbers
            if isinstance(value, complex):
                return {"real": value.real, "imag": value.imag, "type": "complex"}

            # Fall back to string representation
            return str(value)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to serialize value: {e}")
            return f"<unserializable: {type(value).__name__}>"

    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format a variable value for display."""
        try:
            # Special handling for tensors
            if hasattr(value, "shape"):  # NumPy, PyTorch, MLX arrays
                type_name = type(value).__name__
                shape = getattr(value, "shape", "?")
                dtype = getattr(value, "dtype", "?")

                parts = [f"{type_name}(shape={shape}, dtype={dtype}"]

                # Try to get mean if possible
                try:
                    if hasattr(value, "mean"):
                        mean = value.mean()
                        parts.append(f"mean={mean:.6f}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to compute mean for {type_name}: {e}")
                    parts.append("mean=<error>")

                # Show first 10 values for better debugging
                try:
                    # Flatten and get first 10 elements
                    flat = value.flatten() if hasattr(value, "flatten") else value.reshape(-1)

                    # Handle MLX arrays
                    try:
                        import mlx.core as mx  # noqa: PLC0415

                        if isinstance(flat, mx.array):
                            mx.eval(flat)  # Ensure evaluated
                    except ImportError:
                        pass

                    # Get first 10 elements
                    num_show = min(10, len(flat) if hasattr(flat, "__len__") else flat.shape[0])
                    if num_show > 0:
                        sample = flat[:num_show]
                        # Convert to list for display
                        if hasattr(sample, "tolist"):
                            sample_list = sample.tolist()
                        elif hasattr(sample, "item") and num_show == 1:
                            sample_list = [sample.item()]
                        else:
                            sample_list = [float(x) for x in sample]

                        # Format with limited precision
                        sample_str = ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in sample_list)
                        parts.append(f"values=[{sample_str}...]")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to get sample values for {type_name}: {e}")
                    # If we can't get samples, that's okay - continue without them

                return ", ".join(parts) + ")"

            # Regular values
            str_value = str(value)
            if len(str_value) > max_length:
                return str_value[:max_length] + "..."
            return str_value
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error formatting value of type {type(value)}: {e}")
            return f"<error formatting value: {type(value).__name__}: {e}>"

    def set_checkpoint_breakpoint(self, checkpoint_name: str, description: str = "") -> DebuggerResponse:
        """
        Set a semantic checkpoint breakpoint.

        Args:
            checkpoint_name: Name of the checkpoint to break at
            description: Optional description of the checkpoint

        Returns:
            DebuggerResponse with success status
        """
        try:
            self.debugger.set_checkpoint_breakpoint(checkpoint_name, description)
            return DebuggerResponse(
                success=True,
                message=f"Checkpoint breakpoint set: {checkpoint_name}",
                data={"checkpoint_name": checkpoint_name, "description": description},
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to set checkpoint breakpoint: {e}\n{traceback.format_exc()}")
            return DebuggerResponse(success=False, message="Failed to set checkpoint breakpoint", error=str(e))

    def remove_checkpoint_breakpoint(self, checkpoint_name: str) -> DebuggerResponse:
        """
        Remove a semantic checkpoint breakpoint.

        Args:
            checkpoint_name: Name of the checkpoint to remove

        Returns:
            DebuggerResponse with success status
        """
        try:
            self.debugger.remove_checkpoint_breakpoint(checkpoint_name)
            return DebuggerResponse(
                success=True,
                message=f"Checkpoint breakpoint removed: {checkpoint_name}",
                data={"checkpoint_name": checkpoint_name},
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to remove checkpoint breakpoint: {e}")
            return DebuggerResponse(success=False, message="Failed to remove checkpoint breakpoint", error=str(e))

    def set_checkpoint_record_all(self, enabled: bool) -> DebuggerResponse:
        """
        Enable or disable recording of ALL semantic checkpoints.

        Args:
            enabled: True to record all checkpoints, False to only record breakpointed ones

        Returns:
            DebuggerResponse with success status
        """
        try:
            self.debugger.set_record_all_checkpoints(enabled)
            mode = "enabled" if enabled else "disabled"
            return DebuggerResponse(
                success=True,
                message=f"Record-all checkpoints {mode}",
                data={"enabled": enabled},
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to set record-all mode: {e}")
            return DebuggerResponse(success=False, message="Failed to set record-all mode", error=str(e))

    def set_checkpoint_break_all(self, enabled: bool) -> DebuggerResponse:
        """
        Enable or disable breaking at ALL semantic checkpoints.

        Args:
            enabled: True to break at all checkpoints, False to only break at explicit ones

        Returns:
            DebuggerResponse with success status
        """
        try:
            self.debugger.set_break_all_checkpoints(enabled)
            mode = "enabled" if enabled else "disabled"
            return DebuggerResponse(
                success=True,
                message=f"Break-all checkpoints {mode}",
                data={"enabled": enabled},
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to set break-all mode: {e}")
            return DebuggerResponse(success=False, message="Failed to set break-all mode", error=str(e))

    def list_checkpoint_breakpoints(self) -> DebuggerResponse:
        """
        List all semantic checkpoint breakpoints.

        Returns:
            DebuggerResponse with checkpoint breakpoint list
        """
        try:
            checkpoints = self.debugger.list_checkpoint_breakpoints()
            checkpoint_list = [{"name": name, "enabled": bp.enabled} for name, bp in checkpoints.items()]
            return DebuggerResponse(
                success=True,
                message=f"Found {len(checkpoint_list)} checkpoint breakpoint(s)",
                data={"checkpoints": checkpoint_list},
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to list checkpoint breakpoints: {e}")
            return DebuggerResponse(success=False, message="Failed to list checkpoint breakpoints", error=str(e))

    def get_current_checkpoint(self) -> DebuggerResponse:
        """
        Get information about the current checkpoint if paused at one.

        Returns:
            DebuggerResponse with current checkpoint info
        """
        try:
            checkpoint_info = self.debugger.get_current_checkpoint()
            if checkpoint_info:
                return DebuggerResponse(success=True, message="At checkpoint", data={"checkpoint": checkpoint_info})
            else:
                return DebuggerResponse(success=True, message="Not at checkpoint", data={"checkpoint": None})
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to get current checkpoint: {e}")
            return DebuggerResponse(success=False, message="Failed to get current checkpoint", error=str(e))

    def get_checkpoint_history(self) -> DebuggerResponse:
        """
        Get history of all checkpoints hit in this session.

        Returns:
            DebuggerResponse with checkpoint history
        """
        try:
            history = self.debugger.get_checkpoint_history()
            return DebuggerResponse(
                success=True, message=f"Found {len(history)} checkpoint hit(s)", data={"history": history}
            )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to get checkpoint history: {e}")
            return DebuggerResponse(success=False, message="Failed to get checkpoint history", error=str(e))

    def get_coverage(self) -> DebuggerResponse:
        """
        Get coverage data from the debugger.

        Returns:
            DebuggerResponse with coverage data (file -> list of executed line numbers)
        """
        try:
            coverage_data = self.debugger.get_coverage_data()
            if coverage_data is None:
                return DebuggerResponse(
                    success=False,
                    message="Coverage data not available (coverage mode not enabled)",
                    error="Coverage mode not enabled",
                )

            # Convert sets to lists for JSON serialization
            coverage_dict = {file: sorted(lines) for file, lines in coverage_data.items()}

            return DebuggerResponse(
                success=True,
                message=f"Coverage data for {len(coverage_dict)} file(s)",
                data={"coverage_data": coverage_dict},
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get coverage data: {e}")
            return DebuggerResponse(success=False, message="Failed to get coverage data", error=str(e))

    def get_checkpoint_verification_status(self) -> DebuggerResponse:
        """
        Get checkpoint verification status including hit counts and order.

        Returns:
            DebuggerResponse with verification information
        """
        try:
            data = {
                "hit_counts": dict(self.debugger.checkpoint_hit_counts),
                "checkpoint_order": list(self.debugger.checkpoint_order),
                "current_checkpoint": self.debugger.current_checkpoint,
            }

            # Add summary
            total_hits = sum(self.debugger.checkpoint_hit_counts.values())
            unique_checkpoints = len(self.debugger.checkpoint_hit_counts)
            message = f"📊 Verification Status: {total_hits} total hits across {unique_checkpoints} unique checkpoints"

            return DebuggerResponse(success=True, message=message, data=data)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to get verification status: {e}")
            return DebuggerResponse(success=False, message="Failed to get verification status", error=str(e))

    def set_conditional_checkpoint_breakpoint(
        self, checkpoint_name: str, context_condition: Dict[str, Any]
    ) -> DebuggerResponse:
        """
        Set a conditional breakpoint that triggers when checkpoint context matches.

        Args:
            checkpoint_name: Name of the checkpoint to break at
            context_condition: Dictionary of context conditions (e.g., {"block": 0, "timestep": 0})

        Returns:
            DebuggerResponse indicating success or failure
        """
        try:
            self.debugger.set_conditional_checkpoint_breakpoint(checkpoint_name, context_condition)
            message = f"✅ Conditional breakpoint set at '{checkpoint_name}' with context: {context_condition}"
            return DebuggerResponse(success=True, message=message)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to set conditional checkpoint breakpoint: {e}")
            return DebuggerResponse(
                success=False, message="Failed to set conditional checkpoint breakpoint", error=str(e)
            )


# Convenience function for creating a singleton service
_service_instance: Optional[DebuggerService] = None


def get_debugger_service() -> DebuggerService:
    """Get or create the global debugger service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DebuggerService()
    return _service_instance
