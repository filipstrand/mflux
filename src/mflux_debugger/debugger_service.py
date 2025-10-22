"""
Debugger service layer - transport-agnostic debugging operations.

This module provides a clean API for debugging operations that can be
used by different transport layers (MCP, FastAPI, gRPC, etc.).
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mflux_debugger.lightweight_debugger import DebugState, LightweightDebugger
from mflux_debugger.trace_recorder import TraceRecorder

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

    def __init__(self, enable_rich_context: bool = True, enable_trace: bool = True):
        """Initialize the debugger service.

        Args:
            enable_rich_context: Enable rich context (code context, call stack, auto-preview)
            enable_trace: Enable trace recording (enabled by default)
        """
        self.debugger = LightweightDebugger()
        self.enable_rich_context = enable_rich_context
        self.enable_trace = enable_trace
        self.trace_recorder: Optional[TraceRecorder] = None
        self._execution_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

    def start_session(self, script_path: str) -> DebuggerResponse:
        """
        Start a debugging session.

        Note: This doesn't actually run the script yet - it just prepares the debugger.
        The script starts executing when continue_execution() is first called.

        IMPORTANT: This automatically terminates any existing debug session first
        to prevent memory issues and ensure clean state.
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

            # Create a fresh debugger instance for this session
            self.debugger = LightweightDebugger()

            # Store the script path but don't start yet
            # We'll start when the user calls continue for the first time
            resolved_path = str(Path(script_path).resolve())
            self.debugger.script_path = resolved_path
            self.debugger.add_watch_file(resolved_path)

            # Initialize trace recorder if enabled
            if self.enable_trace:
                self.trace_recorder = TraceRecorder(script_path=resolved_path)
                logger.info(f"Trace recording enabled: {self.trace_recorder.trace_file}")

            message = f"✅ Debug session started for `{Path(script_path).name}`\n\n"
            if cleaned_up:
                message += "🧹 Cleaned up previous session\n\n"
            if self.enable_trace and self.trace_recorder:
                message += f"📝 Trace recording to: `{self.trace_recorder.trace_file.name}`\n\n"
            message += "Session is initialized and ready.\n"
            message += "Set breakpoints and use `debug_continue()` to start execution."

            return DebuggerResponse(
                success=True,
                message=message,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start session: {e}", exc_info=True)
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def set_breakpoint(self, file_path: str, line: int, condition: Optional[str] = None) -> DebuggerResponse:
        """Set a breakpoint."""
        try:
            self.debugger.set_breakpoint(file_path, line, condition)

            # Record breakpoint in trace
            if self.trace_recorder:
                self.trace_recorder.record_breakpoint(file_path, line, condition)

            return DebuggerResponse(
                success=True,
                message=f"✅ Verified breakpoint set\n📍 File: `{file_path}` (line {line})",
                data={"file": file_path, "line": line, "condition": condition},
            )
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

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

                    # Record step in trace
                    if self.trace_recorder:
                        self.trace_recorder.record_step(
                            location=location_dict,
                            code_context=response_data.get("code_context"),
                            call_stack=response_data.get("call_stack"),
                            variable_preview=response_data.get("variable_preview"),
                            step_type="continue",
                        )

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

                # Start execution in background thread
                def _execute():
                    try:
                        if self.debugger.state == DebugState.NOT_STARTED and hasattr(self.debugger, "script_path"):
                            logger.info("Starting script in background thread")
                            self.debugger.start_script(self.debugger.script_path)
                        else:
                            logger.info("Continuing execution in background thread")
                            self.debugger.continue_execution()

                        # Record step with rich context if we paused (for trace)
                        if self.debugger.state == DebugState.PAUSED and self.trace_recorder:
                            location_tuple = self.debugger.get_location()
                            if location_tuple:
                                file_path, line, func_name = location_tuple
                                location_dict = {"file": file_path, "line": line, "function": func_name}

                                # Get rich context
                                if self.enable_rich_context:
                                    code_context = self._get_code_context(file_path, line, context_lines=3)
                                    call_stack = self._get_call_stack_info()
                                    variables = self.debugger.list_variables()
                                    variable_preview = self._smart_variable_preview(variables, limit=5)
                                else:
                                    code_context = None
                                    call_stack = None
                                    variable_preview = None

                                # Record the step
                                self.trace_recorder.record_step(
                                    location=location_dict,
                                    code_context=code_context,
                                    call_stack=call_stack,
                                    variable_preview=variable_preview,
                                    step_type="continue_async",
                                )
                                logger.info(f"Recorded async pause at {file_path}:{line}")

                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Error in background execution: {e}", exc_info=True)

                self._execution_thread = threading.Thread(target=_execute, daemon=True)
                self._execution_thread.start()

                return DebuggerResponse(
                    success=True,
                    message="▶️ Execution started in background\n\n"
                    "💡 Use `debug_status()` or `debug_location()` to check if paused.",
                    data={"state": "running"},
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
            variables = self.debugger.list_variables()

            lines = ["**Variables in current scope:**"]
            for name, value in variables.items():
                formatted = self._format_value(value, max_length=100)
                lines.append(f"  • **{name}**: {formatted}")

            return DebuggerResponse(success=True, message="\n".join(lines), data={"variables": variables})
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def inspect_variable(self, name: str, show_stats: bool = False) -> DebuggerResponse:
        """Inspect a specific variable."""
        try:
            variables = self.debugger.list_variables()
            if name not in variables:
                return DebuggerResponse(success=False, message=f"❌ Variable `{name}` not found")

            value = variables[name]
            result = f"**{name}**\n"

            # Type and basic info
            type_name = type(value).__name__
            result += f"Type: `{type_name}`\n"

            # Special handling for tensors
            if hasattr(value, "shape"):
                result += f"Shape: `{value.shape}`\n"
                result += f"Dtype: `{getattr(value, 'dtype', 'unknown')}`\n"

                if show_stats and hasattr(value, "mean"):
                    try:
                        result += "\n**Statistics:**\n"
                        result += f"  • Mean: {value.mean():.6f}\n"
                        if hasattr(value, "std"):
                            result += f"  • Std: {value.std():.6f}\n"
                        if hasattr(value, "min"):
                            result += f"  • Min: {value.min():.6f}\n"
                        if hasattr(value, "max"):
                            result += f"  • Max: {value.max():.6f}\n"
                    except Exception as e:  # noqa: BLE001
                        result += f"  (stats unavailable: {e})\n"
            else:
                result += f"\nValue: `{self._format_value(value, max_length=500)}`\n"

            return DebuggerResponse(success=True, message=result, data={"name": name, "value": value})
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def evaluate(self, expression: str) -> DebuggerResponse:
        """Evaluate a Python expression in the current context."""
        try:
            result = self.debugger.evaluate(expression)
            formatted = self._format_value(result, max_length=500)

            # Record evaluation in trace
            if self.trace_recorder:
                location_tuple = self.debugger.get_location()
                location_dict = None
                if location_tuple:
                    file_path, line, func_name = location_tuple
                    location_dict = {"file": file_path, "line": line, "function": func_name}
                self.trace_recorder.record_evaluation(expression, result, location_dict)

            message = f"**Expression:** `{expression}`\n\n**Result:** {formatted}"
            # Return formatted string instead of raw result to avoid serialization issues with MLX arrays
            return DebuggerResponse(success=True, message=message, data={"expression": expression, "result": formatted})
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Evaluation failed: {e}", error=str(e))

    def get_location(self) -> DebuggerResponse:
        """Get the current execution location with optional rich context."""
        try:
            location_tuple = self.debugger.get_location()
            if location_tuple:
                file_path, line, func_name = location_tuple
                location_dict = {"file": file_path, "line": line, "function": func_name}

                # Build response data
                response_data = {"location": location_dict}

                # Add rich context if enabled
                if self.enable_rich_context:
                    variables = self.debugger.list_variables()
                    response_data["code_context"] = self._get_code_context(file_path, line, context_lines=3)
                    response_data["call_stack"] = self._get_call_stack_info()
                    response_data["variable_preview"] = self._smart_variable_preview(variables, limit=5)

                message = f"📍 Current location: `{file_path}:{line}` in `{func_name}()`"
                return DebuggerResponse(success=True, message=message, data=response_data)
            else:
                return DebuggerResponse(success=False, message="❌ No current location")
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

    def check_status(self) -> DebuggerResponse:
        """Check the current debugger status."""
        state = self.debugger.state
        state_messages = {
            DebugState.NOT_STARTED: "❌ Debug session not started",
            DebugState.RUNNING: "▶️ Program is running",
            DebugState.PAUSED: "⏸️ Program is stopped",
            DebugState.FINISHED: "✅ Program finished",
        }
        message = state_messages.get(state, f"Unknown state: {state}")
        return DebuggerResponse(success=True, message=message, data={"state": state.value})

    def terminate(self) -> DebuggerResponse:
        """Terminate the debugging session and save trace."""
        try:
            self.debugger.terminate()

            message = "✅ Debug session terminated and subprocess cleaned up."

            # Save trace if recording was enabled
            if self.trace_recorder:
                self.trace_recorder.save()
                summary = self.trace_recorder.get_summary()
                message += f"\n\n{summary}"

            return DebuggerResponse(success=True, message=message)
        except Exception as e:  # noqa: BLE001
            return DebuggerResponse(success=False, message=f"❌ Error: {e}", error=str(e))

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

                    # Try to get basic stats (will trigger MLX eval if needed)
                    try:
                        import mlx.core as mx  # noqa: PLC0415

                        if isinstance(value, mx.array):
                            # Already evaluated by auto_eval_mlx
                            var_info["device"] = "mps"  # MLX always uses Metal
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

                # Record step in trace
                if self.trace_recorder:
                    self.trace_recorder.record_step(
                        location=location_dict,
                        code_context=response_data.get("code_context"),
                        call_stack=response_data.get("call_stack"),
                        variable_preview=response_data.get("variable_preview"),
                        step_type=step_type,
                    )

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
                except Exception:  # noqa: BLE001
                    pass

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
                except Exception:  # noqa: BLE001
                    # If we can't get samples, that's okay
                    pass

                return ", ".join(parts) + ")"

            # Regular values
            str_value = str(value)
            if len(str_value) > max_length:
                return str_value[:max_length] + "..."
            return str_value
        except Exception as e:  # noqa: BLE001
            return f"<error formatting value: {e}>"


# Convenience function for creating a singleton service
_service_instance: Optional[DebuggerService] = None


def get_debugger_service() -> DebuggerService:
    """Get or create the global debugger service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DebuggerService()
    return _service_instance
