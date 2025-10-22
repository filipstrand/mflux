"""
Trace recorder for debugging sessions.

Records all steps, evaluations, and state changes during a debug session
for offline review and comparison between implementations.
"""

import atexit
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceRecorder:
    """Records debugging session steps for offline analysis."""

    def __init__(self, script_path: str, traces_dir: str = "mflux_debugger/traces"):
        """
        Initialize trace recorder.

        Args:
            script_path: Path to script being debugged
            traces_dir: Directory to store trace files
        """
        self.script_path = script_path
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID from timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate trace filename
        script_name = Path(script_path).stem
        self.trace_file = self.traces_dir / f"{script_name}_{self.session_id}.json"

        # Session data
        self.trace_data = {
            "session_id": self.session_id,
            "script_path": script_path,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "breakpoints": [],
            "steps": [],
        }

        self.step_count = 0
        self._auto_save = True  # Auto-save after each step for crash resilience

        # Register emergency save handler for unexpected exits
        atexit.register(self._emergency_save)

        logger.info(f"Trace recording started: {self.trace_file}")

    def record_breakpoint(self, file_path: str, line: int, condition: Optional[str] = None):
        """Record a breakpoint being set."""
        self.trace_data["breakpoints"].append({"file": file_path, "line": line, "condition": condition})
        if self._auto_save:
            self._save_incremental()

    def record_step(
        self,
        location: Dict[str, Any],
        code_context: Optional[Dict[str, Any]] = None,
        call_stack: Optional[List[Dict[str, Any]]] = None,
        variable_preview: Optional[Dict[str, Any]] = None,
        step_type: str = "pause",
    ):
        """
        Record a debug step (pause, step_over, step_into, etc.).

        Args:
            location: Current location (file, line, function)
            code_context: Code around current line
            call_stack: Current call stack
            variable_preview: Preview of variables (shapes/types, not full data)
            step_type: Type of step (pause, step_over, step_into, step_out)
        """
        self.step_count += 1

        step_entry = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "location": location,
        }

        # Include code context if provided (full context for replay)
        if code_context:
            step_entry["code_context"] = {
                "before": code_context.get("before", []),
                "current": code_context.get("current"),
                "after": code_context.get("after", []),
            }

        # Include call stack (just function names for compactness)
        if call_stack:
            step_entry["call_stack"] = [
                {"function": frame.get("function"), "line": frame.get("line")} for frame in call_stack
            ]

        # Include variable preview (already compact - shapes/types only)
        if variable_preview:
            step_entry["variables"] = variable_preview

        self.trace_data["steps"].append(step_entry)

        if self._auto_save:
            self._save_incremental()

    def record_evaluation(self, expression: str, result: Any, location: Optional[Dict[str, Any]] = None):
        """
        Record an evaluation.

        Args:
            expression: Expression that was evaluated
            result: Result of evaluation (will be truncated if large)
            location: Optional location where evaluation occurred
        """
        self.step_count += 1

        # Truncate large results
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "... (truncated)"

        eval_entry = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "step_type": "evaluation",
            "expression": expression,
            "result": result_str,
        }

        if location:
            eval_entry["location"] = location

        self.trace_data["steps"].append(eval_entry)

        if self._auto_save:
            self._save_incremental()

    def save(self) -> str:
        """
        Save trace to disk.

        Returns:
            Path to saved trace file
        """
        self.trace_data["end_time"] = datetime.now().isoformat()
        self.trace_data["total_steps"] = self.step_count

        try:
            with open(self.trace_file, "w") as f:
                json.dump(self.trace_data, f, indent=2)

            logger.info(f"Trace saved: {self.trace_file} ({self.step_count} steps)")
            return str(self.trace_file)

        except Exception as e:
            logger.error(f"Failed to save trace: {e}", exc_info=True)
            return f"Error: {e}"

    def get_summary(self) -> str:
        """Get a summary of the trace."""
        duration = (
            datetime.fromisoformat(self.trace_data["end_time"]) - datetime.fromisoformat(self.trace_data["start_time"])
            if self.trace_data["end_time"]
            else None
        )

        summary = f"""
📝 Debug Trace Summary

Session ID: {self.session_id}
Script: {Path(self.script_path).name}
Steps: {self.step_count}
Breakpoints: {len(self.trace_data["breakpoints"])}
Duration: {duration.total_seconds():.1f}s
Trace file: {self.trace_file}
"""
        return summary.strip()

    def _save_incremental(self):
        """
        Save trace incrementally (without marking as complete).

        This is called after each step to ensure we don't lose data on crashes.
        The trace is saved without setting end_time, so we know it's incomplete.
        """
        try:
            temp_data = self.trace_data.copy()
            temp_data["total_steps"] = self.step_count
            # Don't set end_time - this indicates incomplete trace

            with open(self.trace_file, "w") as f:
                json.dump(temp_data, f, indent=2)

        except Exception as e:  # noqa: BLE001
            # Don't let save failures crash the debug session
            logger.warning(f"Failed to auto-save trace: {e}")

    def _emergency_save(self):
        """
        Emergency save on unexpected exit (called by atexit).

        Marks the trace as incomplete but preserves all recorded data.
        """
        try:
            # Only save if we have steps and haven't already saved
            if self.step_count > 0 and not self.trace_data.get("end_time"):
                self.trace_data["end_time"] = datetime.now().isoformat()
                self.trace_data["total_steps"] = self.step_count
                self.trace_data["incomplete"] = True  # Flag as incomplete

                with open(self.trace_file, "w") as f:
                    json.dump(self.trace_data, f, indent=2)

                logger.warning(f"Emergency save: {self.trace_file} (incomplete trace)")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Emergency save failed: {e}")
