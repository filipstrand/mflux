"""
Execution Profiler Service

Lightweight profiler that captures function call traces WITHOUT capturing variable state.
Use this for discovery: understanding what code actually executes before setting breakpoints.

Key Features:
- Records function calls with file, line, function name, stack depth
- Smart filtering to exclude noise (stdlib, framework internals)
- Minimal overhead - no variable serialization
- Outputs execution profile for analysis

Workflow:
1. Profile execution to see what runs
2. Analyze profile to identify key functions
3. Set informed breakpoints in the real debugger
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FunctionCall:
    """Record of a single function call."""

    timestamp: float
    event: str  # "call" or "return"
    function: str
    file: str
    line: int
    depth: int


@dataclass
class ExecutionProfile:
    """Complete execution profile of a script run."""

    session_id: str
    script_path: str
    start_time: float
    end_time: float | None = None
    calls: list[FunctionCall] = field(default_factory=list)
    filter_paths: list[str] = field(default_factory=list)
    max_depth: int | None = None


class ExecutionProfiler:
    """
    Lightweight profiler that traces function calls without capturing state.

    Filtering strategy:
    - Include: Files matching filter_paths (e.g., project code, specific libraries)
    - Exclude: Python stdlib, common frameworks (unless explicitly included)
    - Depth limit: Stop recording beyond certain stack depth to avoid noise
    """

    def __init__(
        self,
        filter_paths: list[str] | None = None,
        max_depth: int | None = 20,
        exclude_patterns: list[str] | None = None,
    ):
        """
        Initialize profiler with filters.

        Args:
            filter_paths: List of path prefixes to include (e.g., ["/path/to/project", "transformers"])
                         If None, includes project root by default
            max_depth: Maximum stack depth to record (None = unlimited)
            exclude_patterns: Additional patterns to exclude (e.g., ["test_", "debug_"])
        """
        self.profile: ExecutionProfile | None = None
        self.current_depth = 0
        self.start_time = 0.0

        # Setup filters
        if filter_paths is None:
            # Default: include current working directory (project root)
            project_root = str(Path.cwd())
            self.filter_paths = [project_root]
        else:
            self.filter_paths = filter_paths

        self.max_depth = max_depth

        # Default exclusions (common noise sources)
        default_excludes = [
            "/lib/python",  # stdlib
            "/site-packages/",  # third-party (unless explicitly in filter_paths)
            "<frozen",  # frozen imports
            "importlib",
            "_bootstrap",
            "contextlib",
            "abc.py",
        ]

        self.exclude_patterns = default_excludes + (exclude_patterns or [])

    def _should_trace(self, filename: str) -> bool:
        """Determine if we should trace this file."""
        # Check exclusions first (fast rejection)
        for pattern in self.exclude_patterns:
            if pattern in filename:
                # But allow if explicitly in filter_paths
                if not any(filter_path in filename for filter_path in self.filter_paths):
                    return False

        # Check if file matches any filter path
        if self.filter_paths:
            return any(filter_path in filename for filter_path in self.filter_paths)

        return False

    def _trace_calls(self, frame: Any, event: str, arg: Any) -> Any:  # noqa: ARG002
        """Trace function to capture call/return events."""
        if event not in ("call", "return"):
            return self._trace_calls

        # Check depth limit
        if self.max_depth is not None and self.current_depth > self.max_depth:
            return self._trace_calls

        code = frame.f_code
        filename = code.co_filename
        function_name = code.co_name
        line_number = frame.f_lineno

        # Apply filtering
        if not self._should_trace(filename):
            return self._trace_calls

        # Update depth tracking
        if event == "call":
            self.current_depth += 1
        elif event == "return":
            self.current_depth = max(0, self.current_depth - 1)

        # Record the call
        timestamp = time.time() - self.start_time
        call = FunctionCall(
            timestamp=timestamp,
            event=event,
            function=function_name,
            file=filename,
            line=line_number,
            depth=self.current_depth,
        )

        if self.profile:
            self.profile.calls.append(call)

        return self._trace_calls

    def start(self, script_path: str, session_id: str) -> None:
        """Start profiling."""
        self.start_time = time.time()
        self.current_depth = 0

        self.profile = ExecutionProfile(
            session_id=session_id,
            script_path=script_path,
            start_time=self.start_time,
            filter_paths=self.filter_paths,
            max_depth=self.max_depth,
        )

        # Install trace function
        sys.settrace(self._trace_calls)

    def stop(self) -> ExecutionProfile:
        """Stop profiling and return profile."""
        # Remove trace function
        sys.settrace(None)

        if self.profile:
            self.profile.end_time = time.time()

        return self.profile

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the profile."""
        if not self.profile or not self.profile.calls:
            return {}

        # Count unique functions
        unique_functions: dict[str, dict[str, Any]] = {}
        for call in self.profile.calls:
            if call.event == "call":
                key = f"{call.file}::{call.function}"
                if key not in unique_functions:
                    unique_functions[key] = {
                        "function": call.function,
                        "file": call.file,
                        "first_line": call.line,
                        "hit_count": 0,
                        "max_depth": call.depth,
                    }
                unique_functions[key]["hit_count"] = int(unique_functions[key]["hit_count"]) + 1
                current_depth: int = int(unique_functions[key]["max_depth"])
                unique_functions[key]["max_depth"] = max(current_depth, call.depth)

        # Sort by hit count
        sorted_functions = sorted(unique_functions.values(), key=lambda x: x["hit_count"], reverse=True)

        duration = self.profile.end_time - self.profile.start_time if self.profile.end_time else 0.0

        return {
            "total_calls": len(self.profile.calls),
            "unique_functions": len(unique_functions),
            "duration": duration,
            "max_depth_reached": max((c.depth for c in self.profile.calls), default=0),
            "top_functions": sorted_functions[:20],  # Top 20 most called
        }

    def save(self, output_dir: Path | None = None) -> Path:
        """Save profile to JSON file."""
        if not self.profile:
            raise RuntimeError("No profile data to save")

        if output_dir is None:
            # Save to mflux_debugger/prune in current directory
            output_dir = Path("mflux_debugger") / "prune"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        script_name = Path(self.profile.script_path).stem
        filename = f"{script_name}_{self.profile.session_id}.json"
        output_path = output_dir / filename

        # Prepare data for serialization
        profile_data = {
            "session_id": self.profile.session_id,
            "script_path": self.profile.script_path,
            "start_time": self.profile.start_time,
            "end_time": self.profile.end_time,
            "filter_paths": self.profile.filter_paths,
            "max_depth": self.profile.max_depth,
            "summary": self.get_summary(),
            "calls": [
                {
                    "timestamp": call.timestamp,
                    "event": call.event,
                    "function": call.function,
                    "file": call.file,
                    "line": call.line,
                    "depth": call.depth,
                }
                for call in self.profile.calls
            ],
        }

        # Save to file
        with output_path.open("w") as f:
            json.dump(profile_data, f, indent=2)

        return output_path


def create_profiler(
    filter_paths: list[str] | None = None,
    max_depth: int | None = 20,
    exclude_patterns: list[str] | None = None,
) -> ExecutionProfiler:
    """
    Factory function to create a profiler with common configurations.

    Args:
        filter_paths: Paths to include in trace (default: project root)
        max_depth: Maximum stack depth to trace (default: 20)
        exclude_patterns: Additional exclusion patterns

    Returns:
        ExecutionProfiler instance
    """
    return ExecutionProfiler(filter_paths=filter_paths, max_depth=max_depth, exclude_patterns=exclude_patterns)
