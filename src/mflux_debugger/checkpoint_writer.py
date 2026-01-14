"""
Checkpoint writer that always logs debug_checkpoint() calls to JSON.

This provides automatic state capture at semantic checkpoints, whether
the interactive debugger is running or not.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mflux_debugger.log_paths import (
    get_run_checkpoints_dir,
    get_run_output_log_path,
    get_run_session_dir,
    get_runs_archive_dir,
    get_runs_latest_dir,
)

logger = logging.getLogger(__name__)

# Storage limits
CHECKPOINT_WARNING_SIZE_GB = 1.0
CHECKPOINT_HARD_LIMIT_GB = 10.0


class CheckpointWriter:
    """Writes checkpoint state to JSON files automatically."""

    def __init__(self, script_name: str, ab_run_id: Optional[str] = None):
        """
        Initialize checkpoint writer for a script.

        Args:
            script_name: Name of the script being debugged (used for directory)
            ab_run_id: Optional A/B debugging run identifier. When provided,
                all existing sessions in `runs/latest/` whose directory name
                does *not* start with this ID will be archived. This ensures
                that `runs/latest/` only contains data for the current A/B run.
        """
        self.script_name = script_name
        self.ab_run_id = ab_run_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Archive old sessions
        self._archive_old_sessions()

        # Check storage limits before starting
        self._check_storage_limits()

        # Create session directory (logs/runs/latest/{run_id__}{script}_{timestamp}/)
        self.session_dir = get_run_session_dir(script_name, self.timestamp, ab_run_id=self.ab_run_id)

        # Checkpoints go in a subdirectory
        self.checkpoints_dir = get_run_checkpoints_dir(script_name, self.timestamp, ab_run_id=self.ab_run_id)

        # Script output log file
        self.output_log_path = get_run_output_log_path(script_name, self.timestamp, ab_run_id=self.ab_run_id)
        self.output_log_file = None  # Will be opened when first checkpoint is hit

        # Track checkpoint hit counts
        self.checkpoint_hits: Dict[str, int] = {}

        logger.info(f"📂 Run session started: {self.session_dir.relative_to(self.session_dir.parent.parent.parent)}")

    def _archive_old_sessions(self):
        """
        Archive existing run sessions.

        Behavior:
          - When `ab_run_id` is provided, ANY session directory in
            `runs/latest/` whose name does *not* start with
            `{ab_run_id}__` is moved to `runs/archive/`.
          - When `ab_run_id` is None, we fall back to the previous
            behavior and only archive sessions for this script name.
        """
        latest_dir = get_runs_latest_dir()
        if not latest_dir.exists():
            return

        archive_dir = get_runs_archive_dir()
        archived_count = 0

        if self.ab_run_id:
            # Global A/B mode: keep only directories for this run_id in latest
            prefix = f"{self.ab_run_id}__"
            session_dirs = [p for p in latest_dir.iterdir() if p.is_dir()]
            for session_dir in session_dirs:
                if session_dir.name.startswith(prefix):
                    continue
                archive_path = archive_dir / session_dir.name
                try:
                    if archive_path.exists():
                        shutil.rmtree(archive_path)
                    shutil.move(str(session_dir), str(archive_path))
                    logger.debug(f"📦 Archived old session (different run_id): {session_dir.name}")
                    archived_count += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to archive {session_dir.name}: {e}")
        else:
            # Legacy mode: archive sessions for this script only
            pattern = f"{self.script_name}_*"
            old_sessions = list(latest_dir.glob(pattern))
            if not old_sessions:
                return

            for session_dir in old_sessions:
                if session_dir.is_dir():
                    archive_path = archive_dir / session_dir.name
                    try:
                        if archive_path.exists():
                            # If already archived, remove the old archive
                            shutil.rmtree(archive_path)
                        shutil.move(str(session_dir), str(archive_path))
                        logger.debug(f"📦 Archived old session: {session_dir.name}")
                        archived_count += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Failed to archive {session_dir.name}: {e}")

        if archived_count > 0:
            logger.info(f"📦 Archived {archived_count} old run session(s)")

    def _check_storage_limits(self):
        """Check storage usage and enforce limits (entire mflux_debugger/ directory)."""
        from mflux_debugger.log_paths import get_debugger_root

        debugger_root = get_debugger_root()
        if not debugger_root.exists():
            return

        total_size = sum(f.stat().st_size for f in debugger_root.rglob("*") if f.is_file())
        size_gb = total_size / (1024**3)

        if size_gb >= CHECKPOINT_HARD_LIMIT_GB:
            error_msg = (
                f"❌ Debugger storage exceeded {CHECKPOINT_HARD_LIMIT_GB}GB limit!\n"
                f"   Current size: {size_gb:.2f}GB\n"
                f"   Directory: {debugger_root}\n\n"
                f"   Please clean up:\n"
                f"   • Run: mflux-debug-clean --yes\n"
                f"   • Or preview first: mflux-debug-clean --dry-run"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if size_gb >= CHECKPOINT_WARNING_SIZE_GB:
            warning_msg = (
                f"⚠️  Debugger storage size: {size_gb:.2f}GB (warning threshold: {CHECKPOINT_WARNING_SIZE_GB}GB)\n"
                f"   Directory: {debugger_root}\n"
                f"   Consider cleaning up: mflux-debug-clean"
            )
            logger.warning(warning_msg)
            print(warning_msg, flush=True)

    def capture_checkpoint(
        self,
        checkpoint_name: str,
        frame: Any,
        variables: Dict[str, Any],
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Capture state at a checkpoint and write to JSON.

        Args:
            checkpoint_name: Name of the checkpoint
            frame: Execution frame
            variables: Variables to capture
            context: Execution context (block, timestep, etc.)
            metadata: Additional metadata
        """
        # Track hit count
        if checkpoint_name not in self.checkpoint_hits:
            self.checkpoint_hits[checkpoint_name] = 0
        self.checkpoint_hits[checkpoint_name] += 1
        hit_count = self.checkpoint_hits[checkpoint_name]

        # Get location info
        filename = frame.f_code.co_filename
        line_no = frame.f_lineno
        func_name = frame.f_code.co_name

        # Get code context
        code_context = self._get_code_context(filename, line_no)

        # Auto-evaluate MLX arrays before serializing
        self._auto_eval_mlx_in_scope(variables)

        # Serialize variables
        serialized_vars = self._serialize_variables(variables)

        # Create checkpoint data
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "hit_count": hit_count,
            "timestamp": datetime.now().isoformat(),
            "location": {
                "file": filename,
                "line": line_no,
                "function": func_name,
            },
            "code_context": code_context,
            "variables": serialized_vars,
        }

        if context:
            checkpoint_data["context"] = context

        if metadata:
            checkpoint_data["metadata"] = metadata

        # Write to file: checkpoints/checkpoint_<name>_hit<count>.json
        safe_name = checkpoint_name.replace(":", "_").replace("/", "_")
        output_file = self.checkpoints_dir / f"checkpoint_{safe_name}_hit{hit_count:03d}.json"

        # Ensure JSON-safe
        json_safe_data = self._ensure_json_safe(checkpoint_data)
        output_file.write_text(json.dumps(json_safe_data, indent=2))

        # Log to script output file with reference to checkpoint JSON
        self._log_checkpoint_to_output(checkpoint_name, hit_count, output_file, metadata)

        logger.debug(
            f"📸 Checkpoint logged: {checkpoint_name} (hit #{hit_count}) -> {output_file.relative_to(self.session_dir)}"
        )

    def _get_code_context(self, filename: str, line_no: int, context_lines: int = 5) -> Dict:
        """Get code context around the checkpoint."""
        try:
            with open(filename) as f:
                lines = f.readlines()

            start = max(0, line_no - context_lines - 1)
            end = min(len(lines), line_no + context_lines)

            context = {
                "start_line": start + 1,
                "end_line": end,
                "current_line": line_no,
                "lines": {},
            }

            for i in range(start, end):
                line_num = i + 1
                marker = " --> " if line_num == line_no else "     "
                context["lines"][line_num] = f"{marker}{lines[i].rstrip()}"

            return context
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not read code context: {e}"}

    def _log_checkpoint_to_output(
        self, checkpoint_name: str, hit_count: int, json_file: Path, metadata: Optional[Dict]
    ) -> None:
        """Log checkpoint hit to script output file with reference to JSON."""
        # Open log file if not already open
        if self.output_log_file is None:
            self.output_log_file = open(self.output_log_path, "w", buffering=1)  # Line buffered
            # Write header
            self.output_log_file.write("=" * 80 + "\n")
            self.output_log_file.write(f"Script Output Log: {self.script_name}\n")
            self.output_log_file.write(f"Session: {self.timestamp}\n")
            self.output_log_file.write("=" * 80 + "\n\n")

        # Format timestamp
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

        # Write checkpoint log entry
        self.output_log_file.write(f"[{now}] 📍 Checkpoint '{checkpoint_name}' (hit #{hit_count})\n")

        # Add metadata info if present
        if metadata:
            if metadata.get("skip"):
                self.output_log_file.write("           └─ skipped=True (logged but not paused)\n")
            meta_info = ", ".join(f"{k}={v}" for k, v in metadata.items() if k != "skip")
            if meta_info:
                self.output_log_file.write(f"           └─ {meta_info}\n")

        # Write JSON file reference (relative path from session dir)
        rel_path = json_file.relative_to(self.session_dir)
        self.output_log_file.write(f"           └─ JSON: {rel_path}\n")
        self.output_log_file.write(f"           └─ Full path: {json_file.absolute()}\n")
        self.output_log_file.write("\n")
        self.output_log_file.flush()

    def __del__(self):
        """Close output log file on cleanup."""
        if self.output_log_file is not None:
            try:
                self.output_log_file.close()
            except Exception:  # noqa: BLE001, S110
                pass

    def _auto_eval_mlx_in_scope(self, variables: Dict[str, Any]) -> None:
        """Automatically evaluate MLX arrays before inspection."""
        try:
            import mlx.core as mx  # noqa: PLC0415

            # Collect all MLX arrays (list comprehension for performance)
            arrays_to_eval = [value for value in variables.values() if isinstance(value, mx.array)]

            # Evaluate all at once
            if arrays_to_eval:
                mx.eval(*arrays_to_eval)
        except ImportError:
            pass  # MLX not installed
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MLX eval failed: {e}")

    def _serialize_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize variables to JSON-compatible format."""
        serialized = {}

        for name, value in variables.items():
            try:
                serialized[name] = self._serialize_value(value)
            except Exception as e:  # noqa: BLE001, PERF203
                serialized[name] = {"error": f"Failed to serialize: {e}", "type": str(type(value).__name__)}

        return serialized

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to JSON-compatible format."""
        # Handle None
        if value is None:
            return None

        # Handle basic types
        if isinstance(value, (bool, int, float, str)):
            return value

        # Handle tensors/arrays
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return self._serialize_tensor(value)

        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]

        # Handle dicts
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Fallback: convert to string
        return {"type": str(type(value).__name__), "value": str(value)}

    def _serialize_tensor(self, tensor: Any) -> Dict[str, Any]:
        """Serialize tensor/array with shape, dtype, stats, and preview."""
        try:
            shape = list(tensor.shape) if hasattr(tensor, "shape") else None
            dtype = str(tensor.dtype) if hasattr(tensor, "dtype") else None

            # Get preview values (first 10 elements flattened)
            preview = []
            try:
                if hasattr(tensor, "flatten"):
                    flat = tensor.flatten()
                    preview_tensor = flat[:10]
                    # Convert to list
                    if hasattr(preview_tensor, "tolist"):
                        preview = preview_tensor.tolist()
                    elif hasattr(preview_tensor, "numpy"):
                        preview = preview_tensor.numpy().tolist()
                    else:
                        preview = [float(x) for x in preview_tensor]
            except Exception:  # noqa: BLE001
                pass

            # Get statistics
            stats = {}
            try:
                # For float16 tensors, cast to float32 before computing statistics
                # to avoid numerical precision issues (NaN/Inf in mean/std)
                tensor_for_stats = tensor
                if hasattr(tensor, "dtype") and str(tensor.dtype) in ("mlx.core.float16", "float16"):
                    try:
                        # Try to cast to float32 for statistics
                        if hasattr(tensor, "astype"):
                            tensor_for_stats = tensor.astype("float32")
                        elif hasattr(tensor, "numpy"):
                            # Convert to numpy and cast
                            np_array = tensor.numpy()
                            tensor_for_stats = np_array.astype("float32")
                    except Exception:  # noqa: BLE001
                        # If casting fails, use original tensor
                        tensor_for_stats = tensor

                if hasattr(tensor_for_stats, "mean"):
                    mean_val = float(tensor_for_stats.mean())
                    # Check if mean is valid (not NaN or Inf)
                    if (
                        mean_val == mean_val and mean_val != float("inf") and mean_val != float("-inf")
                    ):  # NaN and Inf check
                        stats["mean"] = mean_val
                if hasattr(tensor_for_stats, "min"):
                    stats["min"] = float(tensor_for_stats.min())
                if hasattr(tensor_for_stats, "max"):
                    stats["max"] = float(tensor_for_stats.max())
                # Calculate std manually to avoid MLX float16 precision issues
                if hasattr(tensor_for_stats, "std"):
                    try:
                        std_val = float(tensor_for_stats.std())
                        # Check if std is inf or nan (MLX float16 can have precision issues)
                        if not (std_val == float("inf") or std_val != std_val):  # inf or nan check
                            stats["std"] = std_val
                        else:
                            # Fallback: calculate std manually via numpy if available
                            try:
                                import numpy as np

                                if hasattr(tensor_for_stats, "numpy"):
                                    np_array = tensor_for_stats.numpy()
                                    stats["std"] = float(np.std(np_array))
                                elif hasattr(tensor_for_stats, "flatten"):
                                    # Try converting to list and calculating
                                    flat = tensor_for_stats.flatten()
                                    if hasattr(flat, "tolist"):
                                        values = flat.tolist()
                                        stats["std"] = float(np.std(values))
                            except ImportError:
                                pass
                            except Exception:  # noqa: BLE001
                                # If all else fails, skip std
                                pass
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

            return {
                "type": type(tensor).__name__,
                "shape": shape,
                "dtype": dtype,
                "preview": preview,
                "stats": stats,
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"Failed to serialize tensor: {e}"}

    def _ensure_json_safe(self, obj: Any) -> Any:
        """Recursively ensure object is JSON-serializable."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        if isinstance(obj, dict):
            return {k: self._ensure_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._ensure_json_safe(item) for item in obj]

        # Fallback: convert to string
        return str(obj)


# Global checkpoint writer instance (one per script)
_CHECKPOINT_WRITER: Optional[CheckpointWriter] = None


def get_checkpoint_writer(script_name: str, ab_run_id: Optional[str] = None) -> CheckpointWriter:
    """Get or create the global checkpoint writer for a script."""
    global _CHECKPOINT_WRITER

    if (
        _CHECKPOINT_WRITER is None
        or _CHECKPOINT_WRITER.script_name != script_name
        or _CHECKPOINT_WRITER.ab_run_id != ab_run_id
    ):
        _CHECKPOINT_WRITER = CheckpointWriter(script_name, ab_run_id=ab_run_id)

    return _CHECKPOINT_WRITER


def clear_checkpoint_writer():
    """Clear the global checkpoint writer."""
    global _CHECKPOINT_WRITER
    _CHECKPOINT_WRITER = None
