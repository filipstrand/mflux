"""
Async Checkpoint Saving for MLX Training.

Saves checkpoints in background threads so training can continue
without waiting for disk I/O. Critical for large-scale training
where checkpoint saves can take 5-10 seconds.

Features:
- Background saving with ThreadPoolExecutor
- Configurable max pending saves (prevents memory buildup)
- Thread-safe checkpoint path management
- Graceful shutdown with wait_all()
"""

import json
import tempfile
import threading
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mlx.core as mx


class AsyncCheckpointer:
    """
    Saves checkpoints in background thread.

    Training continues while checkpoint writes to disk.
    Handles concurrent saves and prevents memory explosion
    from too many pending saves.

    Args:
        output_dir: Base directory for checkpoints
        max_pending: Maximum number of pending saves (default: 2)

    Example:
        checkpointer = AsyncCheckpointer(output_dir="./checkpoints")

        for step, batch in enumerate(dataloader):
            loss, grads = train_step(batch)
            optimizer.update(model, grads)

            if step % 500 == 0:
                # Non-blocking save
                checkpointer.save_async(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metadata={"loss": loss_value}
                )

        # Wait for all saves before exit
        checkpointer.wait_all()
    """

    def __init__(self, output_dir: Path | str, max_pending: int = 2):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_futures: list[Future] = []
        self._lock = threading.Lock()

        # Track saved checkpoints
        self._saved_paths: list[Path] = []
        self._latest_step = 0

    @property
    def pending_count(self) -> int:
        """Number of pending checkpoint saves."""
        with self._lock:
            self._cleanup_completed()
            return len(self._pending_futures)

    @property
    def saved_checkpoints(self) -> list[Path]:
        """List of saved checkpoint paths."""
        with self._lock:
            return list(self._saved_paths)

    def save_async(
        self,
        model: Any,
        step: int,
        optimizer: Any | None = None,
        lr_scheduler: Any | None = None,
        metadata: dict[str, Any] | None = None,
        lora_only: bool = True,
    ) -> None:
        """
        Queue checkpoint save in background.

        Blocks if too many pending saves to prevent memory explosion.

        Args:
            model: Model to save (extracts LoRA weights if lora_only=True)
            step: Training step number
            optimizer: Optional optimizer state to save
            lr_scheduler: Optional LR scheduler state to save
            metadata: Optional metadata dict to save
            lora_only: If True, only save LoRA/DoRA weights (default: True)
        """
        with self._lock:
            # Clean up completed futures
            self._cleanup_completed()

            # Block if too many pending
            while len(self._pending_futures) >= self._max_pending:
                # Wait for oldest to complete
                if self._pending_futures:
                    oldest = self._pending_futures.pop(0)
                    oldest.result()  # This blocks

        # Create snapshot of state (must happen in main thread)
        snapshot = self._create_snapshot(
            model=model,
            step=step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metadata=metadata,
            lora_only=lora_only,
        )

        # Submit to background thread
        future = self._executor.submit(
            self._save_checkpoint,
            snapshot=snapshot,
            step=step,
        )

        with self._lock:
            self._pending_futures.append(future)
            self._latest_step = max(self._latest_step, step)

    def _create_snapshot(
        self,
        model: Any,
        step: int,
        optimizer: Any | None,
        lr_scheduler: Any | None,
        metadata: dict[str, Any] | None,
        lora_only: bool,
    ) -> dict[str, Any]:
        """Create a snapshot of current state for saving."""
        snapshot = {
            "step": step,
            "weights": {},
            "metadata": metadata or {},
        }

        # Extract weights
        params = model.parameters()

        if lora_only:
            # Only save LoRA/DoRA weights
            for name, param in _flatten_params(params):
                if "lora_A" in name or "lora_B" in name or "magnitude" in name:
                    # Force evaluation and copy
                    mx.eval(param)
                    snapshot["weights"][name] = param
        else:
            # Save all weights
            for name, param in _flatten_params(params):
                mx.eval(param)
                snapshot["weights"][name] = param

        # Save optimizer state if provided
        if optimizer is not None:
            try:
                snapshot["optimizer_state"] = optimizer.state
            except AttributeError:
                pass

        # Save LR scheduler state if provided
        if lr_scheduler is not None:
            try:
                snapshot["lr_scheduler_state"] = lr_scheduler.state_dict()
            except AttributeError:
                pass

        return snapshot

    def _save_checkpoint(self, snapshot: dict[str, Any], step: int) -> Path:
        """
        Actual checkpoint save (runs in background thread).

        Creates a zip file with:
        - weights.safetensors: Model weights
        - metadata.json: Training metadata
        - optimizer.safetensors: Optimizer state (if provided)
        """
        checkpoint_name = f"checkpoint_step_{step:07d}"
        checkpoint_path = self.output_dir / f"{checkpoint_name}.zip"

        # Create temp directory for staging
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save weights
            if snapshot["weights"]:
                weights_path = temp_path / "weights.safetensors"
                mx.save_safetensors(str(weights_path), snapshot["weights"])

            # Save metadata
            metadata = snapshot.get("metadata", {})
            metadata["step"] = step
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save optimizer state
            if "optimizer_state" in snapshot:
                try:
                    opt_path = temp_path / "optimizer.safetensors"
                    # Convert optimizer state to saveable format
                    opt_state = {}
                    for k, v in _flatten_params(snapshot["optimizer_state"]):
                        if isinstance(v, mx.array):
                            opt_state[k] = v
                    if opt_state:
                        mx.save_safetensors(str(opt_path), opt_state)
                except Exception:
                    pass  # Skip optimizer save on error

            # Save LR scheduler state
            if "lr_scheduler_state" in snapshot:
                lr_path = temp_path / "lr_scheduler.json"
                with open(lr_path, "w") as f:
                    json.dump(snapshot["lr_scheduler_state"], f, indent=2)

            # Create zip file
            with zipfile.ZipFile(checkpoint_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in temp_path.iterdir():
                    zf.write(file_path, file_path.name)

        with self._lock:
            self._saved_paths.append(checkpoint_path)

        return checkpoint_path

    def _cleanup_completed(self) -> None:
        """Remove completed futures from pending list."""
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def wait_all(self, timeout: float | None = None) -> list[Path]:
        """
        Wait for all pending saves to complete.

        Call this before exiting training to ensure all checkpoints are saved.

        Args:
            timeout: Optional timeout in seconds per save

        Returns:
            List of all saved checkpoint paths
        """
        with self._lock:
            futures = list(self._pending_futures)

        paths = []
        for future in futures:
            try:
                path = future.result(timeout=timeout)
                if path:
                    paths.append(path)
            except Exception as e:
                print(f"Warning: Checkpoint save failed: {e}")

        with self._lock:
            self._cleanup_completed()

        return paths

    def shutdown(self) -> None:
        """Shutdown executor and wait for pending saves."""
        self.wait_all()
        self._executor.shutdown(wait=True)

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> list[Path]:
        """
        Remove old checkpoints, keeping only the most recent.

        Args:
            keep_last: Number of recent checkpoints to keep (default: 5)

        Returns:
            List of deleted checkpoint paths
        """
        with self._lock:
            if len(self._saved_paths) <= keep_last:
                return []

            # Sort by step number (extracted from filename)
            sorted_paths = sorted(
                self._saved_paths,
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
            )

            # Delete oldest
            to_delete = sorted_paths[:-keep_last]
            deleted = []

            for path in to_delete:
                try:
                    if path.exists():
                        path.unlink()
                        deleted.append(path)
                        self._saved_paths.remove(path)
                except Exception as e:
                    print(f"Warning: Failed to delete {path}: {e}")

            return deleted


class NoOpCheckpointer:
    """
    No-op checkpointer for when checkpointing is disabled.

    Provides the same interface but does nothing.
    """

    def __init__(self):
        pass

    @property
    def pending_count(self) -> int:
        return 0

    @property
    def saved_checkpoints(self) -> list[Path]:
        return []

    def save_async(self, **kwargs) -> None:
        pass

    def wait_all(self, timeout: float | None = None) -> list[Path]:
        return []

    def shutdown(self) -> None:
        pass

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> list[Path]:
        return []


def _flatten_params(params: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested parameter dictionary."""
    items = []
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_params(value, name))
        else:
            items.append((name, value))
    return items


def load_checkpoint(
    checkpoint_path: Path | str,
) -> dict[str, Any]:
    """
    Load checkpoint from zip file.

    Args:
        checkpoint_path: Path to checkpoint zip file

    Returns:
        Dict with keys: weights, metadata, optimizer_state (optional), lr_scheduler_state (optional)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    result = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            zf.extractall(temp_path)

        # Load weights
        weights_path = temp_path / "weights.safetensors"
        if weights_path.exists():
            result["weights"] = mx.load(str(weights_path))

        # Load metadata
        metadata_path = temp_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                result["metadata"] = json.load(f)

        # Load optimizer state
        opt_path = temp_path / "optimizer.safetensors"
        if opt_path.exists():
            result["optimizer_state"] = mx.load(str(opt_path))

        # Load LR scheduler state
        lr_path = temp_path / "lr_scheduler.json"
        if lr_path.exists():
            with open(lr_path) as f:
                result["lr_scheduler_state"] = json.load(f)

    return result
