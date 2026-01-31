"""Best checkpoint selection for Z-Image training.

Provides automatic selection and retention of checkpoints based on
validation metrics (loss, CLIP score, etc.).

Features:
- Keep best N checkpoints by any metric
- Automatic deletion of worse checkpoints
- Metric history tracking for trend analysis
- Support for both minimization (loss) and maximization (CLIP score)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointRecord:
    """Record of a single checkpoint with its metrics."""

    step: int
    path: Path
    validation_loss: float | None = None
    clip_score: float | None = None
    timestamp: str | None = None

    def get_metric(self, metric_name: str) -> float | None:
        """Get a specific metric value."""
        if metric_name == "validation_loss":
            return self.validation_loss
        elif metric_name == "clip_score":
            return self.clip_score
        return None


@dataclass
class BestCheckpointSelector:
    """Selects and retains the best checkpoints by a given metric.

    Keeps track of all checkpoints and their metrics, automatically
    identifying which checkpoints should be deleted when new ones are added.

    Usage:
        selector = BestCheckpointSelector(
            keep_best_n=3,
            metric="validation_loss",
            higher_is_better=False,  # Lower loss is better
        )

        # After each checkpoint save:
        paths_to_delete = selector.record_checkpoint(
            step=1000,
            path=Path("/checkpoints/step_1000.zip"),
            validation_loss=0.05,
        )

        # Delete the returned paths
        for path in paths_to_delete:
            path.unlink()
    """

    keep_best_n: int
    metric: str = "validation_loss"
    higher_is_better: bool = False
    checkpoints: list[CheckpointRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.keep_best_n < 1:
            raise ValueError(f"keep_best_n must be >= 1, got {self.keep_best_n}")
        if self.metric not in ("validation_loss", "clip_score"):
            raise ValueError(f"Unsupported metric: {self.metric}")

    def record_checkpoint(
        self,
        step: int,
        path: Path,
        validation_loss: float | None = None,
        clip_score: float | None = None,
        timestamp: str | None = None,
    ) -> list[Path]:
        """Record a new checkpoint and return paths to delete.

        Args:
            step: Training step number
            path: Path to checkpoint file
            validation_loss: Validation loss at this checkpoint
            clip_score: CLIP score at this checkpoint
            timestamp: Checkpoint creation timestamp

        Returns:
            List of checkpoint paths that should be deleted
            (not among the best N).
        """
        record = CheckpointRecord(
            step=step,
            path=path,
            validation_loss=validation_loss,
            clip_score=clip_score,
            timestamp=timestamp,
        )
        self.checkpoints.append(record)

        # Determine which checkpoints to keep
        return self._compute_deletions()

    def _compute_deletions(self) -> list[Path]:
        """Compute which checkpoints should be deleted.

        Returns:
            List of paths to delete.
        """
        if len(self.checkpoints) <= self.keep_best_n:
            return []

        # Sort by metric (handling None values)
        def sort_key(record: CheckpointRecord) -> tuple[bool, float]:
            metric_value = record.get_metric(self.metric)
            if metric_value is None:
                # None values go to the end (will be deleted first)
                return (True, 0.0)
            return (False, -metric_value if self.higher_is_better else metric_value)

        sorted_checkpoints = sorted(self.checkpoints, key=sort_key)

        # Keep best N
        to_keep = set(r.path for r in sorted_checkpoints[: self.keep_best_n])
        to_delete = [r.path for r in self.checkpoints if r.path not in to_keep]

        # Update internal state to only keep the best
        self.checkpoints = [r for r in self.checkpoints if r.path in to_keep]

        return to_delete

    def get_best_checkpoint(self) -> CheckpointRecord | None:
        """Get the best checkpoint by the configured metric.

        Returns:
            Best CheckpointRecord or None if no checkpoints.
        """
        if not self.checkpoints:
            return None

        def sort_key(record: CheckpointRecord) -> tuple[bool, float]:
            metric_value = record.get_metric(self.metric)
            if metric_value is None:
                return (True, 0.0)
            return (False, -metric_value if self.higher_is_better else metric_value)

        return min(self.checkpoints, key=sort_key)

    def get_checkpoint_history(self) -> list[dict[str, Any]]:
        """Get history of all tracked checkpoints.

        Returns:
            List of checkpoint dictionaries with step and metrics.
        """
        return [
            {
                "step": r.step,
                "path": str(r.path),
                "validation_loss": r.validation_loss,
                "clip_score": r.clip_score,
                "timestamp": r.timestamp,
            }
            for r in self.checkpoints
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize selector state for checkpointing.

        Returns:
            Dictionary that can be JSON-serialized.
        """
        return {
            "keep_best_n": self.keep_best_n,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "checkpoints": [
                {
                    "step": r.step,
                    "path": str(r.path),
                    "validation_loss": r.validation_loss,
                    "clip_score": r.clip_score,
                    "timestamp": r.timestamp,
                }
                for r in self.checkpoints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BestCheckpointSelector":
        """Restore selector from serialized state.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored BestCheckpointSelector instance.
        """
        selector = cls(
            keep_best_n=data["keep_best_n"],
            metric=data["metric"],
            higher_is_better=data["higher_is_better"],
        )

        for cp_data in data.get("checkpoints", []):
            record = CheckpointRecord(
                step=cp_data["step"],
                path=Path(cp_data["path"]),
                validation_loss=cp_data.get("validation_loss"),
                clip_score=cp_data.get("clip_score"),
                timestamp=cp_data.get("timestamp"),
            )
            selector.checkpoints.append(record)

        return selector


def create_checkpoint_selector(
    keep_best_n: int | None = None,
    metric: str = "validation_loss",
    higher_is_better: bool | None = None,
) -> BestCheckpointSelector | None:
    """Factory function to create checkpoint selector.

    Args:
        keep_best_n: Number of best checkpoints to keep (None = disabled)
        metric: Metric name for comparison
        higher_is_better: Whether higher metric is better (auto-detected if None)

    Returns:
        BestCheckpointSelector or None if disabled.
    """
    if keep_best_n is None or keep_best_n <= 0:
        return None

    # Auto-detect higher_is_better based on metric
    if higher_is_better is None:
        higher_is_better = metric in ("clip_score",)  # CLIP score = higher is better

    return BestCheckpointSelector(
        keep_best_n=keep_best_n,
        metric=metric,
        higher_is_better=higher_is_better,
    )
