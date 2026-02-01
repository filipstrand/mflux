"""
Gradient Accumulation for MLX Training.

Accumulates gradients across multiple forward-backward passes before
applying optimizer updates. Critical for training with larger effective
batch sizes on memory-constrained systems.

IMPORTANT: Uses mx.eval() to prevent lazy evaluation graph explosion.
Without forcing evaluation, the computation graph grows unboundedly
across accumulation steps, causing memory issues and slowdowns.
"""

from typing import Any

import mlx.core as mx
from mlx.utils import tree_map


class GradientAccumulator:
    """
    Accumulates gradients over multiple steps before optimizer update.

    This enables larger effective batch sizes by accumulating gradients
    from multiple micro-batches before applying a single optimizer step.

    CRITICAL: Forces evaluation with mx.eval() after each accumulation
    to prevent MLX lazy evaluation graph explosion.

    Args:
        accumulation_steps: Number of gradient accumulation steps.
            Effective batch size = micro_batch_size * accumulation_steps

    Example:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch in dataloader:
            loss, grads = train_step(batch)

            # Accumulate returns averaged grads every 4 steps, None otherwise
            averaged_grads = accumulator.accumulate(grads)

            if averaged_grads is not None:
                optimizer.update(model=model, gradients=averaged_grads)
                lr_scheduler.step()
    """

    def __init__(self, accumulation_steps: int = 1):
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.accumulation_steps = accumulation_steps
        self._accumulated: dict[str, mx.array] | None = None
        self._count: int = 0

    @property
    def count(self) -> int:
        """Number of gradients accumulated in current window."""
        return self._count

    @property
    def is_accumulating(self) -> bool:
        """Whether currently in an accumulation window."""
        return self._count > 0 and self._count < self.accumulation_steps

    def accumulate(self, grads: dict[str, Any]) -> dict[str, Any] | None:
        """
        Accumulate gradients and return averaged gradients when ready.

        Args:
            grads: Gradient dictionary from nn.value_and_grad()

        Returns:
            Averaged gradients if accumulation window complete, None otherwise.
            When returned, the accumulator is reset for the next window.
        """
        if self._accumulated is None:
            # First accumulation: copy gradients (multiply by 1.0 to create new arrays)
            self._accumulated = tree_map(lambda g: g * 1.0, grads)
        else:
            # Add to accumulated gradients
            self._accumulated = tree_map(
                lambda acc, g: acc + g,
                self._accumulated,
                grads,
            )

        # CRITICAL: Force evaluation to prevent lazy graph explosion
        # Without this, the computation graph grows unboundedly across
        # accumulation steps, causing memory issues and slowdowns.
        mx.eval(self._accumulated)

        self._count += 1

        if self._count >= self.accumulation_steps:
            # Window complete: compute average and return
            averaged = tree_map(
                lambda g: g / self.accumulation_steps,
                self._accumulated,
            )
            # Force evaluation before returning
            mx.eval(averaged)

            # Reset for next window
            self._accumulated = None
            self._count = 0

            return averaged

        # Still accumulating
        return None

    def reset(self) -> None:
        """
        Reset accumulator state.

        Call this if training is interrupted mid-window and you want
        to start fresh.
        """
        self._accumulated = None
        self._count = 0

    def flush(self) -> dict[str, Any] | None:
        """
        Flush any partially accumulated gradients.

        Use this at the end of training to apply remaining gradients
        from an incomplete accumulation window.

        Returns:
            Averaged gradients if there were partial accumulations, None otherwise.
            The accumulator is reset after flushing.
        """
        if self._accumulated is None or self._count == 0:
            return None

        # Average over actual accumulated count, not full window
        averaged = tree_map(
            lambda g: g / self._count,
            self._accumulated,
        )
        mx.eval(averaged)

        # Reset for safety
        self._accumulated = None
        self._count = 0

        return averaged

    def state_dict(self) -> dict[str, Any]:
        """
        Return accumulator state for checkpointing.

        WARNING: Accumulated gradients are NOT saved. If checkpointing mid-window
        (count > 0), the partial gradients will be lost on resume. For best results,
        checkpoint at window boundaries (when count == 0).
        """
        if self._count > 0:
            import warnings

            warnings.warn(
                f"Checkpointing gradient accumulator mid-window "
                f"({self._count}/{self.accumulation_steps} steps). "
                f"Partial gradients will be lost on resume.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "accumulation_steps": self.accumulation_steps,
            "count": self._count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Load accumulator state from checkpoint.

        Note: Since accumulated gradients are not saved (to avoid large checkpoints),
        we reset _count to 0 to start a fresh accumulation window. This ensures
        correct gradient averaging and prevents incorrect accumulation from a
        restored _count with no corresponding gradients.
        """
        self.accumulation_steps = state["accumulation_steps"]
        # Always reset count to start fresh window - gradients weren't saved
        # so restoring non-zero count would cause incorrect averaging
        self._count = 0
        self._accumulated = None


class NoOpAccumulator:
    """
    No-op accumulator that simply returns gradients unchanged.

    Use this when accumulation_steps=1 to avoid overhead.
    """

    def __init__(self):
        self.accumulation_steps = 1

    @property
    def count(self) -> int:
        return 0

    @property
    def is_accumulating(self) -> bool:
        return False

    def accumulate(self, grads: dict[str, Any]) -> dict[str, Any]:
        """Return gradients unchanged."""
        return grads

    def reset(self) -> None:
        pass

    def flush(self) -> dict[str, Any] | None:
        """No-op flush - always returns None since we never accumulate."""
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"accumulation_steps": 1, "count": 0}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def create_accumulator(accumulation_steps: int) -> GradientAccumulator | NoOpAccumulator:
    """
    Factory function to create appropriate accumulator.

    Args:
        accumulation_steps: Number of accumulation steps (1 = no accumulation)

    Returns:
        NoOpAccumulator if steps=1, GradientAccumulator otherwise
    """
    if accumulation_steps == 1:
        return NoOpAccumulator()
    return GradientAccumulator(accumulation_steps)
