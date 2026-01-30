"""
Learning Rate Schedulers for Z-Image Training.

Provides CosineAnnealing, OneCycle, and LinearWarmup schedulers
that integrate with MLX optimizers.
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim


class LRScheduler(ABC):
    """
    Base class for learning rate schedulers.

    Provides a common interface for stepping through learning rate schedules
    and integrating with MLX optimizers.
    """

    def __init__(self, optimizer: optim.Optimizer, initial_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    @abstractmethod
    def get_lr(self) -> float:
        """Calculate the learning rate for the current step."""
        pass

    def step(self) -> float:
        """
        Advance the scheduler by one step and update the optimizer's learning rate.

        The LR is calculated BEFORE incrementing step_count so that:
        - First call returns LR for step 0
        - Second call returns LR for step 1
        - etc.

        Returns:
            The learning rate for the current step (before incrementing)
        """
        # Calculate LR for current step BEFORE incrementing
        lr = self.get_lr()
        self.optimizer.learning_rate = lr
        # Then increment for next call
        self._step_count += 1
        return lr

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "step_count": self._step_count,
            "initial_lr": self.initial_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self._step_count = int(state["step_count"])
        self.initial_lr = float(state["initial_lr"])

    def save(self, path: Path) -> None:
        """Save scheduler state to file."""
        state = self.state_dict()
        mx.save_safetensors(str(path), {k: mx.array([v]) for k, v in state.items()})

    @classmethod
    def load(cls, path: Path, optimizer: optim.Optimizer, **kwargs) -> "LRScheduler":
        """Load scheduler from saved state."""
        loaded = mx.load(str(path))
        state = {k: float(v.item()) for k, v in loaded.items()}
        scheduler = cls(optimizer, initial_lr=state["initial_lr"], **kwargs)
        scheduler.load_state_dict({k: int(v) if k == "step_count" else v for k, v in state.items()})
        return scheduler


class ConstantLR(LRScheduler):
    """
    Constant learning rate scheduler.

    Maintains the initial learning rate throughout training.
    Supports optional warmup period.

    Args:
        optimizer: MLX optimizer to adjust
        initial_lr: Learning rate to maintain
        warmup_steps: Number of linear warmup steps (default: 0)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        warmup_steps: int = 0,
    ):
        super().__init__(optimizer, initial_lr)
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
            # Linear warmup
            if self.warmup_steps == 1:
                return self.initial_lr
            return self.initial_lr * ((self._step_count + 1) / self.warmup_steps)
        return self.initial_lr

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["warmup_steps"] = self.warmup_steps
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.warmup_steps = int(state.get("warmup_steps", 0))


class LinearWarmupLR(LRScheduler):
    """
    Linear warmup scheduler.

    Linearly increases learning rate from 0 to initial_lr over warmup_steps,
    then maintains constant learning rate.

    Args:
        optimizer: MLX optimizer to adjust
        initial_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        warmup_steps: int,
    ):
        super().__init__(optimizer, initial_lr)
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        if self._step_count < self.warmup_steps:
            # Linear warmup: scale from 0 to initial_lr
            # Edge case: warmup_steps=1 means step 0 should get initial_lr directly
            # Use (step + 1) / warmup_steps to ensure first step gets non-zero LR
            if self.warmup_steps == 1:
                return self.initial_lr
            return self.initial_lr * ((self._step_count + 1) / self.warmup_steps)
        return self.initial_lr

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["warmup_steps"] = self.warmup_steps
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.warmup_steps = int(state["warmup_steps"])


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler with optional warmup.

    Learning rate follows a cosine curve from initial_lr to min_lr.
    If warmup_steps > 0, linearly warms up first.

    Formula (after warmup):
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        optimizer: MLX optimizer to adjust
        initial_lr: Peak learning rate (after warmup)
        total_steps: Total number of training steps
        warmup_steps: Number of linear warmup steps (default: 0)
        min_lr: Minimum learning rate at end of schedule (default: 0)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer, initial_lr)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})")

    def get_lr(self) -> float:
        # Linear warmup phase
        if self._step_count < self.warmup_steps:
            # Edge case: warmup_steps=1 means step 0 should get initial_lr directly
            if self.warmup_steps == 1:
                return self.initial_lr
            # Use (step + 1) / warmup_steps to ensure first step gets non-zero LR
            return self.initial_lr * ((self._step_count + 1) / self.warmup_steps)

        # Cosine annealing phase
        progress = (self._step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]

        # Cosine annealing formula
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "min_lr": self.min_lr,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr = float(state["min_lr"])


class OneCycleLR(LRScheduler):
    """
    1cycle learning rate scheduler for super-convergence.

    Implements the 1cycle policy from Smith & Topin (2019):
    - Warmup: Linear increase from initial_lr/div_factor to initial_lr
    - Annealing: Cosine decrease from initial_lr to initial_lr/final_div_factor

    This schedule often achieves better results in fewer epochs.

    Args:
        optimizer: MLX optimizer to adjust
        initial_lr: Maximum learning rate at peak
        total_steps: Total number of training steps
        pct_start: Percentage of cycle spent in warmup (default: 0.3)
        div_factor: Factor to divide initial_lr for starting LR (default: 25)
        final_div_factor: Factor to divide initial_lr for final LR (default: 1e4)

    Example:
        With initial_lr=1e-3, div_factor=25, final_div_factor=1e4:
        - Start LR: 1e-3/25 = 4e-5
        - Peak LR: 1e-3
        - Final LR: 1e-3/1e4 = 1e-7
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        super().__init__(optimizer, initial_lr)
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        if pct_start <= 0 or pct_start >= 1:
            raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")

        if total_steps < 2:
            raise ValueError(f"OneCycleLR requires total_steps >= 2, got {total_steps}")

        # Compute phase boundaries
        self.warmup_steps = int(total_steps * pct_start)
        # Ensure at least 1 step in each phase for proper scheduling
        if self.warmup_steps < 1:
            self.warmup_steps = 1
        if self.warmup_steps >= total_steps:
            self.warmup_steps = total_steps - 1
        self.anneal_steps = total_steps - self.warmup_steps

        # Compute LR boundaries
        self.start_lr = initial_lr / div_factor
        self.final_lr = initial_lr / final_div_factor

    def get_lr(self) -> float:
        if self._step_count < self.warmup_steps:
            # Warmup phase: linear from start_lr to initial_lr
            # Use (step + 1) / warmup_steps to ensure last warmup step reaches initial_lr
            # This matches the behavior of LinearWarmupLR and CosineAnnealingLR
            progress = (self._step_count + 1) / self.warmup_steps
            return self.start_lr + (self.initial_lr - self.start_lr) * progress
        else:
            # Annealing phase: cosine from initial_lr to final_lr
            progress = (self._step_count - self.warmup_steps) / max(1, self.anneal_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "total_steps": self.total_steps,
                "pct_start": self.pct_start,
                "div_factor": self.div_factor,
                "final_div_factor": self.final_div_factor,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.total_steps = int(state["total_steps"])
        self.pct_start = float(state["pct_start"])
        self.div_factor = float(state["div_factor"])
        self.final_div_factor = float(state["final_div_factor"])
        # Recompute derived values with bounds checks
        self.warmup_steps = int(self.total_steps * self.pct_start)
        # Ensure at least 1 step in each phase for proper scheduling
        if self.warmup_steps < 1:
            self.warmup_steps = 1
        if self.warmup_steps >= self.total_steps:
            self.warmup_steps = self.total_steps - 1
        self.anneal_steps = self.total_steps - self.warmup_steps
        self.start_lr = self.initial_lr / self.div_factor
        self.final_lr = self.initial_lr / self.final_div_factor


class NoOpScheduler(LRScheduler):
    """
    No-op scheduler that maintains constant learning rate without warmup.

    Used when scheduler is disabled.
    """

    def __init__(self, optimizer: optim.Optimizer, initial_lr: float):
        super().__init__(optimizer, initial_lr)

    def get_lr(self) -> float:
        return self.initial_lr


def create_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    initial_lr: float,
    total_steps: int,
    **kwargs,
) -> LRScheduler:
    """
    Factory function to create a learning rate scheduler.

    Args:
        name: Scheduler name ("constant", "cosine", "onecycle", or "linear_warmup")
        optimizer: MLX optimizer to adjust
        initial_lr: Peak/target learning rate
        total_steps: Total number of training steps
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Configured LRScheduler instance

    Raises:
        ValueError: If scheduler name is unknown
    """
    name_lower = name.lower().replace("_", "").replace("-", "")

    if name_lower in ("constant", "none", ""):
        return ConstantLR(
            optimizer=optimizer,
            initial_lr=initial_lr,
            warmup_steps=kwargs.get("warmup_steps", 0),
        )
    elif name_lower in ("cosine", "cosineannealing", "cosinelr"):
        return CosineAnnealingLR(
            optimizer=optimizer,
            initial_lr=initial_lr,
            total_steps=total_steps,
            warmup_steps=kwargs.get("warmup_steps", 0),
            min_lr=kwargs.get("min_lr", 0.0),
        )
    elif name_lower in ("onecycle", "1cycle", "onecyclelr"):
        return OneCycleLR(
            optimizer=optimizer,
            initial_lr=initial_lr,
            total_steps=total_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
        )
    elif name_lower in ("linearwarmup", "warmup", "linear"):
        return LinearWarmupLR(
            optimizer=optimizer,
            initial_lr=initial_lr,
            warmup_steps=kwargs.get("warmup_steps", total_steps // 10),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}. Supported: constant, cosine, onecycle, linear_warmup")
