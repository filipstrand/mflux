from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
    BFloat16AdamW,
    MixedPrecisionAdam,
    create_precision_optimizer,
)
from mflux.models.z_image.variants.training.state.training_spec import TrainingSpec
from mflux.models.z_image.variants.training.state.zip_util import ZipUtil


class Optimizer:
    """Optimizer wrapper with state persistence for Z-Image training.

    Supports memory-efficient bf16 momentum storage for AdamW/Adam optimizers,
    providing ~12GB memory savings on 6B parameter models.
    """

    def __init__(self, optimizer: optim.Optimizer | BFloat16AdamW | MixedPrecisionAdam):
        self.optimizer = optimizer

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Optimizer":
        """Create optimizer from training specification.

        Args:
            training_spec: Training specification containing optimizer config

        Returns:
            Optimizer instance with optional bf16 momentum optimization
        """
        opt_spec = training_spec.optimizer

        # Check if bf16 momentum is enabled (default: True for memory savings)
        use_bf16_momentum = getattr(opt_spec, "use_bf16_momentum", True)

        # Create optimizer using precision-aware factory
        optimizer = create_precision_optimizer(
            name=opt_spec.name,
            learning_rate=opt_spec.learning_rate,
            weight_decay=getattr(opt_spec, "weight_decay", 0.01),
            betas=[opt_spec.beta1, opt_spec.beta2],
            eps=opt_spec.eps,
            use_bf16_momentum=use_bf16_momentum,
        )

        opt = Optimizer(optimizer)

        # Load state from checkpoint if available
        if opt_spec.state_path is not None:
            ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=opt_spec.state_path,
                loader=lambda path: opt._load_state(path),
            )

        return opt

    def _load_state(self, path: str) -> None:
        """Load optimizer state from safetensors file.

        Args:
            path: Path to the safetensors state file

        Raises:
            ValueError: If state file is invalid or incompatible
        """
        try:
            state = mx.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load optimizer state from {path}: {e}") from e

        # Validate state is a dictionary
        if not isinstance(state, dict):
            raise ValueError(f"Invalid optimizer state: expected dict, got {type(state).__name__}")

        # Validate state is not empty (empty state indicates corruption)
        if len(state) == 0:
            raise ValueError("Invalid optimizer state: state dictionary is empty")

        self.optimizer.state = state

    def save(self, path: Path) -> None:
        """Save optimizer state to safetensors file.

        For bf16 momentum optimizers, the state is saved with reduced precision,
        resulting in ~25% smaller checkpoints.
        """
        if hasattr(self.optimizer, "save"):
            # Use custom save method for precision optimizers
            self.optimizer.save(path)
        else:
            # Standard MLX optimizer
            mx.save_safetensors(str(path), dict(self.optimizer.state))

    def update(self, model: nn.Module, gradients: dict) -> None:
        """Update model parameters with gradients."""
        self.optimizer.update(model=model, gradients=gradients)

    def zero_grad(self) -> None:
        """Reset gradients (not typically needed in MLX but included for API consistency)."""
        pass

    def get_precision_stats(self) -> dict[str, Any] | None:
        """Get precision optimizer statistics if available.

        Returns:
            Dictionary with precision stats, or None for standard optimizers
        """
        if hasattr(self.optimizer, "get_stats"):
            return self.optimizer.get_stats()
        return None
