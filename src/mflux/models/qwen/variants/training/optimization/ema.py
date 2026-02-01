"""
Exponential Moving Average (EMA) for Model Weights.

Maintains an exponentially weighted average of model parameters
during training. EMA models often produce smoother, more stable
outputs than the raw trained model.

Paper: Polyak averaging (1990s), popularized in modern deep learning
by works like progressive GAN training.

IMPORTANT: MLX arrays don't have .copy() method. We use the `p * 1.0`
pattern to create copies of arrays.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten


class EMAModel:
    """
    Exponential Moving Average model for weight smoothing.

    Maintains shadow weights that are an exponential moving average
    of the model's weights during training:
        shadow = decay * shadow + (1 - decay) * current

    Args:
        model: The model whose weights to track
        decay: EMA decay factor (default: 0.9999)
            Higher = slower updates, smoother weights
            Common values: 0.999 (faster), 0.9999 (default), 0.99999 (very slow)

    Example:
        ema = EMAModel(model, decay=0.9999)

        for batch in dataloader:
            loss, grads = train_step(batch)
            optimizer.update(model=model, gradients=grads)

            # Update EMA after each optimizer step
            ema.update(model)

            # For validation, temporarily swap in EMA weights
            ema.apply_shadow(model)
            validate(model)
            ema.restore(model)

        # At end of training, optionally copy EMA weights to model
        ema.copy_to(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        self.decay = decay
        self._backup: dict[str, mx.array] | None = None

        # Initialize shadow weights from model
        # MLX arrays don't have .copy() - use p * 1.0 to create copies
        self._shadow = tree_map(lambda p: p * 1.0, model.parameters())

        # Force evaluation to materialize shadow weights
        mx.eval(self._shadow)

    @property
    def shadow(self) -> dict[str, Any]:
        """Shadow (EMA) weights."""
        return self._shadow

    def update(self, model: nn.Module) -> None:
        """
        Update shadow weights with current model weights.

        shadow = decay * shadow + (1 - decay) * current

        Args:
            model: Model with current weights
        """
        self._shadow = tree_map(
            lambda shadow, current: self.decay * shadow + (1.0 - self.decay) * current,
            self._shadow,
            model.parameters(),
        )

        # Force evaluation to prevent lazy graph buildup
        mx.eval(self._shadow)

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Temporarily swap shadow weights into the model.

        Call restore() after using the model with shadow weights
        to swap back the original weights.

        Args:
            model: Model to apply shadow weights to

        Raises:
            RuntimeError: If called while backup already exists (nested calls)
        """
        if self._backup is not None:
            raise RuntimeError(
                "apply_shadow called while backup exists. Call restore() before calling apply_shadow() again."
            )

        # Backup current weights
        self._backup = tree_map(lambda p: p * 1.0, model.parameters())
        mx.eval(self._backup)

        # Apply shadow weights to model
        self._update_model_params(model, self._shadow)

    def restore(self, model: nn.Module) -> None:
        """
        Restore original weights after apply_shadow().

        Args:
            model: Model to restore original weights to

        Raises:
            RuntimeError: If called without prior apply_shadow()
        """
        if self._backup is None:
            raise RuntimeError("restore() called without prior apply_shadow(). Nothing to restore.")

        # Restore original weights
        self._update_model_params(model, self._backup)
        self._backup = None

    def copy_to(self, model: nn.Module) -> None:
        """
        Permanently copy shadow weights to the model.

        Use at the end of training to finalize with EMA weights.

        Args:
            model: Model to copy shadow weights to
        """
        self._update_model_params(model, self._shadow)

    @staticmethod
    def _update_model_params(model: nn.Module, params: dict[str, Any]) -> None:
        """
        Update model parameters from a parameter dictionary.

        Args:
            model: Model to update
            params: Parameter dictionary matching model structure
        """
        # Flatten both to lists of (name, param) tuples
        model_flat = tree_flatten(model.parameters())
        params_flat = tree_flatten(params)

        if len(model_flat) != len(params_flat):
            raise ValueError(f"Parameter count mismatch: model has {len(model_flat)}, params has {len(params_flat)}")

        # Create update dict
        updates = {}
        for (name, _), (_, new_param) in zip(model_flat, params_flat):
            updates[name] = new_param

        # Use model.update() to apply
        model.update(tree_unflatten(list(updates.items())))
        mx.eval(model.parameters())

    def save(self, path: Path | str) -> None:
        """
        Save EMA shadow weights to file.

        Args:
            path: Path to save weights (safetensors format)
        """
        flat = tree_flatten(self._shadow)
        mx.save_safetensors(str(path), dict(flat))

    @classmethod
    def load(cls, path: Path | str, model: nn.Module, decay: float = 0.9999) -> "EMAModel":
        """
        Load EMA weights from file.

        Args:
            path: Path to saved weights
            model: Model (for structure reference)
            decay: EMA decay factor

        Returns:
            EMAModel with loaded shadow weights
        """
        ema = cls(model, decay=decay)

        # Load and reconstruct shadow weights
        loaded = mx.load(str(path))
        ema._shadow = tree_unflatten(list(loaded.items()))
        mx.eval(ema._shadow)

        return ema

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": self._shadow,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.decay = state["decay"]
        self._shadow = state["shadow"]
        mx.eval(self._shadow)


class NoOpEMA:
    """
    No-op EMA that does nothing.

    Use this when EMA is disabled to avoid conditional checks throughout code.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        pass

    def update(self, model: nn.Module) -> None:
        pass

    def apply_shadow(self, model: nn.Module) -> None:
        pass

    def restore(self, model: nn.Module) -> None:
        pass

    def copy_to(self, model: nn.Module) -> None:
        pass

    def save(self, path: Path | str) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def create_ema(
    model: nn.Module,
    enabled: bool = True,
    decay: float = 0.9999,
) -> EMAModel | NoOpEMA:
    """
    Factory function to create EMA or no-op.

    Args:
        model: Model to track
        enabled: Whether to enable EMA
        decay: EMA decay factor

    Returns:
        EMAModel if enabled, NoOpEMA otherwise
    """
    if enabled:
        return EMAModel(model, decay=decay)
    return NoOpEMA(model, decay=decay)
