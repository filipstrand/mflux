"""Precision-optimized optimizer wrappers for memory-efficient training.

This module provides optimizer wrappers that use mixed precision for
optimizer state to reduce memory usage during training.

Memory Impact:
- Standard AdamW: 6B params × 8 bytes (fp32 m + fp32 v) = 48GB
- BFloat16AdamW: 6B params × 6 bytes (bf16 m + fp32 v) = 36GB
- Savings: ~12GB (enables larger batch sizes)

Note: Variance (v) stays in fp32 for numerical stability, while
momentum (m) can safely use bf16 with minimal quality impact.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn


class BFloat16AdamW:
    """AdamW wrapper with bf16 momentum storage for memory efficiency.

    This wrapper stores the first moment (momentum) in bfloat16 precision
    while keeping the second moment (variance) in float32 for stability.

    The bfloat16 format has:
    - 1 sign bit
    - 8 exponent bits (same as float32)
    - 7 mantissa bits (vs 23 in float32)

    This provides the same dynamic range as float32 but with reduced
    precision, which is acceptable for momentum since it's a running
    average that doesn't require high precision.

    Memory savings: ~12GB for 6B parameter model

    Usage:
        optimizer = BFloat16AdamW(
            learning_rate=1e-4,
            weight_decay=0.01,
            momentum_precision=mx.bfloat16,
        )
        optimizer.update(model, gradients)

    Attributes:
        optimizer: Underlying MLX AdamW optimizer
        momentum_precision: Precision for momentum storage (default bf16)
        _cast_momentum: Whether to cast momentum after updates
    """

    # Supported precision types for momentum storage
    SUPPORTED_PRECISIONS = {mx.bfloat16, mx.float16, mx.float32}

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: list[float] | None = None,
        eps: float = 1e-8,
        momentum_precision: mx.Dtype = mx.bfloat16,
    ):
        """Initialize bf16 AdamW optimizer.

        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            betas: Beta coefficients [beta1, beta2] for momentum and variance
            eps: Epsilon for numerical stability
            momentum_precision: Precision for momentum storage (default bf16)

        Raises:
            ValueError: If learning_rate, weight_decay, or eps are invalid
            TypeError: If momentum_precision is not a supported dtype
        """
        # Validate hyperparameters
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        if betas is None:
            betas = [0.9, 0.999]
        if len(betas) != 2:
            raise ValueError(f"betas must have exactly 2 elements, got {len(betas)}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"betas must be in range [0, 1), got {betas}")

        # Validate precision type
        if momentum_precision not in self.SUPPORTED_PRECISIONS:
            supported_str = ", ".join(str(p) for p in self.SUPPORTED_PRECISIONS)
            raise TypeError(f"momentum_precision must be one of {supported_str}, got {momentum_precision}")

        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.momentum_precision = momentum_precision
        self._cast_momentum = momentum_precision != mx.float32
        self._total_casts = 0

    @property
    def state(self) -> dict[str, Any]:
        """Access underlying optimizer state."""
        return self.optimizer.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Set underlying optimizer state."""
        self.optimizer.state = value
        # Cast momentum to target precision after loading
        if self._cast_momentum:
            self._cast_momentum_to_precision()

    def update(self, model: nn.Module, gradients: dict) -> None:
        """Update model parameters with gradients.

        After the optimizer update, casts momentum to bf16 precision.
        This happens after each update to maintain memory savings.

        Args:
            model: Model to update
            gradients: Gradient dictionary from value_and_grad
        """
        # Standard optimizer update
        self.optimizer.update(model=model, gradients=gradients)

        # Cast momentum to target precision
        if self._cast_momentum:
            self._cast_momentum_to_precision()

    def _cast_momentum_to_precision(self) -> None:
        """Cast momentum values to target precision.

        Iterates through optimizer state and casts 'm' (momentum)
        values to the configured precision.

        Note: We only cast 'm' (first moment), not 'v' (second moment)
        to maintain numerical stability in variance computation.
        """
        for key in self.optimizer.state:
            state_entry = self.optimizer.state[key]
            if isinstance(state_entry, dict) and "m" in state_entry:
                m = state_entry["m"]
                if m.dtype != self.momentum_precision:
                    state_entry["m"] = m.astype(self.momentum_precision)
                    self._total_casts += 1

    def save(self, path: Path) -> None:
        """Save optimizer state to safetensors file.

        Note: State is saved as-is (with bf16 momentum), which reduces
        checkpoint size by ~25%.
        """
        mx.save_safetensors(str(path), dict(self.optimizer.state))

    def get_stats(self) -> dict:
        """Get precision optimizer statistics.

        Returns:
            Dictionary with cast statistics
        """
        # Calculate memory estimate
        state = self.optimizer.state
        num_momentum_arrays = sum(1 for key in state if isinstance(state[key], dict) and "m" in state[key])

        return {
            "total_casts": self._total_casts,
            "momentum_precision": str(self.momentum_precision),
            "num_momentum_arrays": num_momentum_arrays,
            "estimated_savings_gb": num_momentum_arrays * 0.002,  # Rough estimate per array
        }


class MixedPrecisionAdam:
    """Adam optimizer with configurable precision for momentum and variance.

    Provides full control over precision for both optimizer state components.

    Args:
        learning_rate: Learning rate
        betas: Beta coefficients
        eps: Epsilon for numerical stability
        momentum_precision: Precision for first moment (m)
        variance_precision: Precision for second moment (v)

    Raises:
        ValueError: If learning_rate or eps are invalid
        TypeError: If precision types are not supported
    """

    # Supported precision types
    SUPPORTED_PRECISIONS = {mx.bfloat16, mx.float16, mx.float32}

    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: list[float] | None = None,
        eps: float = 1e-8,
        momentum_precision: mx.Dtype = mx.bfloat16,
        variance_precision: mx.Dtype = mx.float32,
    ):
        # Validate hyperparameters
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        if betas is None:
            betas = [0.9, 0.999]
        if len(betas) != 2:
            raise ValueError(f"betas must have exactly 2 elements, got {len(betas)}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"betas must be in range [0, 1), got {betas}")

        # Validate precision types
        for name, precision in [("momentum_precision", momentum_precision), ("variance_precision", variance_precision)]:
            if precision not in self.SUPPORTED_PRECISIONS:
                supported_str = ", ".join(str(p) for p in self.SUPPORTED_PRECISIONS)
                raise TypeError(f"{name} must be one of {supported_str}, got {precision}")

        self.optimizer = optim.Adam(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
        )
        self.momentum_precision = momentum_precision
        self.variance_precision = variance_precision

    @property
    def state(self) -> dict[str, Any]:
        return self.optimizer.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self.optimizer.state = value
        self._cast_state_to_precision()

    def update(self, model: nn.Module, gradients: dict) -> None:
        self.optimizer.update(model=model, gradients=gradients)
        self._cast_state_to_precision()

    def _cast_state_to_precision(self) -> None:
        for key in self.optimizer.state:
            state_entry = self.optimizer.state[key]
            if isinstance(state_entry, dict):
                if "m" in state_entry and state_entry["m"].dtype != self.momentum_precision:
                    state_entry["m"] = state_entry["m"].astype(self.momentum_precision)
                if "v" in state_entry and state_entry["v"].dtype != self.variance_precision:
                    state_entry["v"] = state_entry["v"].astype(self.variance_precision)

    def save(self, path: Path) -> None:
        mx.save_safetensors(str(path), dict(self.optimizer.state))


def create_precision_optimizer(
    name: str,
    learning_rate: float,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    use_bf16_momentum: bool = True,
) -> BFloat16AdamW | optim.Optimizer:
    """Factory function to create optimizer with optional precision optimization.

    Args:
        name: Optimizer name ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay (for AdamW)
        betas: Beta coefficients
        eps: Epsilon for stability
        use_bf16_momentum: Whether to use bf16 for momentum (saves ~12GB)

    Returns:
        Optimizer instance (BFloat16AdamW, MixedPrecisionAdam, or standard)

    Raises:
        ValueError: If optimizer name is not supported
    """
    if betas is None:
        betas = [0.9, 0.999]

    name_lower = name.lower()

    if name_lower == "adamw":
        if use_bf16_momentum:
            return BFloat16AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                momentum_precision=mx.bfloat16,
            )
        return optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    elif name_lower == "adam":
        if use_bf16_momentum:
            return MixedPrecisionAdam(
                learning_rate=learning_rate,
                betas=betas,
                eps=eps,
                momentum_precision=mx.bfloat16,
                variance_precision=mx.float32,
            )
        return optim.Adam(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
        )

    elif name_lower == "sgd":
        # SGD doesn't have momentum state that benefits from bf16
        return optim.SGD(learning_rate=learning_rate)

    else:
        raise ValueError(f"Unsupported optimizer: {name}")
