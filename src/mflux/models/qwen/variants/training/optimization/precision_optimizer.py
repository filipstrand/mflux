"""Precision-optimized optimizer wrappers for Qwen training.

This module provides optimizer wrappers that use mixed precision for
optimizer state to reduce memory usage during training.

Memory Impact (Qwen-Image 6B model):
- Standard AdamW: 6B params x 8 bytes (fp32 m + fp32 v) = 48GB
- BFloat16AdamW: 6B params x 6 bytes (bf16 m + fp32 v) = 36GB
- Savings: ~12GB (enables larger batch sizes)

Note: Variance (v) stays in fp32 for numerical stability, while
momentum (m) can safely use bf16 with minimal quality impact.

Based on Z-Image precision_optimizer.py with Qwen-specific optimizations.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

# Supported precision types for optimizer state storage
SUPPORTED_PRECISIONS = {mx.bfloat16, mx.float16, mx.float32}


def _validate_learning_rate(learning_rate: float) -> None:
    """Validate learning rate is positive."""
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")


def _validate_eps(eps: float) -> None:
    """Validate epsilon is positive."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


def _validate_betas(betas: list[float] | None) -> list[float]:
    """Validate and return betas with defaults."""
    if betas is None:
        betas = [0.9, 0.999]
    if len(betas) != 2:
        raise ValueError(f"betas must have exactly 2 elements, got {len(betas)}")
    if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
        raise ValueError(f"betas must be in range [0, 1), got {betas}")
    return betas


def _validate_precision(precision: mx.Dtype, name: str = "precision") -> None:
    """Validate precision is a supported dtype."""
    if precision not in SUPPORTED_PRECISIONS:
        supported_str = ", ".join(str(p) for p in SUPPORTED_PRECISIONS)
        raise TypeError(f"{name} must be one of {supported_str}, got {precision}")


def _validate_weight_decay(weight_decay: float) -> None:
    """Validate weight decay is non-negative."""
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")


class QwenBFloat16AdamW:
    """AdamW wrapper with bf16 momentum storage for Qwen training.

    This wrapper stores the first moment (momentum) in bfloat16 precision
    while keeping the second moment (variance) in float32 for stability.

    The bfloat16 format has:
    - 1 sign bit
    - 8 exponent bits (same as float32)
    - 7 mantissa bits (vs 23 in float32)

    This provides the same dynamic range as float32 but with reduced
    precision, which is acceptable for momentum since it's a running
    average that doesn't require high precision.

    Memory savings: ~12GB for Qwen 6B parameter model

    Usage:
        optimizer = QwenBFloat16AdamW(
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

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: list[float] | None = None,
        eps: float = 1e-8,
        momentum_precision: mx.Dtype = mx.bfloat16,
    ):
        """Initialize bf16 AdamW optimizer for Qwen.

        Args:
            learning_rate: Learning rate (default 1e-4, typical for fine-tuning)
            weight_decay: Weight decay coefficient (default 0.01)
            betas: Beta coefficients [beta1, beta2] for momentum and variance
            eps: Epsilon for numerical stability
            momentum_precision: Precision for momentum storage (default bf16)

        Raises:
            ValueError: If learning_rate, weight_decay, or eps are invalid
            TypeError: If momentum_precision is not a supported dtype
        """
        _validate_learning_rate(learning_rate)
        _validate_weight_decay(weight_decay)
        _validate_eps(eps)
        betas = _validate_betas(betas)
        _validate_precision(momentum_precision, "momentum_precision")

        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.momentum_precision = momentum_precision
        self._cast_momentum = momentum_precision != mx.float32
        self._total_casts = 0
        # Cache keys that have momentum entries to avoid repeated dict inspection
        self._momentum_keys: set[str] | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Access underlying optimizer state."""
        return self.optimizer.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Set underlying optimizer state."""
        self.optimizer.state = value
        self._momentum_keys = None  # Invalidate cache on state change
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
        self.optimizer.update(model=model, gradients=gradients)
        if self._cast_momentum:
            self._cast_momentum_to_precision()

    def _cast_momentum_to_precision(self) -> None:
        """Cast momentum values to target precision.

        Uses cached key list to avoid repeated dict inspection.

        Note: We only cast 'm' (first moment), not 'v' (second moment)
        to maintain numerical stability in variance computation.
        """
        state = self.optimizer.state

        # Build cache on first call or after state change
        if self._momentum_keys is None:
            self._momentum_keys = {key for key in state if isinstance(state[key], dict) and "m" in state[key]}

        # Only iterate over keys known to have momentum
        for key in self._momentum_keys:
            state_entry = state[key]
            m = state_entry["m"]
            if m.dtype != self.momentum_precision:
                state_entry["m"] = m.astype(self.momentum_precision)
                self._total_casts += 1

    def save(self, path: Path) -> None:
        """Save optimizer state to safetensors file.

        Note: State is saved as-is (with bf16 momentum), which reduces
        checkpoint size by ~25%.

        Args:
            path: Destination path for the checkpoint file
        """
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), dict(self.optimizer.state))

    def get_stats(self) -> dict:
        """Get precision optimizer statistics.

        Returns:
            Dictionary with cast statistics and memory estimates
        """
        state = self.optimizer.state
        num_momentum_arrays = sum(1 for key in state if isinstance(state[key], dict) and "m" in state[key])

        # Each momentum array saves ~2MB when using bf16 vs fp32
        # Calculation: typical layer has ~1M params, bf16 saves 2 bytes/param = 2MB
        ESTIMATED_SAVINGS_PER_ARRAY_GB = 0.002

        return {
            "total_casts": self._total_casts,
            "momentum_precision": str(self.momentum_precision),
            "num_momentum_arrays": num_momentum_arrays,
            "estimated_savings_gb": num_momentum_arrays * ESTIMATED_SAVINGS_PER_ARRAY_GB,
        }


class QwenMixedPrecisionAdam:
    """Adam optimizer with configurable precision for Qwen training.

    Provides full control over precision for both optimizer state components.
    Use this when you need finer control than QwenBFloat16AdamW provides.

    Args:
        learning_rate: Learning rate
        betas: Beta coefficients
        eps: Epsilon for numerical stability
        momentum_precision: Precision for first moment (m)
        variance_precision: Precision for second moment (v)
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: list[float] | None = None,
        eps: float = 1e-8,
        momentum_precision: mx.Dtype = mx.bfloat16,
        variance_precision: mx.Dtype = mx.float32,
    ):
        _validate_learning_rate(learning_rate)
        _validate_eps(eps)
        betas = _validate_betas(betas)
        _validate_precision(momentum_precision, "momentum_precision")
        _validate_precision(variance_precision, "variance_precision")

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
        """Save optimizer state to safetensors file."""
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), dict(self.optimizer.state))


def create_qwen_optimizer(
    name: str,
    learning_rate: float,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    use_bf16_momentum: bool = True,
) -> QwenBFloat16AdamW | optim.Optimizer:
    """Factory function to create optimizer for Qwen training.

    Args:
        name: Optimizer name ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay (for AdamW)
        betas: Beta coefficients
        eps: Epsilon for stability
        use_bf16_momentum: Whether to use bf16 for momentum (saves ~12GB)

    Returns:
        Optimizer instance (QwenBFloat16AdamW, QwenMixedPrecisionAdam, or standard)

    Raises:
        ValueError: If optimizer name is not supported
    """
    if betas is None:
        betas = [0.9, 0.999]

    name_lower = name.lower()

    if name_lower == "adamw":
        if use_bf16_momentum:
            return QwenBFloat16AdamW(
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
            return QwenMixedPrecisionAdam(
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
        return optim.SGD(learning_rate=learning_rate)

    else:
        raise ValueError(f"Unsupported optimizer: {name}")


class QwenGradientAccumulator:
    """Accumulate gradients across micro-batches for Qwen training.

    For effective batch sizes larger than GPU memory allows, accumulate
    gradients over multiple forward/backward passes before optimizer step.

    Uses FP32 accumulation for numerical stability even when individual
    gradients are in FP16/BF16.

    Usage:
        accumulator = QwenGradientAccumulator(accumulation_steps=4)

        for step, batch in enumerate(dataloader):
            gradients = compute_gradients(model, batch)
            accumulator.accumulate(gradients)

            if accumulator.should_step():
                final_grads = accumulator.get_and_reset()
                optimizer.update(model, final_grads)
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        accumulation_dtype: mx.Dtype = mx.float32,
    ):
        """Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps before optimizer update
            accumulation_dtype: Dtype for accumulated gradients

        Raises:
            ValueError: If accumulation_steps < 1
        """
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.accumulation_steps = accumulation_steps
        self.accumulation_dtype = accumulation_dtype
        self._accumulated_grads: dict[str, mx.array | dict] | None = None
        self._current_step = 0
        # Cache: set of (path_tuple, needs_cast) for fast gradient addition
        self._needs_cast_cache: dict[tuple[str, ...], bool] | None = None

    def accumulate(self, gradients: dict) -> None:
        """Accumulate gradients from a micro-batch."""
        if self._accumulated_grads is None:
            self._accumulated_grads, self._needs_cast_cache = self._init_accumulation(gradients)
        else:
            self._add_gradients(gradients)

        self._current_step += 1

    def _init_accumulation(self, gradients: dict) -> tuple[dict, dict[tuple[str, ...], bool]]:
        """Initialize accumulation with first gradient batch.

        Returns:
            Tuple of (accumulated gradients dict, cast cache dict)
        """
        needs_cast: dict[tuple[str, ...], bool] = {}

        def init_recursive(grads: dict, path: tuple[str, ...] = ()) -> dict:
            result = {}
            for key, value in grads.items():
                current_path = path + (key,)
                if isinstance(value, dict):
                    result[key] = init_recursive(value, current_path)
                elif isinstance(value, mx.array):
                    needs_conversion = value.dtype != self.accumulation_dtype
                    needs_cast[current_path] = needs_conversion
                    if needs_conversion:
                        result[key] = value.astype(self.accumulation_dtype)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return result

        accumulated = init_recursive(gradients)
        return accumulated, needs_cast

    def _add_gradients(self, gradients: dict) -> None:
        """Add gradients to accumulator using cached dtype info."""

        def add_recursive(accumulated: dict, new: dict, path: tuple[str, ...] = ()) -> None:
            for key, value in new.items():
                current_path = path + (key,)
                if isinstance(value, dict):
                    add_recursive(accumulated[key], value, current_path)
                elif isinstance(value, mx.array):
                    # Use cached info to avoid repeated dtype checks
                    needs_cast = self._needs_cast_cache.get(current_path, False)
                    if needs_cast:
                        accumulated[key] = accumulated[key] + value.astype(self.accumulation_dtype)
                    else:
                        accumulated[key] = accumulated[key] + value

        add_recursive(self._accumulated_grads, gradients)

    def should_step(self) -> bool:
        """Check if it's time for optimizer step."""
        return self._current_step >= self.accumulation_steps

    def get_and_reset(self) -> dict:
        """Get accumulated gradients and reset accumulator."""
        if self._accumulated_grads is None:
            raise RuntimeError("get_and_reset() called before any gradients were accumulated.")

        # Guard against divide-by-zero (shouldn't happen if _accumulated_grads exists, but be safe)
        if self._current_step < 1:
            raise RuntimeError("get_and_reset() called with zero steps accumulated.")

        scale = 1.0 / self._current_step

        def scale_recursive(grads: dict) -> dict:
            result = {}
            for key, value in grads.items():
                if isinstance(value, dict):
                    result[key] = scale_recursive(value)
                elif isinstance(value, mx.array):
                    result[key] = value * scale
                else:
                    result[key] = value
            return result

        averaged = scale_recursive(self._accumulated_grads)

        self._accumulated_grads = None
        self._current_step = 0
        # Keep cache for next accumulation cycle (structure likely unchanged)

        return averaged

    @property
    def current_step(self) -> int:
        """Get current accumulation step."""
        return self._current_step

    def reset(self) -> None:
        """Reset accumulator without returning gradients."""
        self._accumulated_grads = None
        self._current_step = 0
        # Keep cache for potential reuse (gradient structure typically stable)
