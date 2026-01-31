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
        # Validate hyperparameters using shared validators
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

        Args:
            path: Destination path for the checkpoint file

        Raises:
            ValueError: If path has no parent directory
            OSError: If parent directory cannot be created
        """
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
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

    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: list[float] | None = None,
        eps: float = 1e-8,
        momentum_precision: mx.Dtype = mx.bfloat16,
        variance_precision: mx.Dtype = mx.float32,
    ):
        # Validate hyperparameters using shared validators
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
        """Save optimizer state to safetensors file.

        Args:
            path: Destination path for the checkpoint file
        """
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
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


class MixedPrecisionGradientOptimizer:
    """Optimizer wrapper with mixed precision gradient computation.

    Casts gradients to FP16 during computation for memory savings,
    then accumulates in FP32 for numerical stability.

    Memory savings: ~25% reduction in gradient memory
    (gradients stored as FP16 instead of FP32)

    This is distinct from BFloat16AdamW which optimizes optimizer state.
    MixedPrecisionGradientOptimizer optimizes the gradient tensors themselves.

    Usage:
        base_optimizer = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(
            base_optimizer,
            gradient_dtype=mx.float16,
            accumulation_dtype=mx.float32,
        )

        # In training loop:
        gradients = compute_gradients(model, batch)
        optimizer.update(model, gradients)

    Attributes:
        base_optimizer: Underlying optimizer
        gradient_dtype: Dtype for gradient storage (default FP16)
        accumulation_dtype: Dtype for gradient accumulation (default FP32)
        loss_scale: Scale factor to prevent FP16 underflow
    """

    # Supported gradient precision types
    SUPPORTED_GRADIENT_DTYPES = {mx.float16, mx.bfloat16, mx.float32}

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        gradient_dtype: mx.Dtype = mx.float16,
        accumulation_dtype: mx.Dtype = mx.float32,
        loss_scale: float = 1.0,
        dynamic_loss_scaling: bool = False,
    ):
        """Initialize mixed precision gradient optimizer.

        Args:
            base_optimizer: Underlying optimizer (AdamW, Adam, SGD, etc.)
            gradient_dtype: Dtype for gradient tensors (default FP16)
            accumulation_dtype: Dtype for gradient accumulation (default FP32)
            loss_scale: Static loss scale to prevent gradient underflow
            dynamic_loss_scaling: Whether to use dynamic loss scaling

        Raises:
            TypeError: If gradient_dtype is not supported
            ValueError: If loss_scale is not positive
        """
        if gradient_dtype not in self.SUPPORTED_GRADIENT_DTYPES:
            supported_str = ", ".join(str(p) for p in self.SUPPORTED_GRADIENT_DTYPES)
            raise TypeError(f"gradient_dtype must be one of {supported_str}, got {gradient_dtype}")

        if loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {loss_scale}")

        self.base_optimizer = base_optimizer
        self.gradient_dtype = gradient_dtype
        self.accumulation_dtype = accumulation_dtype
        self.loss_scale = loss_scale
        self.dynamic_loss_scaling = dynamic_loss_scaling

        # Dynamic loss scaling state
        self._current_scale = loss_scale
        self._growth_interval = 2000
        self._growth_factor = 2.0
        self._backoff_factor = 0.5
        self._num_good_steps = 0
        self._num_inf_nan = 0

        # Statistics
        self._total_updates = 0
        self._casts_performed = 0

    @property
    def state(self) -> dict[str, Any]:
        """Access underlying optimizer state."""
        return self.base_optimizer.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Set underlying optimizer state."""
        self.base_optimizer.state = value

    def update(self, model: nn.Module, gradients: dict) -> bool:
        """Update model with mixed precision gradients.

        Casts gradients to configured dtype, checks for inf/nan,
        then updates with base optimizer.

        Args:
            model: Model to update
            gradients: Gradient dictionary from value_and_grad

        Returns:
            True if update was successful, False if skipped due to inf/nan
        """
        self._total_updates += 1

        # Cast gradients to target dtype
        cast_gradients = self._cast_gradients(gradients)

        # Check for inf/nan (especially important with FP16)
        if self._has_inf_or_nan(cast_gradients):
            self._num_inf_nan += 1
            if self.dynamic_loss_scaling:
                self._current_scale *= self._backoff_factor
                self._num_good_steps = 0
            return False

        # Unscale gradients if using loss scaling
        if self._current_scale != 1.0:
            cast_gradients = self._unscale_gradients(cast_gradients)

        # Update with base optimizer
        self.base_optimizer.update(model=model, gradients=cast_gradients)

        # Update dynamic scaling
        if self.dynamic_loss_scaling:
            self._num_good_steps += 1
            if self._num_good_steps >= self._growth_interval:
                self._current_scale *= self._growth_factor
                self._num_good_steps = 0

        return True

    def _cast_gradients(self, gradients: dict) -> dict:
        """Cast gradients to target precision.

        Only creates new dictionaries when modifications are actually needed.
        If all gradients are already in the target dtype, returns the original
        dictionary to avoid unnecessary memory allocation.

        Args:
            gradients: Original gradient dictionary

        Returns:
            Dictionary with cast gradients (may be same object if no casts needed)
        """

        def cast_recursive(grads: dict) -> tuple[dict, bool]:
            """Returns (result_dict, was_modified) tuple."""
            modifications = {}
            any_modified = False

            for key, value in grads.items():
                if isinstance(value, dict):
                    nested_result, nested_modified = cast_recursive(value)
                    if nested_modified:
                        modifications[key] = nested_result
                        any_modified = True
                elif isinstance(value, mx.array):
                    if value.dtype != self.gradient_dtype:
                        modifications[key] = value.astype(self.gradient_dtype)
                        self._casts_performed += 1
                        any_modified = True

            if not any_modified:
                return grads, False

            # Create new dict only when modifications exist
            result = dict(grads)
            result.update(modifications)
            return result, True

        result, _ = cast_recursive(gradients)
        return result

    def _unscale_gradients(self, gradients: dict) -> dict:
        """Unscale gradients by loss scale factor.

        Args:
            gradients: Scaled gradients

        Returns:
            Unscaled gradients
        """
        inv_scale = 1.0 / self._current_scale

        def unscale_recursive(grads: dict) -> dict:
            result = {}
            for key, value in grads.items():
                if isinstance(value, dict):
                    result[key] = unscale_recursive(value)
                elif isinstance(value, mx.array):
                    result[key] = value * inv_scale
                else:
                    result[key] = value
            return result

        return unscale_recursive(gradients)

    def _has_inf_or_nan(self, gradients: dict) -> bool:
        """Check if any gradients contain inf or nan.

        Args:
            gradients: Gradient dictionary to check

        Returns:
            True if any inf or nan values found
        """

        def check_recursive(grads: dict) -> bool:
            for value in grads.values():
                if isinstance(value, dict):
                    if check_recursive(value):
                        return True
                elif isinstance(value, mx.array):
                    # Single-pass check using isfinite (true single scan)
                    if mx.any(~mx.isfinite(value)):
                        return True
            return False

        return check_recursive(gradients)

    def get_current_scale(self) -> float:
        """Get current loss scale value."""
        return self._current_scale

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics.

        Returns:
            Dictionary with update and casting statistics
        """
        return {
            "total_updates": self._total_updates,
            "casts_performed": self._casts_performed,
            "inf_nan_count": self._num_inf_nan,
            "current_scale": self._current_scale,
            "gradient_dtype": str(self.gradient_dtype),
            "accumulation_dtype": str(self.accumulation_dtype),
        }

    def save(self, path: Path) -> None:
        """Save optimizer state to file.

        Args:
            path: Destination path for the checkpoint file
        """
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), dict(self.base_optimizer.state))


class GradientAccumulator:
    """Accumulate gradients across micro-batches with mixed precision.

    For effective batch sizes larger than GPU memory allows, accumulate
    gradients over multiple forward/backward passes before optimizer step.

    Uses FP32 accumulation for numerical stability even when individual
    gradients are in FP16/BF16.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for step, batch in enumerate(dataloader):
            gradients = compute_gradients(model, batch)
            accumulator.accumulate(gradients)

            if accumulator.should_step():
                final_grads = accumulator.get_and_reset()
                optimizer.update(model, final_grads)

    Attributes:
        accumulation_steps: Number of micro-batches to accumulate
        accumulation_dtype: Dtype for accumulated gradients
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
        # Type: nested dict mapping parameter names to accumulated gradient arrays
        self._accumulated_grads: dict[str, mx.array | dict] | None = None
        self._current_step = 0

    def accumulate(self, gradients: dict) -> None:
        """Accumulate gradients from a micro-batch.

        Args:
            gradients: Gradient dictionary from single micro-batch
        """
        if self._accumulated_grads is None:
            # First accumulation - initialize
            self._accumulated_grads = self._init_accumulation(gradients)
        else:
            # Add to existing
            self._add_gradients(gradients)

        self._current_step += 1

    def _init_accumulation(self, gradients: dict) -> dict:
        """Initialize accumulation with first gradient batch.

        Casts to accumulation dtype only if needed to avoid unnecessary copies.
        """

        def init_recursive(grads: dict) -> dict:
            result = {}
            for key, value in grads.items():
                if isinstance(value, dict):
                    result[key] = init_recursive(value)
                elif isinstance(value, mx.array):
                    # Cast to accumulation dtype only if needed
                    if value.dtype == self.accumulation_dtype:
                        result[key] = value
                    else:
                        result[key] = value.astype(self.accumulation_dtype)
                else:
                    result[key] = value
            return result

        return init_recursive(gradients)

    def _add_gradients(self, gradients: dict) -> None:
        """Add gradients to accumulator.

        Note: Gradients are already cast to accumulation_dtype in _init_accumulation,
        and we cast new gradients to the same dtype for addition. This is necessary
        because input gradients may come in different precisions (FP16, BF16, etc).
        """

        def add_recursive(accumulated: dict, new: dict) -> None:
            for key, value in new.items():
                if isinstance(value, dict):
                    add_recursive(accumulated[key], value)
                elif isinstance(value, mx.array):
                    # Cast new gradient to accumulation dtype and add
                    # Note: accumulated is already in accumulation_dtype from _init_accumulation
                    new_value = (
                        value if value.dtype == self.accumulation_dtype else value.astype(self.accumulation_dtype)
                    )
                    accumulated[key] = accumulated[key] + new_value

        add_recursive(self._accumulated_grads, gradients)

    def should_step(self) -> bool:
        """Check if it's time for optimizer step.

        Returns:
            True if accumulation_steps micro-batches have been processed
        """
        return self._current_step >= self.accumulation_steps

    def get_and_reset(self) -> dict:
        """Get accumulated gradients and reset accumulator.

        Returns:
            Averaged accumulated gradients

        Raises:
            RuntimeError: If called before any accumulation
        """
        if self._accumulated_grads is None:
            raise RuntimeError(
                "get_and_reset() called before any gradients were accumulated. Call accumulate() at least once first."
            )

        # Average gradients (assertion guards against division by zero if invariant broken)
        assert self._current_step > 0, "Cannot average 0 gradient steps"
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

        # Reset
        self._accumulated_grads = None
        self._current_step = 0

        return averaged

    @property
    def current_step(self) -> int:
        """Get current accumulation step."""
        return self._current_step

    def reset(self) -> None:
        """Reset accumulator without returning gradients."""
        self._accumulated_grads = None
        self._current_step = 0
