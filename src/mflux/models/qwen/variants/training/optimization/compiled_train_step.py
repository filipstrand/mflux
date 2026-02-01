"""
Compiled Training Step for MLX.

Wraps training step with mx.compile() for JIT compilation and Metal
kernel fusion. Provides 10-30% speedup over eager execution.

Usage:
    train_step = create_compiled_train_step(model, config, loss_fn)

    for batch in dataloader:
        loss, grads = train_step(batch)
        optimizer.update(model, grads)

Note: First call has compilation overhead (~5-10s warmup).
Subsequent calls are significantly faster.
"""

from typing import Any, Callable

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.qwen.variants.training.optimization.qwen_loss import (
    QwenLoss,
    QwenLossWithRegularization,
    QwenTrainingBatch,
)


def create_compiled_train_step(
    model: Any,
    config: Config,
    use_regularization: bool = False,
    lora_weight_decay: float = 0.0,
) -> Callable[[QwenTrainingBatch], tuple[mx.array, dict[str, Any]]]:
    """
    Create a JIT-compiled training step function.

    Wraps the loss computation and gradient calculation with mx.compile()
    for Metal kernel fusion and optimized execution.

    Args:
        model: QwenImage model
        config: Training configuration
        use_regularization: Whether to use loss with regularization
        lora_weight_decay: L2 regularization on LoRA weights (only if use_regularization=True)

    Returns:
        Compiled training step function that returns (loss, gradients)

    Example:
        train_step = create_compiled_train_step(qwen, config)

        for batch in dataloader:
            loss, grads = train_step(batch)
            mx.eval(loss)  # Force evaluation
            optimizer.update(model, grads)
    """
    # Choose loss function
    if use_regularization:

        def loss_fn(batch: QwenTrainingBatch) -> mx.array:
            return QwenLossWithRegularization.compute_loss_with_regularization(
                qwen=model,
                config=config,
                batch=batch,
                lora_weight_decay=lora_weight_decay,
            )
    else:

        def loss_fn(batch: QwenTrainingBatch) -> mx.array:
            return QwenLoss.compute_loss(model, config, batch)

    # Create value_and_grad function
    grad_fn = nn.value_and_grad(model=model, fn=loss_fn)

    # Wrap with mx.compile() for JIT compilation
    # This fuses Metal kernels for significant speedup
    compiled_fn = mx.compile(grad_fn)

    return compiled_fn


def create_train_step(
    model: Any,
    config: Config,
    compile: bool = True,
    use_regularization: bool = False,
    lora_weight_decay: float = 0.0,
) -> Callable[[QwenTrainingBatch], tuple[mx.array, dict[str, Any]]]:
    """
    Create a training step function with optional compilation.

    Factory function that creates either compiled or eager training step.
    Use compile=False for debugging (easier to trace errors).

    Args:
        model: QwenImage model
        config: Training configuration
        compile: Whether to use mx.compile() (default: True)
        use_regularization: Whether to use loss with regularization
        lora_weight_decay: L2 regularization on LoRA weights

    Returns:
        Training step function that returns (loss, gradients)

    Example:
        # Production: compiled for speed
        train_step = create_train_step(qwen, config, compile=True)

        # Debugging: eager for better error messages
        train_step = create_train_step(qwen, config, compile=False)
    """
    if use_regularization:

        def loss_fn(batch: QwenTrainingBatch) -> mx.array:
            return QwenLossWithRegularization.compute_loss_with_regularization(
                qwen=model,
                config=config,
                batch=batch,
                lora_weight_decay=lora_weight_decay,
            )
    else:

        def loss_fn(batch: QwenTrainingBatch) -> mx.array:
            return QwenLoss.compute_loss(model, config, batch)

    # Create value_and_grad function
    grad_fn = nn.value_and_grad(model=model, fn=loss_fn)

    if compile:
        return mx.compile(grad_fn)
    else:
        return grad_fn


class CompiledTrainStep:
    """
    Compiled training step with warmup and diagnostics.

    Provides additional features over the simple function:
    - Explicit warmup phase
    - Timing diagnostics
    - Compilation status tracking
    """

    def __init__(
        self,
        model: Any,
        config: Config,
        use_regularization: bool = False,
        lora_weight_decay: float = 0.0,
    ):
        """
        Initialize compiled training step.

        Args:
            model: QwenImage model
            config: Training configuration
            use_regularization: Whether to use loss with regularization
            lora_weight_decay: L2 regularization on LoRA weights
        """
        self.model = model
        self.config = config
        self._compiled_fn = create_compiled_train_step(
            model=model,
            config=config,
            use_regularization=use_regularization,
            lora_weight_decay=lora_weight_decay,
        )
        self._is_warmed_up = False
        self._warmup_time: float | None = None

    @property
    def is_warmed_up(self) -> bool:
        """Whether the compiled function has been warmed up."""
        return self._is_warmed_up

    @property
    def warmup_time(self) -> float | None:
        """Time taken for warmup compilation (seconds)."""
        return self._warmup_time

    def warmup(self, batch: QwenTrainingBatch) -> None:
        """
        Explicitly warm up the compiled function.

        Call this once before training to pay compilation cost upfront.
        Subsequent calls will be much faster.

        Args:
            batch: Sample batch for warmup (can be any valid batch)
        """
        import time

        if self._is_warmed_up:
            return

        start = time.perf_counter()

        # Run once to trigger compilation
        loss, grads = self._compiled_fn(batch)
        mx.eval(loss)  # Force evaluation to complete compilation

        self._warmup_time = time.perf_counter() - start
        self._is_warmed_up = True

    def __call__(self, batch: QwenTrainingBatch) -> tuple[mx.array, dict[str, Any]]:
        """
        Execute compiled training step.

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss, gradients)
        """
        if not self._is_warmed_up:
            # Auto-warmup on first call
            self.warmup(batch)

        return self._compiled_fn(batch)
