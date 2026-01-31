"""Activation checkpointing for Z-Image training.

Reduces activation memory 5-10x by trading compute for memory.
Uses MLX's mx.checkpoint() to recompute activations during backward pass.

Memory Impact (6B model, batch_size=1, 1024x1024):
- Without checkpointing: ~15GB activations
- With checkpointing: ~1.5-3GB activations
- Trade-off: ~30-50% slower training

Designed for:
- Very large batch sizes
- High-resolution training (2K+)
- Memory-constrained systems (< 128GB)
- Full fine-tuning with 512GB unified memory
"""

import logging

import mlx.core as mx
from mlx import nn

logger = logging.getLogger(__name__)


class ActivationCheckpointer:
    """Applies activation checkpointing to transformer layers.

    Wraps transformer layers with mx.checkpoint() to trade compute
    for memory during training. Activations are recomputed during
    the backward pass instead of being stored.

    Usage:
        # Apply checkpointing to every 2nd layer
        ActivationCheckpointer.apply_to_model(
            model.transformer,
            checkpoint_every_n_layers=2,
        )

    Note:
        - Only affects training (backward pass)
        - Forward pass unchanged
        - Adds ~30-50% training time overhead
    """

    # Memory reduction factors for estimation
    CHECKPOINTING_MEMORY_FACTOR = 0.15  # ~85% reduction
    COMPUTE_OVERHEAD_FACTOR = 1.4  # ~40% slower

    @staticmethod
    def apply_to_model(
        transformer: nn.Module,
        checkpoint_every_n_layers: int = 2,
        layer_names: list[str] | None = None,
    ) -> int:
        """Apply checkpointing to transformer layers.

        Args:
            transformer: Transformer module with 'layers' attribute
            checkpoint_every_n_layers: Apply checkpoint every N layers (1 = all)
            layer_names: Specific layer attribute names to checkpoint
                        Default: ["layers", "noise_refiner", "context_refiner"]

        Returns:
            Number of layers wrapped with checkpointing.
        """
        if layer_names is None:
            layer_names = ["layers", "noise_refiner", "context_refiner"]

        wrapped_count = 0

        for layer_name in layer_names:
            if not hasattr(transformer, layer_name):
                continue

            layers = getattr(transformer, layer_name)
            if not isinstance(layers, list):
                continue

            for i, layer in enumerate(layers):
                if i % checkpoint_every_n_layers == 0:
                    wrapped_layer = ActivationCheckpointer._wrap_layer(layer)
                    layers[i] = wrapped_layer
                    wrapped_count += 1

        if wrapped_count > 0:
            logger.info(
                f"Applied activation checkpointing to {wrapped_count} layers (every {checkpoint_every_n_layers} layers)"
            )

        return wrapped_count

    @staticmethod
    def _wrap_layer(layer: nn.Module) -> nn.Module:
        """Wrap a single layer with mx.checkpoint().

        Creates a wrapper that uses mx.checkpoint() during the forward
        pass, causing MLX to recompute activations during backward.

        Args:
            layer: The layer module to wrap

        Returns:
            Wrapped layer module
        """
        original_call = layer.__call__

        def checkpointed_forward(*args, **kwargs):
            # mx.checkpoint() tells MLX to not store intermediate activations
            # and instead recompute them during the backward pass
            return mx.checkpoint(lambda: original_call(*args, **kwargs))()

        # Create a wrapper module that preserves the original layer
        layer._original_call = original_call
        layer.__call__ = checkpointed_forward
        layer._checkpointed = True

        return layer

    @staticmethod
    def remove_from_model(transformer: nn.Module) -> int:
        """Remove checkpointing from all layers.

        Restores original layer behavior for inference or debugging.

        Args:
            transformer: Transformer with checkpointed layers

        Returns:
            Number of layers unwrapped.
        """
        layer_names = ["layers", "noise_refiner", "context_refiner"]
        unwrapped_count = 0

        for layer_name in layer_names:
            if not hasattr(transformer, layer_name):
                continue

            layers = getattr(transformer, layer_name)
            if not isinstance(layers, list):
                continue

            for layer in layers:
                if getattr(layer, "_checkpointed", False):
                    layer.__call__ = layer._original_call
                    del layer._original_call
                    del layer._checkpointed
                    unwrapped_count += 1

        if unwrapped_count > 0:
            logger.info(f"Removed activation checkpointing from {unwrapped_count} layers")

        return unwrapped_count

    @staticmethod
    def estimate_memory_savings(
        num_layers: int,
        checkpoint_every_n: int,
        batch_size: int = 1,
        base_activation_gb: float = 15.0,
    ) -> dict[str, float]:
        """Estimate memory savings from checkpointing.

        Args:
            num_layers: Total number of layers
            checkpoint_every_n: Checkpoint frequency
            batch_size: Training batch size
            base_activation_gb: Base activation memory without checkpointing

        Returns:
            Dictionary with memory estimates
        """
        checkpointed_layers = num_layers // checkpoint_every_n
        checkpoint_ratio = checkpointed_layers / max(1, num_layers)

        # Simplified model: checkpointed layers save memory
        memory_reduction = checkpoint_ratio * (1 - ActivationCheckpointer.CHECKPOINTING_MEMORY_FACTOR)
        new_activation_gb = base_activation_gb * batch_size * (1 - memory_reduction)

        return {
            "original_activation_gb": base_activation_gb * batch_size,
            "checkpointed_activation_gb": new_activation_gb,
            "memory_saved_gb": (base_activation_gb * batch_size) - new_activation_gb,
            "reduction_factor": memory_reduction,
            "checkpointed_layers": checkpointed_layers,
            "estimated_overhead": ActivationCheckpointer.COMPUTE_OVERHEAD_FACTOR,
        }


def apply_gradient_checkpointing(
    model: nn.Module,
    enabled: bool = True,
    checkpoint_every_n_layers: int = 2,
) -> bool:
    """Convenience function to apply or remove gradient checkpointing.

    Args:
        model: Model with transformer attribute
        enabled: Whether to enable checkpointing
        checkpoint_every_n_layers: Apply every N layers

    Returns:
        True if checkpointing was applied, False otherwise
    """
    if not hasattr(model, "transformer"):
        logger.warning("Model has no transformer attribute, skipping checkpointing")
        return False

    if enabled:
        count = ActivationCheckpointer.apply_to_model(
            model.transformer,
            checkpoint_every_n_layers=checkpoint_every_n_layers,
        )
        return count > 0
    else:
        ActivationCheckpointer.remove_from_model(model.transformer)
        return False
