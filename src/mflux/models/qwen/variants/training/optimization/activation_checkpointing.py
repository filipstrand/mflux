"""Activation checkpointing for Qwen-Image training.

Reduces activation memory 5-10x by trading compute for memory.
Uses MLX's mx.checkpoint() to recompute activations during backward pass.

Memory Impact (Qwen 6B model, batch_size=1, 1024x1024):
- Without checkpointing: ~15GB activations
- With checkpointing: ~1.5-3GB activations
- Trade-off: ~30-50% slower training

Qwen Transformer Architecture:
- 60 DiT blocks (transformer_blocks)
- Each block has attention + FFN
- Checkpointing every 2-3 blocks provides good memory/speed balance

Designed for:
- Very large batch sizes on Mac Studio M3 Ultra (512GB)
- High-resolution training (2K+)
- Full fine-tuning scenarios
- LoRA training with limited memory
"""

import logging

import mlx.core as mx
from mlx import nn

logger = logging.getLogger(__name__)


class QwenActivationCheckpointer:
    """Applies activation checkpointing to Qwen transformer layers.

    Wraps transformer layers with mx.checkpoint() to trade compute
    for memory during training. Activations are recomputed during
    the backward pass instead of being stored.

    Qwen-specific configuration:
    - Default layer_names targets "transformer_blocks" (60 blocks)
    - Recommended checkpoint_every_n_layers=2-3 for good balance

    Usage:
        # Apply checkpointing to every 2nd block
        QwenActivationCheckpointer.apply_to_model(
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

    # Qwen-specific layer names
    QWEN_LAYER_NAMES = ["transformer_blocks", "blocks", "layers"]

    @staticmethod
    def apply_to_model(
        transformer: nn.Module,
        checkpoint_every_n_layers: int = 2,
        layer_names: list[str] | None = None,
    ) -> int:
        """Apply checkpointing to Qwen transformer layers.

        Args:
            transformer: QwenTransformer module
            checkpoint_every_n_layers: Apply checkpoint every N layers (1 = all)
                - 1: Maximum memory savings, ~50% slower
                - 2: Good balance (default)
                - 3: Moderate savings, ~30% slower
            layer_names: Specific layer attribute names to checkpoint
                        Default: ["transformer_blocks", "blocks", "layers"]

        Returns:
            Number of layers wrapped with checkpointing.

        Raises:
            ValueError: If checkpoint_every_n_layers is less than 1
        """
        if checkpoint_every_n_layers < 1:
            raise ValueError(f"checkpoint_every_n_layers must be >= 1, got {checkpoint_every_n_layers}")

        if layer_names is None:
            layer_names = QwenActivationCheckpointer.QWEN_LAYER_NAMES

        wrapped_count = 0

        for layer_name in layer_names:
            if not hasattr(transformer, layer_name):
                continue

            layers = getattr(transformer, layer_name)
            if not isinstance(layers, list):
                continue

            for i, layer in enumerate(layers):
                if i % checkpoint_every_n_layers == 0:
                    wrapped_layer = QwenActivationCheckpointer._wrap_layer(layer)
                    layers[i] = wrapped_layer
                    wrapped_count += 1

        if wrapped_count > 0:
            logger.info(
                f"Applied activation checkpointing to {wrapped_count} Qwen transformer blocks "
                f"(every {checkpoint_every_n_layers} layers)"
            )
        else:
            import warnings

            warnings.warn(
                f"No layers were checkpointed. Verify layer_names {layer_names} match transformer structure.",
                UserWarning,
                stacklevel=2,
            )

        return wrapped_count

    @staticmethod
    def _wrap_layer(layer: nn.Module) -> nn.Module:
        """Wrap a single layer with mx.checkpoint().

        Creates a wrapper that uses mx.checkpoint() during the forward
        pass, causing MLX to recompute activations during backward.

        Args:
            layer: The transformer block to wrap

        Returns:
            Wrapped layer module
        """
        original_call = layer.__call__

        def checkpointed_forward(*args, **kwargs):
            # mx.checkpoint() tells MLX to not store intermediate activations
            # and instead recompute them during the backward pass
            return mx.checkpoint(lambda: original_call(*args, **kwargs))()

        # Preserve the original layer for unwrapping later
        layer._original_call = original_call
        layer.__call__ = checkpointed_forward
        layer._checkpointed = True

        return layer

    @staticmethod
    def remove_from_model(transformer: nn.Module) -> int:
        """Remove checkpointing from all Qwen transformer layers.

        Restores original layer behavior for inference or debugging.

        Args:
            transformer: QwenTransformer with checkpointed layers

        Returns:
            Number of layers unwrapped.
        """
        layer_names = QwenActivationCheckpointer.QWEN_LAYER_NAMES
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
            logger.info(f"Removed activation checkpointing from {unwrapped_count} Qwen layers")

        return unwrapped_count

    @staticmethod
    def estimate_memory_savings(
        num_layers: int = 60,  # Qwen default
        checkpoint_every_n: int = 2,
        batch_size: int = 1,
        base_activation_gb: float = 15.0,
    ) -> dict[str, float]:
        """Estimate memory savings from checkpointing for Qwen.

        Args:
            num_layers: Total number of transformer blocks (default 60 for Qwen)
            checkpoint_every_n: Checkpoint frequency
            batch_size: Training batch size
            base_activation_gb: Base activation memory without checkpointing

        Returns:
            Dictionary with memory estimates
        """
        checkpointed_layers = num_layers // checkpoint_every_n
        checkpoint_ratio = checkpointed_layers / max(1, num_layers)

        # Simplified model: checkpointed layers save memory
        memory_reduction = checkpoint_ratio * (1 - QwenActivationCheckpointer.CHECKPOINTING_MEMORY_FACTOR)
        new_activation_gb = base_activation_gb * batch_size * (1 - memory_reduction)

        return {
            "original_activation_gb": base_activation_gb * batch_size,
            "checkpointed_activation_gb": round(new_activation_gb, 2),
            "memory_saved_gb": round((base_activation_gb * batch_size) - new_activation_gb, 2),
            "reduction_percent": round(memory_reduction * 100, 1),
            "checkpointed_layers": checkpointed_layers,
            "total_layers": num_layers,
            "estimated_slowdown_percent": round((QwenActivationCheckpointer.COMPUTE_OVERHEAD_FACTOR - 1) * 100, 1),
        }


def apply_qwen_gradient_checkpointing(
    model: nn.Module,
    enabled: bool = True,
    checkpoint_every_n_layers: int = 2,
) -> bool:
    """Convenience function to apply or remove gradient checkpointing for Qwen.

    Args:
        model: Qwen model with transformer attribute
        enabled: Whether to enable checkpointing
        checkpoint_every_n_layers: Apply every N layers

    Returns:
        True if checkpointing was applied, False otherwise
    """
    if not hasattr(model, "transformer"):
        logger.warning("Model has no transformer attribute, skipping checkpointing")
        return False

    if enabled:
        count = QwenActivationCheckpointer.apply_to_model(
            model.transformer,
            checkpoint_every_n_layers=checkpoint_every_n_layers,
        )
        return count > 0
    else:
        QwenActivationCheckpointer.remove_from_model(model.transformer)
        return False


class SelectiveCheckpointer:
    """Advanced checkpointing with layer-specific strategies.

    For fine-grained control over which layers get checkpointed.
    Useful when you know certain layers are memory-heavy (e.g., attention)
    while others are lightweight (e.g., norms).

    Usage:
        # Checkpoint only attention layers
        SelectiveCheckpointer.checkpoint_by_pattern(
            model.transformer,
            patterns=["attention", "attn"],
        )
    """

    @staticmethod
    def checkpoint_by_pattern(
        transformer: nn.Module,
        patterns: list[str],
    ) -> int:
        """Apply checkpointing to layers matching name patterns.

        Args:
            transformer: Transformer module
            patterns: List of substrings to match in layer names

        Returns:
            Number of layers checkpointed
        """
        wrapped_count = 0
        # Pre-lowercase patterns to avoid repeated lowercasing during recursion
        patterns_lower = [p.lower() for p in patterns]

        def checkpoint_recursive(module: nn.Module, name: str = "") -> None:
            nonlocal wrapped_count

            # Check if this module's name matches any pattern
            name_lower = name.lower()
            should_checkpoint = any(p in name_lower for p in patterns_lower)

            if should_checkpoint and callable(getattr(module, "__call__", None)):
                if not getattr(module, "_checkpointed", False):
                    # Use the same wrapping pattern as _wrap_layer for consistency
                    QwenActivationCheckpointer._wrap_layer(module)
                    wrapped_count += 1

            # Recursively process children
            if hasattr(module, "__dict__"):
                for child_name, child in module.__dict__.items():
                    if isinstance(child, nn.Module):
                        checkpoint_recursive(child, f"{name}.{child_name}" if name else child_name)
                    elif isinstance(child, list):
                        for i, item in enumerate(child):
                            if isinstance(item, nn.Module):
                                checkpoint_recursive(item, f"{name}.{child_name}[{i}]")

        checkpoint_recursive(transformer)

        if wrapped_count > 0:
            logger.info(f"Applied selective checkpointing to {wrapped_count} modules matching {patterns}")

        return wrapped_count

    @staticmethod
    def checkpoint_memory_heavy_layers(transformer: nn.Module) -> int:
        """Checkpoint layers known to be memory-heavy in Qwen.

        Targets:
        - Attention layers (store full attention maps)
        - Cross-attention layers
        - MLP/FFN layers with large intermediate sizes
        """
        return SelectiveCheckpointer.checkpoint_by_pattern(
            transformer,
            patterns=["attention", "attn", "mlp", "feed_forward"],
        )
