"""Runtime quantization utilities for Z-Image models.

Applies quantization to model weights at runtime based on configuration.
Supports per-layer quantization settings for mixed precision strategies.

Usage:
    from mflux.models.z_image.weights.runtime_quantizer import RuntimeQuantizer
    from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

    # Apply quantization to model
    config = QuantizationConfig.from_mode("mixed")
    quantizer = RuntimeQuantizer(config)
    quantizer.quantize_model(model)

    # Or use the convenience function
    quantize_model(model, mode="speed")
"""

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn

from mflux.models.z_image.weights.dynamic_quantization import (
    QuantizationConfig,
    QuantizationMode,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RuntimeQuantizer:
    """Apply quantization to model components at runtime.

    Uses MLX's nn.quantize for efficient on-device quantization.
    Supports mixed precision with per-layer bit widths.

    Attributes:
        config: Quantization configuration
        stats: Dictionary tracking quantization statistics
    """

    def __init__(self, config: QuantizationConfig):
        """Initialize quantizer.

        Args:
            config: Quantization configuration specifying bits per component
        """
        self.config = config
        self.stats = {
            "layers_quantized": 0,
            "layers_skipped": 0,
            "original_size_mb": 0.0,
            "quantized_size_mb": 0.0,
        }

    def quantize_model(
        self,
        model: nn.Module,
        verbose: bool = False,
    ) -> nn.Module:
        """Apply quantization to entire model.

        Quantizes transformer, VAE, and text encoder based on config.

        Args:
            model: Z-Image model with transformer, vae, text_encoder attributes
            verbose: Whether to log quantization progress

        Returns:
            Same model with quantized weights (modified in place)
        """
        if not self.config.is_quantized:
            logger.info("No quantization configured, returning model unchanged")
            return model

        if verbose:
            logger.info(f"Applying quantization: {self.config}")

        # Quantize each component
        if hasattr(model, "transformer") and self.config.transformer_bits:
            self._quantize_component(
                model.transformer,
                "transformer",
                self.config.get_transformer_config().bits,
                verbose,
            )

        if hasattr(model, "vae") and self.config.vae_bits:
            self._quantize_component(
                model.vae,
                "vae",
                self.config.get_vae_config().bits,
                verbose,
            )

        if hasattr(model, "text_encoder") and self.config.text_encoder_bits:
            self._quantize_component(
                model.text_encoder,
                "text_encoder",
                self.config.get_text_encoder_config().bits,
                verbose,
            )

        if verbose:
            self._log_stats()

        return model

    def _quantize_component(
        self,
        component: nn.Module,
        name: str,
        bits: int,
        verbose: bool,
    ) -> None:
        """Quantize a single model component.

        Args:
            component: Model component to quantize
            name: Component name for logging
            bits: Quantization bits
            verbose: Whether to log progress
        """
        if verbose:
            logger.info(f"Quantizing {name} to {bits} bits")

        # Use MLX's built-in quantization
        nn.quantize(
            component,
            bits=bits,
            group_size=self.config.group_size,
        )

        self.stats["layers_quantized"] += self._count_quantized_layers(component)

    def _quantize_mixed(
        self,
        component: nn.Module,
        name: str,
        verbose: bool,
    ) -> None:
        """Apply mixed quantization to component.

        Uses attention_bits for attention layers and ffn_bits for FFN layers.

        Args:
            component: Model component
            name: Component name for logging
            verbose: Whether to log progress
        """
        if verbose:
            logger.info(f"Applying mixed quantization to {name}")

        # Walk through modules and apply per-layer quantization
        for layer_name, module in component.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            bits = self.config.effective_bits_for_layer(layer_name)
            if bits is None:
                self.stats["layers_skipped"] += 1
                continue

            # Quantize this specific layer
            self._quantize_linear_layer(module, bits)
            self.stats["layers_quantized"] += 1

            if verbose:
                logger.debug(f"  {layer_name}: {bits} bits")

    def _quantize_linear_layer(self, layer: nn.Linear, bits: int) -> None:
        """Quantize a single linear layer.

        Args:
            layer: Linear layer to quantize
            bits: Quantization bits
        """
        # Get original weight
        weight = layer.weight

        # Quantize weight using MLX quantization
        # This modifies the layer in place
        quantized_weight = self._quantize_tensor(weight, bits)

        # Replace weight (MLX uses frozen parameters)
        layer.weight = quantized_weight

    def _quantize_tensor(self, tensor: mx.array, bits: int) -> mx.array:
        """Quantize a tensor to specified bits.

        Uses symmetric quantization with per-group scaling.

        Args:
            tensor: Input tensor
            bits: Target bits (2, 4, or 8)

        Returns:
            Quantized tensor
        """
        # This is a simplified quantization for demonstration
        # In practice, MLX's nn.quantize handles this more efficiently

        # Calculate scale based on max absolute value
        max_val = mx.max(mx.abs(tensor))
        n_levels = 2 ** (bits - 1) - 1  # Symmetric quantization

        scale = max_val / n_levels
        scale = mx.maximum(scale, mx.array(1e-8))  # Prevent division by zero

        # Quantize
        quantized = mx.round(tensor / scale)
        quantized = mx.clip(quantized, -n_levels, n_levels)

        # Dequantize (store in float for computation)
        dequantized = quantized * scale

        return dequantized

    def _count_quantized_layers(self, module: nn.Module) -> int:
        """Count quantized linear layers in module.

        Args:
            module: Module to inspect

        Returns:
            Number of linear layers
        """
        # Use modules() instead of named_modules() to avoid string generation overhead
        return sum(1 for m in module.modules() if isinstance(m, nn.Linear))

    def _log_stats(self) -> None:
        """Log quantization statistics."""
        logger.info(
            f"Quantization complete: "
            f"{self.stats['layers_quantized']} layers quantized, "
            f"{self.stats['layers_skipped']} layers skipped"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get quantization statistics.

        Returns:
            Dictionary with quantization statistics
        """
        return self.stats.copy()


def quantize_model(
    model: nn.Module,
    mode: QuantizationMode | str | int | None = None,
    config: QuantizationConfig | None = None,
    verbose: bool = False,
) -> nn.Module:
    """Convenience function to quantize a model.

    Args:
        model: Z-Image model to quantize
        mode: Quantization mode preset (ignored if config provided)
        config: Explicit quantization config (takes precedence over mode)
        verbose: Whether to log progress

    Returns:
        Quantized model (modified in place)

    Examples:
        # Using mode preset
        quantize_model(model, mode="speed")
        quantize_model(model, mode=4)  # INT4

        # Using explicit config
        config = QuantizationConfig(transformer_bits=4, vae_bits=8)
        quantize_model(model, config=config)
    """
    if config is None:
        config = QuantizationConfig.from_mode(mode)

    quantizer = RuntimeQuantizer(config)
    return quantizer.quantize_model(model, verbose=verbose)


def estimate_quantized_size(
    model: nn.Module,
    config: QuantizationConfig,
) -> dict[str, float]:
    """Estimate model size after quantization.

    Args:
        model: Model to estimate
        config: Quantization config

    Returns:
        Dictionary with size estimates in MB
    """
    # Rough estimates based on typical model structure
    # Actual size depends on specific architecture

    def count_params(module: nn.Module) -> int:
        total = 0
        for param in module.parameters():
            total += param.size
        return total

    estimates = {}

    # Full precision size (BF16 = 2 bytes per param)
    total_params = count_params(model)
    full_size_mb = total_params * 2 / (1024 * 1024)
    estimates["full_precision_mb"] = full_size_mb

    # Estimate quantized size
    transformer_params = count_params(model.transformer) if hasattr(model, "transformer") else 0
    vae_params = count_params(model.vae) if hasattr(model, "vae") else 0
    text_encoder_params = count_params(model.text_encoder) if hasattr(model, "text_encoder") else 0

    def bits_to_bytes(bits: int | None) -> float:
        if bits is None:
            return 2.0  # BF16
        return bits / 8.0

    quantized_size_mb = (
        transformer_params * bits_to_bytes(config.transformer_bits)
        + vae_params * bits_to_bytes(config.vae_bits)
        + text_encoder_params * bits_to_bytes(config.text_encoder_bits)
    ) / (1024 * 1024)

    estimates["quantized_mb"] = quantized_size_mb

    # Handle edge case where quantized size is near zero
    if quantized_size_mb < 0.001:
        # Near-zero quantized size means infinite compression (unusual case)
        estimates["compression_ratio"] = float("inf") if full_size_mb > 0 else 1.0
    else:
        estimates["compression_ratio"] = full_size_mb / quantized_size_mb

    estimates["memory_saved_mb"] = full_size_mb - quantized_size_mb

    return estimates
