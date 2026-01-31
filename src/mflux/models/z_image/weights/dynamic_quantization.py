"""Dynamic quantization configuration for Z-Image models.

Provides runtime quantization presets and configuration classes
for flexible model compression strategies.

Usage:
    from mflux.models.z_image.weights.dynamic_quantization import (
        QuantizationMode,
        QuantizationConfig,
    )

    # Use predefined preset
    config = QuantizationConfig.from_mode(QuantizationMode.SPEED)

    # Or custom configuration
    config = QuantizationConfig(
        transformer_bits=4,
        vae_bits=8,
        text_encoder_bits=8,
    )

Presets:
    - NONE: No quantization (full precision BF16)
    - INT2: Extreme compression (smallest, lowest quality)
    - INT4: Balanced (good compression, acceptable quality)
    - INT8: Minimal compression (near-lossless quality)
    - MIXED: Component-specific (INT8 attention + INT4 FFN)
    - SPEED: Optimized for inference speed
    - QUALITY: Optimized for output quality
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Valid quantization bit widths supported by MLX
VALID_QUANTIZATION_BITS: tuple[int | None, ...] = (None, 2, 4, 8)

# Maximum recommended group size for quantization accuracy
# Larger values reduce quantization overhead but decrease accuracy
# Typical values (32-128) balance accuracy with compression ratio
MAX_RECOMMENDED_GROUP_SIZE = 256


class QuantizationMode(Enum):
    """Quantization mode presets.

    These modes provide pre-configured quantization strategies
    for different use cases.
    """

    NONE = "none"  # No quantization - full BF16 precision
    INT2 = "int2"  # Extreme compression - 2-bit weights
    INT4 = "int4"  # Balanced - 4-bit weights
    INT8 = "int8"  # Minimal compression - 8-bit weights
    MIXED = "mixed"  # Component-specific: attention=INT8, FFN=INT4
    SPEED = "speed"  # Optimize for inference speed (INT4 transformer)
    QUALITY = "quality"  # Optimize for output quality (INT8 everywhere)

    @classmethod
    def from_string(cls, value: str | int | None) -> "QuantizationMode | None":
        """Convert string or int to QuantizationMode.

        Args:
            value: Mode string, bits integer, or None

        Returns:
            QuantizationMode or None if no quantization

        Examples:
            QuantizationMode.from_string("speed") -> QuantizationMode.SPEED
            QuantizationMode.from_string(4) -> QuantizationMode.INT4
            QuantizationMode.from_string(None) -> None
        """
        if value is None:
            return None

        if isinstance(value, int):
            mapping = {2: cls.INT2, 4: cls.INT4, 8: cls.INT8}
            return mapping.get(value)

        if isinstance(value, str):
            value_lower = value.lower()
            for mode in cls:
                if mode.value == value_lower:
                    return mode

        return None


@dataclass
class ComponentQuantization:
    """Quantization settings for a single model component.

    Attributes:
        bits: Quantization bits (2, 4, 8) or None for no quantization
        group_size: Group size for weight grouping (64 is typical)
        exclude_layers: Layer names to exclude from quantization
    """

    bits: int | None = None
    group_size: int = 64
    exclude_layers: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.bits is not None and self.bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, 8, or None. Got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1. Got {self.group_size}")


@dataclass
class QuantizationConfig:
    """Complete quantization configuration for Z-Image model.

    Allows per-component quantization settings for fine-grained control.

    Attributes:
        transformer_bits: Bits for transformer (2, 4, 8, or None)
        vae_bits: Bits for VAE (typically 8 or None for quality)
        text_encoder_bits: Bits for text encoder
        attention_bits: Override for attention layers (None = use transformer_bits)
        ffn_bits: Override for FFN layers (None = use transformer_bits)
        group_size: Weight group size for quantization
        exclude_layers: Global list of layer patterns to exclude
    """

    transformer_bits: int | None = None
    vae_bits: int | None = None
    text_encoder_bits: int | None = None
    attention_bits: int | None = None  # Override for attention specifically
    ffn_bits: int | None = None  # Override for FFN specifically
    group_size: int = 64
    exclude_layers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate bits values using module constant
        for bit_attr in ("transformer_bits", "vae_bits", "text_encoder_bits", "attention_bits", "ffn_bits"):
            value = getattr(self, bit_attr)
            if value not in VALID_QUANTIZATION_BITS:
                raise ValueError(f"{bit_attr} must be 2, 4, 8, or None. Got {value}")

        # Validate group_size
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1. Got {self.group_size}")
        if self.group_size > MAX_RECOMMENDED_GROUP_SIZE:
            import warnings

            warnings.warn(
                f"Large group_size ({self.group_size}) may reduce quantization accuracy. Typical values are 32-128.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_mode(cls, mode: QuantizationMode | str | int | None) -> "QuantizationConfig":
        """Create config from a quantization mode preset.

        Args:
            mode: QuantizationMode, mode string, bits integer, or None

        Returns:
            QuantizationConfig with appropriate settings

        Examples:
            QuantizationConfig.from_mode(QuantizationMode.SPEED)
            QuantizationConfig.from_mode("quality")
            QuantizationConfig.from_mode(4)  # Equivalent to INT4
        """
        # Convert to enum if needed
        if not isinstance(mode, QuantizationMode):
            mode = QuantizationMode.from_string(mode)

        if mode is None or mode == QuantizationMode.NONE:
            return cls()  # No quantization

        if mode == QuantizationMode.INT2:
            return cls(
                transformer_bits=2,
                vae_bits=8,  # Keep VAE higher for quality
                text_encoder_bits=4,
            )

        if mode == QuantizationMode.INT4:
            return cls(
                transformer_bits=4,
                vae_bits=8,
                text_encoder_bits=4,
            )

        if mode == QuantizationMode.INT8:
            return cls(
                transformer_bits=8,
                vae_bits=8,
                text_encoder_bits=8,
            )

        if mode == QuantizationMode.MIXED:
            # INT8 for attention (quality-sensitive), INT4 for FFN (compression-tolerant)
            return cls(
                transformer_bits=4,  # Default
                attention_bits=8,  # Override for attention
                ffn_bits=4,  # Override for FFN
                vae_bits=8,
                text_encoder_bits=8,
            )

        if mode == QuantizationMode.SPEED:
            # Prioritize inference speed with aggressive compression
            return cls(
                transformer_bits=4,
                vae_bits=4,
                text_encoder_bits=4,
                group_size=128,  # Larger groups for faster quantization
            )

        if mode == QuantizationMode.QUALITY:
            # Prioritize output quality with minimal compression
            return cls(
                transformer_bits=8,
                vae_bits=None,  # Keep VAE at full precision
                text_encoder_bits=8,
                group_size=32,  # Smaller groups for better accuracy
            )

        # Fallback
        return cls()

    @classmethod
    def from_bits(cls, bits: int | None) -> "QuantizationConfig":
        """Create uniform config with same bits for all components.

        Args:
            bits: Quantization bits for all components

        Returns:
            QuantizationConfig with uniform settings
        """
        return cls(
            transformer_bits=bits,
            vae_bits=bits,
            text_encoder_bits=bits,
        )

    def get_transformer_config(self) -> ComponentQuantization:
        """Get quantization config for transformer component."""
        return ComponentQuantization(
            bits=self.transformer_bits,
            group_size=self.group_size,
            exclude_layers=self.exclude_layers,
        )

    def get_vae_config(self) -> ComponentQuantization:
        """Get quantization config for VAE component."""
        return ComponentQuantization(
            bits=self.vae_bits,
            group_size=self.group_size,
            exclude_layers=self.exclude_layers,
        )

    def get_text_encoder_config(self) -> ComponentQuantization:
        """Get quantization config for text encoder component."""
        return ComponentQuantization(
            bits=self.text_encoder_bits,
            group_size=self.group_size,
            exclude_layers=self.exclude_layers,
        )

    def effective_bits_for_layer(self, layer_name: str) -> int | None:
        """Get effective quantization bits for a specific layer.

        Handles mixed quantization by checking layer type.

        Args:
            layer_name: Full layer name/path

        Returns:
            Bits to use, or None for no quantization
        """
        # Check exclusions first
        for pattern in self.exclude_layers:
            if pattern in layer_name:
                return None

        # Cache lowercased name for pattern matching (avoid repeated lower() calls)
        layer_name_lower = layer_name.lower()

        # Check for attention override
        if self.attention_bits is not None:
            attention_patterns = ["attention", "attn", "self_attn", "cross_attn"]
            if any(p in layer_name_lower for p in attention_patterns):
                return self.attention_bits

        # Check for FFN override
        if self.ffn_bits is not None:
            ffn_patterns = ["ffn", "mlp", "feed_forward", "dense"]
            if any(p in layer_name_lower for p in ffn_patterns):
                return self.ffn_bits

        # Default to transformer bits
        return self.transformer_bits

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "transformer_bits": self.transformer_bits,
            "vae_bits": self.vae_bits,
            "text_encoder_bits": self.text_encoder_bits,
            "attention_bits": self.attention_bits,
            "ffn_bits": self.ffn_bits,
            "group_size": self.group_size,
            "exclude_layers": self.exclude_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationConfig":
        """Deserialize config from dictionary."""
        return cls(
            transformer_bits=data.get("transformer_bits"),
            vae_bits=data.get("vae_bits"),
            text_encoder_bits=data.get("text_encoder_bits"),
            attention_bits=data.get("attention_bits"),
            ffn_bits=data.get("ffn_bits"),
            group_size=data.get("group_size", 64),
            exclude_layers=data.get("exclude_layers", []),
        )

    @property
    def is_quantized(self) -> bool:
        """Check if any component is quantized."""
        return any(
            [
                self.transformer_bits is not None,
                self.vae_bits is not None,
                self.text_encoder_bits is not None,
            ]
        )

    @property
    def min_bits(self) -> int | None:
        """Get minimum quantization bits across all components."""
        bits_list = [b for b in [self.transformer_bits, self.vae_bits, self.text_encoder_bits] if b is not None]
        return min(bits_list) if bits_list else None

    def __str__(self) -> str:
        """Human-readable representation."""
        if not self.is_quantized:
            return "QuantizationConfig(none)"

        parts = []
        if self.transformer_bits:
            parts.append(f"transformer={self.transformer_bits}b")
        if self.vae_bits:
            parts.append(f"vae={self.vae_bits}b")
        if self.text_encoder_bits:
            parts.append(f"text_encoder={self.text_encoder_bits}b")

        return f"QuantizationConfig({', '.join(parts)})"
