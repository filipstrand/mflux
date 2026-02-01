"""Dynamic quantization configuration for Qwen image models.

Provides runtime quantization presets and configuration classes
for flexible model compression strategies.

Usage:
    from mflux.models.qwen.weights.qwen_quantization import (
        QwenQuantizationMode,
        QwenQuantizationConfig,
    )

    # Use predefined preset
    config = QwenQuantizationConfig.from_mode(QwenQuantizationMode.SPEED)

    # Or custom configuration
    config = QwenQuantizationConfig(
        transformer_bits=4,
        vae_bits=8,
        text_encoder_bits=8,
    )

Memory Impact (qwen-image-2512):
    - NONE (BF16): ~35 GB total
    - CONSERVATIVE: ~18 GB (text encoder preserved, transformer INT4)
    - MIXED: ~12 GB (INT8 attention + INT4 FFN)
    - SPEED: ~9 GB (aggressive INT4 everywhere)

Presets:
    - NONE: No quantization (full precision BF16)
    - INT2: Extreme compression (smallest, lowest quality)
    - INT4: Balanced (good compression, acceptable quality)
    - INT8: Minimal compression (near-lossless quality)
    - MIXED: Component-specific (INT8 attention + INT4 FFN)
    - SPEED: Optimized for inference speed
    - QUALITY: Optimized for output quality
    - CONSERVATIVE: Preserve text encoder quality
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Valid quantization bit widths supported by MLX
VALID_QUANTIZATION_BITS: tuple[int | None, ...] = (None, 2, 4, 8)

# Maximum recommended group size for quantization accuracy
MAX_RECOMMENDED_GROUP_SIZE = 256

# Hard maximum to prevent integer overflow in downstream calculations
MAX_GROUP_SIZE = 8192


class QwenQuantizationMode(Enum):
    """Quantization mode presets for Qwen image models.

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
    CONSERVATIVE = "conservative"  # Preserve text encoder, INT4 transformer

    @classmethod
    def from_string(cls, value: str | int | None, strict: bool = True) -> "QwenQuantizationMode | None":
        """Convert string or int to QwenQuantizationMode.

        Args:
            value: Mode string, bits integer, or None
            strict: If True, raise ValueError for invalid strings. If False, return None.

        Returns:
            QwenQuantizationMode or None if no quantization

        Raises:
            ValueError: If strict=True and value is an unrecognized string or int

        Examples:
            QwenQuantizationMode.from_string("speed") -> QwenQuantizationMode.SPEED
            QwenQuantizationMode.from_string(4) -> QwenQuantizationMode.INT4
            QwenQuantizationMode.from_string(None) -> None
        """
        if value is None:
            return None

        if isinstance(value, int):
            mapping = {2: cls.INT2, 4: cls.INT4, 8: cls.INT8}
            result = mapping.get(value)
            if result is None and strict:
                raise ValueError(f"Invalid quantization bits: {value}. Valid values: 2, 4, 8")
            return result

        if isinstance(value, str):
            value_lower = value.lower()
            for mode in cls:
                if mode.value == value_lower:
                    return mode
            if strict:
                valid_modes = ", ".join(m.value for m in cls)
                raise ValueError(f"Invalid quantization mode: '{value}'. Valid modes: {valid_modes}")

        return None


@dataclass
class QwenComponentQuantization:
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
        if self.group_size > MAX_GROUP_SIZE:
            raise ValueError(f"group_size must be <= {MAX_GROUP_SIZE}. Got {self.group_size}")


@dataclass
class QwenQuantizationConfig:
    """Complete quantization configuration for Qwen image model.

    Allows per-component quantization settings for fine-grained control.

    Attributes:
        transformer_bits: Bits for transformer (2, 4, 8, or None)
        vae_bits: Bits for VAE (typically 8 or None for quality)
        text_encoder_bits: Bits for text encoder (None recommended to preserve semantics)
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
        for bit_attr in ("transformer_bits", "vae_bits", "text_encoder_bits", "attention_bits", "ffn_bits"):
            value = getattr(self, bit_attr)
            if value not in VALID_QUANTIZATION_BITS:
                raise ValueError(f"{bit_attr} must be 2, 4, 8, or None. Got {value}")

        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1. Got {self.group_size}")
        if self.group_size > MAX_GROUP_SIZE:
            raise ValueError(f"group_size must be <= {MAX_GROUP_SIZE}. Got {self.group_size}")
        if self.group_size > MAX_RECOMMENDED_GROUP_SIZE:
            import warnings

            warnings.warn(
                f"Large group_size ({self.group_size}) may reduce quantization accuracy. Typical values are 32-128.",
                UserWarning,
                stacklevel=3,  # Account for __post_init__ -> dataclass machinery -> caller
            )

    @classmethod
    def from_mode(cls, mode: QwenQuantizationMode | str | int | None) -> "QwenQuantizationConfig":
        """Create config from a quantization mode preset.

        Args:
            mode: QwenQuantizationMode, mode string, bits integer, or None

        Returns:
            QwenQuantizationConfig with appropriate settings

        Examples:
            QwenQuantizationConfig.from_mode(QwenQuantizationMode.SPEED)
            QwenQuantizationConfig.from_mode("quality")
            QwenQuantizationConfig.from_mode(4)  # Equivalent to INT4
        """
        # Convert to enum if needed
        if not isinstance(mode, QwenQuantizationMode):
            mode = QwenQuantizationMode.from_string(mode)

        if mode is None or mode == QwenQuantizationMode.NONE:
            return cls()  # No quantization

        if mode == QwenQuantizationMode.INT2:
            return cls(
                transformer_bits=2,
                vae_bits=8,  # Keep VAE higher for quality
                text_encoder_bits=4,
            )

        if mode == QwenQuantizationMode.INT4:
            return cls(
                transformer_bits=4,
                vae_bits=8,
                text_encoder_bits=4,
            )

        if mode == QwenQuantizationMode.INT8:
            return cls(
                transformer_bits=8,
                vae_bits=8,
                text_encoder_bits=8,
            )

        if mode == QwenQuantizationMode.MIXED:
            # INT8 for attention (quality-sensitive), INT4 for FFN (compression-tolerant)
            return cls(
                transformer_bits=4,  # Default
                attention_bits=8,  # Override for attention
                ffn_bits=4,  # Override for FFN
                vae_bits=8,
                text_encoder_bits=8,
            )

        if mode == QwenQuantizationMode.SPEED:
            # Prioritize inference speed with aggressive compression
            return cls(
                transformer_bits=4,
                vae_bits=4,
                text_encoder_bits=4,
                group_size=128,  # Larger groups for faster quantization
            )

        if mode == QwenQuantizationMode.QUALITY:
            # Prioritize output quality with minimal compression
            return cls(
                transformer_bits=8,
                vae_bits=None,  # Keep VAE at full precision
                text_encoder_bits=8,
                group_size=32,  # Smaller groups for better accuracy
            )

        if mode == QwenQuantizationMode.CONSERVATIVE:
            # Preserve text encoder quality (critical for semantic understanding)
            # Qwen text encoder is sensitive to quantization
            return cls(
                transformer_bits=4,
                vae_bits=8,
                text_encoder_bits=None,  # Keep text encoder at full precision
                group_size=64,
            )

        # Fallback
        return cls()

    @classmethod
    def from_bits(cls, bits: int | None) -> "QwenQuantizationConfig":
        """Create uniform config with same bits for all components.

        Args:
            bits: Quantization bits for all components (2, 4, 8, or None)

        Returns:
            QwenQuantizationConfig with uniform settings

        Raises:
            ValueError: If bits is not a valid quantization value
        """
        if bits is not None and bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, 8, or None. Got {bits}")
        return cls(
            transformer_bits=bits,
            vae_bits=bits,
            text_encoder_bits=bits,
        )

    def get_transformer_config(self) -> QwenComponentQuantization:
        """Get quantization config for transformer component."""
        return QwenComponentQuantization(
            bits=self.transformer_bits,
            group_size=self.group_size,
            exclude_layers=self.exclude_layers,
        )

    def get_vae_config(self) -> QwenComponentQuantization:
        """Get quantization config for VAE component."""
        return QwenComponentQuantization(
            bits=self.vae_bits,
            group_size=self.group_size,
            exclude_layers=self.exclude_layers,
        )

    def get_text_encoder_config(self) -> QwenComponentQuantization:
        """Get quantization config for text encoder component."""
        return QwenComponentQuantization(
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

        layer_name_lower = layer_name.lower()

        # Check for attention override
        if self.attention_bits is not None:
            # Qwen uses "attn" in transformer block attention layers
            attention_patterns = ["attention", "attn", "self_attn", "cross_attn", "qkv", "q_proj", "k_proj", "v_proj"]
            if any(p in layer_name_lower for p in attention_patterns):
                return self.attention_bits

        # Check for FFN override
        if self.ffn_bits is not None:
            # Qwen uses "mlp" and "feed_forward" patterns
            ffn_patterns = ["ffn", "mlp", "feed_forward", "dense", "gate_proj", "up_proj", "down_proj"]
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
    def from_dict(cls, data: dict[str, Any]) -> "QwenQuantizationConfig":
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
            return "QwenQuantizationConfig(none)"

        parts = []
        if self.transformer_bits:
            parts.append(f"transformer={self.transformer_bits}b")
        if self.vae_bits:
            parts.append(f"vae={self.vae_bits}b")
        if self.text_encoder_bits:
            parts.append(f"text_encoder={self.text_encoder_bits}b")
        if self.attention_bits:
            parts.append(f"attn={self.attention_bits}b")
        if self.ffn_bits:
            parts.append(f"ffn={self.ffn_bits}b")

        return f"QwenQuantizationConfig({', '.join(parts)})"


def estimate_memory_usage(config: QwenQuantizationConfig) -> dict[str, float]:
    """Estimate memory usage for Qwen model with given quantization config.

    Based on qwen-image-2512 architecture:
    - Text encoder: ~12.6 GB (BF16) - 7B parameters
    - Transformer: ~22 GB (BF16) - ~6B parameters (60 DiT blocks)
    - VAE: ~242 MB (BF16)

    Args:
        config: Quantization configuration

    Returns:
        Dictionary with estimated memory usage in GB for each component
    """
    # Base sizes in GB at BF16 (16 bits = 2 bytes per param)
    # These values are specific to qwen-image-2512 model architecture
    BASE_TEXT_ENCODER_GB = 12.6  # ~7B params × 2 bytes
    BASE_TRANSFORMER_GB = 22.0  # ~6B params × 2 bytes (60 DiT blocks)
    BASE_VAE_GB = 0.242  # ~121M params × 2 bytes

    def scale_by_bits(base_gb: float, bits: int | None) -> float:
        if bits is None:
            return base_gb
        # BF16 is 16 bits, so scaling factor is bits/16
        return base_gb * (bits / 16.0)

    text_encoder_gb = scale_by_bits(BASE_TEXT_ENCODER_GB, config.text_encoder_bits)
    transformer_gb = scale_by_bits(BASE_TRANSFORMER_GB, config.transformer_bits)
    vae_gb = scale_by_bits(BASE_VAE_GB, config.vae_bits)

    total_gb = text_encoder_gb + transformer_gb + vae_gb

    return {
        "text_encoder_gb": round(text_encoder_gb, 2),
        "transformer_gb": round(transformer_gb, 2),
        "vae_gb": round(vae_gb, 2),  # Consistent 2-decimal precision
        "total_gb": round(total_gb, 2),
        "savings_gb": round(BASE_TEXT_ENCODER_GB + BASE_TRANSFORMER_GB + BASE_VAE_GB - total_gb, 2),
    }
