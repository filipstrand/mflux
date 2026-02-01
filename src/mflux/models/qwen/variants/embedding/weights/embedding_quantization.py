"""Quantization configuration for embedding models.

Embedding models require careful quantization to preserve output quality:
- Text encoder: Full precision (critical for embedding quality)
- Vision encoder: INT8 safe (minor quality impact)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantizationMode(Enum):
    """Quantization modes for embedding models."""

    NONE = "none"  # No quantization (full precision)
    CONSERVATIVE = "conservative"  # Only quantize vision encoder
    AGGRESSIVE = "aggressive"  # Quantize everything (not recommended)


@dataclass
class EmbeddingQuantizationConfig:
    """Quantization configuration for embedding models.

    For embedding models, we recommend CONSERVATIVE mode:
    - Text encoder kept at full precision (critical for embedding quality)
    - Vision encoder can be quantized to INT8 (minor quality impact)

    Benchmarks show:
    - Full precision: Quality baseline
    - INT8 vision: <0.5% quality degradation, 20-30% memory savings
    - INT8 everything: 2-5% quality degradation (not recommended)
    """

    mode: QuantizationMode = QuantizationMode.CONSERVATIVE

    # Per-component settings
    text_encoder_bits: Optional[int] = None  # None = full precision
    vision_encoder_bits: int = 8  # INT8 for vision
    score_weight_bits: Optional[int] = None  # None = full precision

    # Quantization parameters
    group_size: int = 64  # Group size for group quantization
    calibration_samples: int = 128  # Samples for calibration

    @classmethod
    def full_precision(cls) -> "EmbeddingQuantizationConfig":
        """Create a full-precision configuration."""
        return cls(
            mode=QuantizationMode.NONE,
            text_encoder_bits=None,
            vision_encoder_bits=None,
            score_weight_bits=None,
        )

    @classmethod
    def conservative(cls) -> "EmbeddingQuantizationConfig":
        """Create a conservative configuration (recommended).

        Only quantizes vision encoder, preserves text encoder precision.
        """
        return cls(
            mode=QuantizationMode.CONSERVATIVE,
            text_encoder_bits=None,
            vision_encoder_bits=8,
            score_weight_bits=None,
        )

    @classmethod
    def aggressive(cls) -> "EmbeddingQuantizationConfig":
        """Create an aggressive configuration (not recommended).

        Quantizes everything including text encoder.
        May cause 2-5% quality degradation.
        """
        return cls(
            mode=QuantizationMode.AGGRESSIVE,
            text_encoder_bits=8,
            vision_encoder_bits=8,
            score_weight_bits=None,  # Always keep score weight at full precision
        )

    def should_quantize_layer(self, layer_name: str) -> Optional[int]:
        """Determine if a layer should be quantized.

        Args:
            layer_name: Full layer name (e.g., "encoder.layers.0.self_attn.q_proj")

        Returns:
            Quantization bits (4, 8) or None for full precision
        """
        if self.mode == QuantizationMode.NONE:
            return None

        # Vision encoder layers
        if "visual" in layer_name or "vision" in layer_name:
            return self.vision_encoder_bits

        # Text encoder layers
        if "encoder" in layer_name and "visual" not in layer_name:
            return self.text_encoder_bits

        # Score weight (reranker)
        if "score_weight" in layer_name:
            return self.score_weight_bits

        # Default to text encoder setting
        return self.text_encoder_bits
