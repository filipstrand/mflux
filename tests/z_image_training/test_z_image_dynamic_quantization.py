"""Tests for Z-Image dynamic quantization infrastructure.

Tests quantization configuration, mode presets, and runtime quantizer.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx import nn


class TestQuantizationMode:
    """Tests for QuantizationMode enum."""

    def test_all_modes_defined(self):
        """Test that all expected modes are defined."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        expected_modes = ["NONE", "INT2", "INT4", "INT8", "MIXED", "SPEED", "QUALITY"]

        for mode_name in expected_modes:
            assert hasattr(QuantizationMode, mode_name)

    def test_mode_values_are_strings(self):
        """Test that mode values are lowercase strings."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        for mode in QuantizationMode:
            assert isinstance(mode.value, str)
            assert mode.value == mode.value.lower()

    def test_from_string_with_mode_name(self):
        """Test converting string to mode."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        assert QuantizationMode.from_string("speed") == QuantizationMode.SPEED
        assert QuantizationMode.from_string("quality") == QuantizationMode.QUALITY
        assert QuantizationMode.from_string("mixed") == QuantizationMode.MIXED

    def test_from_string_with_int(self):
        """Test converting int to mode."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        assert QuantizationMode.from_string(2) == QuantizationMode.INT2
        assert QuantizationMode.from_string(4) == QuantizationMode.INT4
        assert QuantizationMode.from_string(8) == QuantizationMode.INT8

    def test_from_string_with_none(self):
        """Test None input returns None."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        assert QuantizationMode.from_string(None) is None

    def test_from_string_case_insensitive(self):
        """Test string conversion is case insensitive."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationMode

        assert QuantizationMode.from_string("SPEED") == QuantizationMode.SPEED
        assert QuantizationMode.from_string("Speed") == QuantizationMode.SPEED


class TestComponentQuantization:
    """Tests for ComponentQuantization dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        from mflux.models.z_image.weights.dynamic_quantization import ComponentQuantization

        config = ComponentQuantization()

        assert config.bits is None
        assert config.group_size == 64
        assert config.exclude_layers == []

    def test_valid_bits_values(self):
        """Test valid bits values are accepted."""
        from mflux.models.z_image.weights.dynamic_quantization import ComponentQuantization

        for bits in [None, 2, 4, 8]:
            config = ComponentQuantization(bits=bits)
            assert config.bits == bits

    def test_invalid_bits_raises(self):
        """Test invalid bits raises error."""
        from mflux.models.z_image.weights.dynamic_quantization import ComponentQuantization

        with pytest.raises(ValueError, match="bits must be"):
            ComponentQuantization(bits=3)

        with pytest.raises(ValueError, match="bits must be"):
            ComponentQuantization(bits=16)

    def test_invalid_group_size_raises(self):
        """Test invalid group_size raises error."""
        from mflux.models.z_image.weights.dynamic_quantization import ComponentQuantization

        with pytest.raises(ValueError, match="group_size"):
            ComponentQuantization(group_size=0)


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_default_no_quantization(self):
        """Test default config has no quantization."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig()

        assert config.transformer_bits is None
        assert config.vae_bits is None
        assert config.text_encoder_bits is None
        assert config.is_quantized is False

    def test_custom_bits(self):
        """Test custom bits configuration."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(
            transformer_bits=4,
            vae_bits=8,
            text_encoder_bits=8,
        )

        assert config.transformer_bits == 4
        assert config.vae_bits == 8
        assert config.text_encoder_bits == 8
        assert config.is_quantized is True

    def test_from_mode_none(self):
        """Test creating config from NONE mode."""
        from mflux.models.z_image.weights.dynamic_quantization import (
            QuantizationConfig,
            QuantizationMode,
        )

        config = QuantizationConfig.from_mode(QuantizationMode.NONE)

        assert config.is_quantized is False

    def test_from_mode_int4(self):
        """Test creating config from INT4 mode."""
        from mflux.models.z_image.weights.dynamic_quantization import (
            QuantizationConfig,
            QuantizationMode,
        )

        config = QuantizationConfig.from_mode(QuantizationMode.INT4)

        assert config.transformer_bits == 4
        assert config.is_quantized is True

    def test_from_mode_mixed(self):
        """Test creating config from MIXED mode."""
        from mflux.models.z_image.weights.dynamic_quantization import (
            QuantizationConfig,
            QuantizationMode,
        )

        config = QuantizationConfig.from_mode(QuantizationMode.MIXED)

        # Mixed mode should have different attention and FFN bits
        assert config.attention_bits == 8
        assert config.ffn_bits == 4

    def test_from_mode_speed(self):
        """Test creating config from SPEED mode."""
        from mflux.models.z_image.weights.dynamic_quantization import (
            QuantizationConfig,
            QuantizationMode,
        )

        config = QuantizationConfig.from_mode(QuantizationMode.SPEED)

        # Speed mode should be aggressively quantized
        assert config.transformer_bits == 4
        assert config.group_size >= 64  # Larger groups for speed

    def test_from_mode_quality(self):
        """Test creating config from QUALITY mode."""
        from mflux.models.z_image.weights.dynamic_quantization import (
            QuantizationConfig,
            QuantizationMode,
        )

        config = QuantizationConfig.from_mode(QuantizationMode.QUALITY)

        # Quality mode should preserve precision
        assert config.transformer_bits == 8
        assert config.vae_bits is None  # Keep VAE at full precision

    def test_from_mode_with_string(self):
        """Test from_mode accepts string."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig.from_mode("speed")

        assert config.transformer_bits == 4

    def test_from_mode_with_int(self):
        """Test from_mode accepts int."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig.from_mode(4)

        assert config.transformer_bits == 4

    def test_from_bits_uniform(self):
        """Test from_bits creates uniform config."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig.from_bits(8)

        assert config.transformer_bits == 8
        assert config.vae_bits == 8
        assert config.text_encoder_bits == 8

    def test_effective_bits_for_layer_default(self):
        """Test effective_bits_for_layer returns transformer_bits by default."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(transformer_bits=4)

        bits = config.effective_bits_for_layer("layers.0.linear")

        assert bits == 4

    def test_effective_bits_for_attention_override(self):
        """Test attention_bits override."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(
            transformer_bits=4,
            attention_bits=8,
        )

        bits = config.effective_bits_for_layer("layers.0.self_attn.query")

        assert bits == 8

    def test_effective_bits_for_ffn_override(self):
        """Test ffn_bits override."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(
            transformer_bits=8,
            ffn_bits=4,
        )

        bits = config.effective_bits_for_layer("layers.0.mlp.dense")

        assert bits == 4

    def test_effective_bits_excludes_layers(self):
        """Test exclude_layers returns None."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(
            transformer_bits=4,
            exclude_layers=["embed", "norm"],
        )

        bits = config.effective_bits_for_layer("embed_tokens")

        assert bits is None

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        original = QuantizationConfig(
            transformer_bits=4,
            vae_bits=8,
            attention_bits=8,
            exclude_layers=["norm"],
        )

        data = original.to_dict()
        restored = QuantizationConfig.from_dict(data)

        assert restored.transformer_bits == original.transformer_bits
        assert restored.vae_bits == original.vae_bits
        assert restored.attention_bits == original.attention_bits
        assert restored.exclude_layers == original.exclude_layers

    def test_min_bits_property(self):
        """Test min_bits returns smallest quantization."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(
            transformer_bits=4,
            vae_bits=8,
            text_encoder_bits=4,
        )

        assert config.min_bits == 4

    def test_min_bits_none_when_unquantized(self):
        """Test min_bits returns None when no quantization."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig()

        assert config.min_bits is None

    def test_str_representation(self):
        """Test string representation."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig

        config = QuantizationConfig(transformer_bits=4)

        str_repr = str(config)

        assert "QuantizationConfig" in str_repr
        assert "4b" in str_repr


class TestRuntimeQuantizer:
    """Tests for RuntimeQuantizer class."""

    def test_init_with_config(self):
        """Test initializing with config."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import RuntimeQuantizer

        config = QuantizationConfig(transformer_bits=4)
        quantizer = RuntimeQuantizer(config)

        assert quantizer.config == config

    def test_stats_initialized(self):
        """Test that stats are initialized."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import RuntimeQuantizer

        config = QuantizationConfig()
        quantizer = RuntimeQuantizer(config)

        stats = quantizer.get_stats()

        assert "layers_quantized" in stats
        assert "layers_skipped" in stats

    def test_quantize_model_returns_model(self):
        """Test that quantize_model returns the model."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import RuntimeQuantizer

        config = QuantizationConfig()  # No quantization
        quantizer = RuntimeQuantizer(config)

        model = MagicMock()
        result = quantizer.quantize_model(model)

        assert result is model


class TestQuantizeModelFunction:
    """Tests for quantize_model convenience function."""

    def test_with_mode_string(self):
        """Test quantize_model with mode string."""
        from mflux.models.z_image.weights.runtime_quantizer import quantize_model

        model = MagicMock()
        model.transformer = nn.Linear(64, 64)
        model.vae = nn.Linear(64, 64)
        model.text_encoder = nn.Linear(64, 64)

        result = quantize_model(model, mode="speed")

        assert result is model

    def test_with_mode_int(self):
        """Test quantize_model with mode int."""
        from mflux.models.z_image.weights.runtime_quantizer import quantize_model

        model = MagicMock()
        result = quantize_model(model, mode=4)

        assert result is model

    def test_with_config(self):
        """Test quantize_model with explicit config."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import quantize_model

        model = MagicMock()
        config = QuantizationConfig(transformer_bits=8)

        result = quantize_model(model, config=config)

        assert result is model


class TestEstimateQuantizedSize:
    """Tests for size estimation function."""

    def test_returns_expected_keys(self):
        """Test that estimate returns expected keys."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import estimate_quantized_size

        model = MagicMock()
        model.parameters.return_value = [mx.ones((100, 100))]
        model.transformer = MagicMock()
        model.transformer.parameters.return_value = [mx.ones((50, 50))]
        model.vae = MagicMock()
        model.vae.parameters.return_value = [mx.ones((25, 25))]
        model.text_encoder = MagicMock()
        model.text_encoder.parameters.return_value = [mx.ones((25, 25))]

        config = QuantizationConfig(transformer_bits=4)

        estimates = estimate_quantized_size(model, config)

        assert "full_precision_mb" in estimates
        assert "quantized_mb" in estimates
        assert "compression_ratio" in estimates
        assert "memory_saved_mb" in estimates

    def test_quantized_smaller_than_full(self):
        """Test that quantized size is smaller than full precision."""
        from mflux.models.z_image.weights.dynamic_quantization import QuantizationConfig
        from mflux.models.z_image.weights.runtime_quantizer import estimate_quantized_size

        # Create model with known parameters
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Linear(1000, 1000)
                self.vae = nn.Linear(500, 500)
                self.text_encoder = nn.Linear(500, 500)

        model = MockModel()
        config = QuantizationConfig.from_bits(4)

        estimates = estimate_quantized_size(model, config)

        # INT4 should be ~4x smaller than BF16
        assert estimates["quantized_mb"] < estimates["full_precision_mb"]
        assert estimates["compression_ratio"] > 1.0
