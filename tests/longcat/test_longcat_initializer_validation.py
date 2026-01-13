"""
Unit tests for LongCat initializer validation.

Tests verify:
1. Model config type validation
2. Model config attribute validation
3. Quantize value validation
4. LoRA paths and scales length matching
"""

import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.longcat.longcat_initializer import LongCatInitializer


class TestLongCatInitializerValidation:
    """Tests for input validation in LongCat initializer."""

    @pytest.mark.fast
    def test_validate_valid_config(self):
        """Verify validation passes with valid ModelConfig."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = None
        lora_scales = None

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_invalid_config_type(self):
        """Verify validation fails with non-ModelConfig type."""
        invalid_config = {"model_name": "test"}  # dict instead of ModelConfig
        quantize = None
        lora_paths = None
        lora_scales = None

        with pytest.raises(TypeError, match="model_config must be a ModelConfig instance"):
            LongCatInitializer._validate_inputs(invalid_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_missing_model_name_attribute(self):
        """Verify validation fails if model_config lacks model_name."""
        # Create a ModelConfig subclass without model_name
        class InvalidConfig(ModelConfig):
            def __init__(self):
                # Don't call super().__init__, just set max_sequence_length
                self.max_sequence_length = 512

        invalid_config = InvalidConfig()
        quantize = None
        lora_paths = None
        lora_scales = None

        with pytest.raises(TypeError, match="model_config is missing required attributes.*model_name"):
            LongCatInitializer._validate_inputs(invalid_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_missing_max_sequence_length_attribute(self):
        """Verify validation fails if model_config lacks max_sequence_length."""
        # Create a ModelConfig subclass without max_sequence_length
        class InvalidConfig(ModelConfig):
            def __init__(self):
                # Don't call super().__init__, just set model_name
                self.model_name = "test-model"

        invalid_config = InvalidConfig()
        quantize = None
        lora_paths = None
        lora_scales = None

        with pytest.raises(TypeError, match="model_config is missing required attributes.*max_sequence_length"):
            LongCatInitializer._validate_inputs(invalid_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_quantize_none(self):
        """Verify validation passes with quantize=None."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = None
        lora_scales = None

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_quantize_4(self):
        """Verify validation passes with quantize=4."""
        model_config = ModelConfig.longcat()
        quantize = 4
        lora_paths = None
        lora_scales = None

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_quantize_8(self):
        """Verify validation passes with quantize=8."""
        model_config = ModelConfig.longcat()
        quantize = 8
        lora_paths = None
        lora_scales = None

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_invalid_quantize_value(self):
        """Verify validation fails with invalid quantize value."""
        model_config = ModelConfig.longcat()
        quantize = 16  # Invalid value
        lora_paths = None
        lora_scales = None

        with pytest.raises(ValueError, match="quantize must be 4, 8, or None"):
            LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_invalid_quantize_negative(self):
        """Verify validation fails with negative quantize value."""
        model_config = ModelConfig.longcat()
        quantize = -1
        lora_paths = None
        lora_scales = None

        with pytest.raises(ValueError, match="quantize must be 4, 8, or None"):
            LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_matching_lora_lengths(self):
        """Verify validation passes when lora_paths and lora_scales have same length."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = ["path1.safetensors", "path2.safetensors"]
        lora_scales = [1.0, 0.5]

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_mismatched_lora_lengths(self):
        """Verify validation fails when lora_paths and lora_scales have different lengths."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = ["path1.safetensors", "path2.safetensors", "path3.safetensors"]
        lora_scales = [1.0, 0.5]  # Only 2 scales for 3 paths

        with pytest.raises(
            ValueError, match="lora_paths and lora_scales must have the same length.*3 paths and 2 scales"
        ):
            LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_lora_paths_only(self):
        """Verify validation passes when only lora_paths is provided."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = ["path1.safetensors"]
        lora_scales = None

        # Should not raise any exception (scales will be defaulted elsewhere)
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_lora_scales_only(self):
        """Verify validation passes when only lora_scales is provided."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = None
        lora_scales = [1.0, 0.5]

        # Should not raise any exception (paths=None, so length check is skipped)
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_empty_lora_lists(self):
        """Verify validation passes with empty lora lists."""
        model_config = ModelConfig.longcat()
        quantize = None
        lora_paths = []
        lora_scales = []

        # Should not raise any exception (both empty = same length)
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)

    @pytest.mark.fast
    def test_validate_all_valid_parameters(self):
        """Verify validation passes with all valid parameters."""
        model_config = ModelConfig.longcat()
        quantize = 4
        lora_paths = ["lora1.safetensors", "lora2.safetensors"]
        lora_scales = [1.0, 0.8]

        # Should not raise any exception
        LongCatInitializer._validate_inputs(model_config, quantize, lora_paths, lora_scales)
