"""
Integration tests for Chroma model.

These are slow tests that require model loading and actual inference.
Run with: pytest tests/chroma/test_chroma_integration.py -m slow
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from mflux.models.chroma.variants.txt2img.chroma import Chroma
from mflux.models.common.config import ModelConfig


class TestChromaGeneration:
    """Integration tests for Chroma image generation."""

    @pytest.mark.slow
    def test_chroma_generates_valid_image(self):
        """Test that Chroma generates a valid image."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,  # Use 8-bit for faster test
        )

        image = chroma.generate_image(
            seed=42,
            prompt="a simple red circle on white background",
            num_inference_steps=4,  # Minimal steps for test
            height=256,
            width=256,
        )

        # Verify image was generated
        assert image is not None
        assert image.image is not None

        # Verify image dimensions
        assert image.image.shape[0] == 256  # height
        assert image.image.shape[1] == 256  # width
        assert image.image.shape[2] == 3  # RGB channels

    @pytest.mark.slow
    def test_chroma_deterministic_with_seed(self):
        """Test that same seed produces same output."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,
        )

        image1 = chroma.generate_image(
            seed=42,
            prompt="a blue square",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        image2 = chroma.generate_image(
            seed=42,
            prompt="a blue square",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        # Same seed should produce identical results
        np.testing.assert_array_equal(image1.image, image2.image)

    @pytest.mark.slow
    def test_chroma_different_seeds_produce_different_images(self):
        """Test that different seeds produce different outputs."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,
        )

        image1 = chroma.generate_image(
            seed=42,
            prompt="a cat",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        image2 = chroma.generate_image(
            seed=123,
            prompt="a cat",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        # Different seeds should produce different results
        assert not np.array_equal(image1.image, image2.image)


class TestChromaQuantization:
    """Integration tests for Chroma quantization."""

    @pytest.mark.slow
    def test_chroma_4bit_quantization(self):
        """Test 4-bit quantization works."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=4,
        )

        image = chroma.generate_image(
            seed=42,
            prompt="a green triangle",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        assert image is not None
        assert image.image is not None
        assert image.image.shape == (256, 256, 3)

    @pytest.mark.slow
    def test_chroma_8bit_quantization(self):
        """Test 8-bit quantization works."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,
        )

        image = chroma.generate_image(
            seed=42,
            prompt="a yellow circle",
            num_inference_steps=4,
            height=256,
            width=256,
        )

        assert image is not None
        assert image.image is not None
        assert image.image.shape == (256, 256, 3)


class TestChromaSaveLoad:
    """Integration tests for Chroma model saving and loading."""

    @pytest.mark.slow
    def test_save_and_load_quantized_model(self):
        """Test saving and loading a quantized Chroma model."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "chroma-test")

        try:
            # Create and save model
            chroma1 = Chroma(
                model_config=ModelConfig.chroma(),
                quantize=8,
            )

            image1 = chroma1.generate_image(
                seed=42,
                prompt="test image",
                num_inference_steps=4,
                height=256,
                width=256,
            )

            chroma1.save_model(model_path)

            # Verify saved files exist
            assert os.path.exists(model_path)
            assert os.path.exists(os.path.join(model_path, "transformer"))
            assert os.path.exists(os.path.join(model_path, "vae"))
            assert os.path.exists(os.path.join(model_path, "text_encoder"))

            # Load saved model
            chroma2 = Chroma(
                model_config=ModelConfig.chroma(),
                model_path=model_path,
            )

            image2 = chroma2.generate_image(
                seed=42,
                prompt="test image",
                num_inference_steps=4,
                height=256,
                width=256,
            )

            # Same seed/prompt should produce identical results
            np.testing.assert_array_equal(image1.image, image2.image)

        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestChromaModelConfig:
    """Tests for Chroma model configuration."""

    @pytest.mark.fast
    def test_chroma_config_exists(self):
        """Verify Chroma config is registered."""
        config = ModelConfig.chroma()
        assert config is not None

    @pytest.mark.fast
    def test_chroma_config_values(self):
        """Verify Chroma config has correct values."""
        config = ModelConfig.chroma()

        assert config.model_name == "lodestones/Chroma1-HD"
        assert "chroma" in config.aliases
        assert config.max_sequence_length == 512
        assert config.supports_guidance is True

    @pytest.mark.fast
    def test_chroma_from_name(self):
        """Verify Chroma can be loaded by name."""
        config = ModelConfig.from_name("chroma", base_model=None)
        assert config is not None
        assert config.model_name == "lodestones/Chroma1-HD"

    @pytest.mark.fast
    def test_chroma_aliases(self):
        """Verify all Chroma aliases work."""
        for alias in ["chroma", "chroma-hd", "chroma1-hd"]:
            config = ModelConfig.from_name(alias, base_model=None)
            assert config is not None
            assert config.model_name == "lodestones/Chroma1-HD"


class TestChromaPromptEncoding:
    """Tests for Chroma prompt encoding (T5-only)."""

    @pytest.mark.slow
    def test_prompt_encoding_works(self):
        """Test that prompt encoding produces valid embeddings."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,
        )

        # Encode a prompt
        prompt = "a beautiful sunset over the ocean"
        embeddings = chroma._encode_prompt(prompt)

        # Verify embeddings shape [batch, seq_len, 4096]
        assert embeddings.ndim == 3
        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[2] == 4096  # T5 hidden size

    @pytest.mark.slow
    def test_prompt_caching(self):
        """Test that prompt embeddings are cached."""
        chroma = Chroma(
            model_config=ModelConfig.chroma(),
            quantize=8,
        )

        prompt = "a test prompt for caching"

        # First call should not be cached
        assert prompt not in chroma.prompt_cache

        # Encode prompt
        embeddings1 = chroma._encode_prompt(prompt)

        # Should now be cached
        assert prompt in chroma.prompt_cache

        # Second call should return cached value
        embeddings2 = chroma._encode_prompt(prompt)

        # Should be identical (same object from cache)
        assert embeddings1 is embeddings2
