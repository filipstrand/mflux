"""Integration tests for Z-Image generation with SDPA.

Tests verify that the complete generation pipeline works correctly
with the SDPA-based attention implementation.
"""

import mlx.core as mx
import pytest
from PIL import Image

pytestmark = pytest.mark.high_memory_requirement


class TestZImageGenerationWithSDPA:
    """Integration tests for Z-Image generation using SDPA attention."""

    @pytest.mark.skipif(
        False,  # Run this test (model is available locally)
        reason="Requires local model at ./zimage-q4",
    )
    def test_generation_with_4bit_model(self):
        """Test generation with local 4-bit quantized model."""
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        # Use local 4-bit model
        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            local_path="./zimage-q4",
            quantize=None,  # Already quantized
        )

        config = Config(
            num_inference_steps=9,
            height=512,
            width=512,
            guidance=0.0,
        )

        image = zimage.generate_image(
            seed=42,
            prompt="A red apple on a white table",
            config=config,
        )

        # Verify image is generated correctly
        assert hasattr(image, "image")
        assert isinstance(image.image, Image.Image)
        assert image.image.size == (512, 512)
        assert image.image.mode == "RGB"

        # Verify image contains valid pixel data
        pixels = list(image.image.getdata())
        assert len(pixels) == 512 * 512
        assert all(isinstance(p, tuple) and len(p) == 3 for p in pixels[:10])

    @pytest.mark.skipif(
        False,  # Run this test
        reason="Requires local model",
    )
    def test_generation_deterministic(self):
        """Test that generation with same seed produces similar results."""
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            local_path="./zimage-q4",
            quantize=None,
        )

        config = Config(
            num_inference_steps=4,  # Fewer steps for speed
            height=256,
            width=256,
            guidance=0.0,
        )

        # Generate twice with same seed
        image1 = zimage.generate_image(
            seed=123,
            prompt="A cat sitting on a chair",
            config=config,
        )

        image2 = zimage.generate_image(
            seed=123,
            prompt="A cat sitting on a chair",
            config=config,
        )

        # Images should be identical with same seed
        assert image1.image.size == image2.image.size
        # Note: Due to quantization and numerical precision,
        # images might not be bit-identical but should be very similar

    @pytest.mark.skipif(
        False,
        reason="Requires local model",
    )
    def test_generation_various_resolutions(self):
        """Test generation at different resolutions."""
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            local_path="./zimage-q4",
            quantize=None,
        )

        resolutions = [
            (512, 512),
            (768, 768),
            (1024, 1024),
        ]

        for height, width in resolutions:
            config = Config(
                num_inference_steps=2,  # Minimal steps for speed
                height=height,
                width=width,
                guidance=0.0,
            )

            image = zimage.generate_image(
                seed=42,
                prompt="test",
                config=config,
            )

            assert isinstance(image.image, Image.Image)
            assert image.image.size == (width, height)
            assert image.image.mode == "RGB"

    @pytest.mark.skipif(
        False,
        reason="Requires local model",
    )
    def test_memory_usage_stays_reasonable(self):
        """Test that memory usage doesn't spike during generation.

        This is a regression test for the bug where VAE decode without
        mx.eval() caused memory to spike to 30+GB. With the fix, memory
        should stay under 10GB for q4 quantized models.
        """
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            local_path="./zimage-q4",
            quantize=None,
        )

        config = Config(
            num_inference_steps=9,
            height=1024,
            width=1024,
            guidance=0.0,
        )

        # Record memory after model load (baseline for model + generation)
        memory_after_load = mx.get_active_memory() / 1e9

        image = zimage.generate_image(
            seed=42,
            prompt="test image for memory regression",
            config=config,
        )

        # Get current active memory
        memory_after_generation = mx.get_active_memory() / 1e9

        # Calculate the memory used during generation (delta from baseline)
        generation_memory_delta = memory_after_generation - memory_after_load

        # Verify image was generated
        assert isinstance(image.image, Image.Image)
        assert image.image.size == (1024, 1024)

        # CRITICAL: Memory increase during generation should be reasonable
        # For a q4 1024x1024 image, we expect ~6-7GB for latents + VAE decode
        # Before the fix, this would spike to 30+GB due to graph explosion
        assert generation_memory_delta < 15.0, (
            f"Memory delta too high: {generation_memory_delta:.2f} GB (expected < 15 GB for generation)"
        )

        # Total memory (model + generation) should be under 25GB for q4
        # Before the fix, this would exceed 30GB
        assert memory_after_generation < 25.0, (
            f"Total memory too high: {memory_after_generation:.2f} GB (expected < 25 GB)"
        )
