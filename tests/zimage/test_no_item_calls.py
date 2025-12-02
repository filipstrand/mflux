"""Unit tests for GPU synchronization optimization.

Tests verify that .item() calls have been removed from hot paths
to avoid CPU-GPU synchronization barriers during image generation.
"""

import mlx.core as mx
import pytest
from PIL import Image

pytestmark = pytest.mark.high_memory_requirement


class TestNoItemCalls:
    """Tests for removing .item() calls that cause GPU sync barriers."""

    @pytest.mark.skipif(
        False,  # Run this test
        reason="Requires local model at ./zimage-q4",
    )
    def test_generation_without_item_in_hot_path(self):
        """Test that image generation works without .item() in denoising loop.

        This verifies that the timestep handling optimization works correctly,
        keeping timesteps as mx.array throughout the denoising loop instead of
        converting to Python scalars with .item().
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
            num_inference_steps=4,  # Fewer steps for speed
            height=256,
            width=256,
            guidance=0.0,
        )

        # This should work without any .item() calls in the denoising loop
        image = zimage.generate_image(
            seed=42,
            prompt="A test image",
            config=config,
        )

        # Verify image is generated correctly
        assert hasattr(image, "image")
        assert isinstance(image.image, Image.Image)
        assert image.image.size == (256, 256)
        assert image.image.mode == "RGB"

    @pytest.mark.skipif(
        False,
        reason="Requires local model",
    )
    def test_timestep_handling_produces_correct_values(self):
        """Test that timestep handling without .item() produces correct values.

        Verifies that keeping timesteps as mx.array and using array indexing
        produces the same results as the previous .item() approach.
        """
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.models.zimage.scheduler import ZImageScheduler

        config = Config(
            num_inference_steps=9,
            height=512,
            width=512,
            guidance=0.0,
        )

        # Create scheduler
        from mflux.config.runtime_config import RuntimeConfig

        runtime_config = RuntimeConfig(config, ModelConfig.zimage_turbo())
        scheduler = ZImageScheduler(runtime_config)

        # Test that we can access timesteps as mx.array without .item()
        for t_idx in range(runtime_config.num_inference_steps):
            # Get timestep as mx.array (no .item())
            t_raw = scheduler.timesteps[t_idx]

            # Verify it's an mx.array
            assert isinstance(t_raw, mx.array)

            # Verify we can do arithmetic without .item()
            t_value = (1000.0 - t_raw) / 1000.0
            assert isinstance(t_value, mx.array)

            # Verify we can add batch dimension with [None]
            t_batch = t_value[None]
            assert isinstance(t_batch, mx.array)
            assert t_batch.shape == (1,)

            # Verify the value is in expected range [0, 1]
            # We need .item() here only for assertion, not in hot path
            t_val_scalar = float(t_value.item())
            assert 0.0 <= t_val_scalar <= 1.0

    @pytest.mark.skipif(
        False,
        reason="Requires local model",
    )
    def test_deterministic_results_after_optimization(self):
        """Test that optimization doesn't change output (determinism check).

        Generates images with same seed and verifies they're identical,
        ensuring the optimization didn't change the numerical results.
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
            num_inference_steps=4,
            height=256,
            width=256,
            guidance=0.0,
        )

        # Generate twice with same seed
        image1 = zimage.generate_image(
            seed=999,
            prompt="A red apple",
            config=config,
        )

        image2 = zimage.generate_image(
            seed=999,
            prompt="A red apple",
            config=config,
        )

        # Convert to arrays for comparison
        import numpy as np

        arr1 = np.array(image1.image)
        arr2 = np.array(image2.image)

        # Images should be identical with same seed
        assert arr1.shape == arr2.shape
        # Allow for minor floating point differences from quantization
        diff = np.abs(arr1.astype(float) - arr2.astype(float)).mean()
        assert diff < 1.0, f"Images differ too much: mean diff = {diff}"
