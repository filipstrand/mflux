"""Test that memory is properly released when using ZImage as a library."""

import gc

import mlx.core as mx
import pytest

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.zimage import ZImage

pytestmark = pytest.mark.high_memory_requirement


class TestLibraryMemoryCleanup:
    """Verify memory cleanup works without CLI-level cleanup."""

    @pytest.mark.skipif(False, reason="Requires local model at ./zimage-q4")
    def test_generation_tensors_released_after_image_deleted(self):
        """After deleting generated image, memory should return to baseline."""
        # Load model
        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            local_path="./zimage-q4",
            quantize=None,
        )

        # Measure baseline (model loaded, no generation)
        mx.eval(zimage.parameters())  # Ensure model is fully loaded
        gc.collect()
        mx.clear_cache()
        mem_baseline = mx.get_active_memory()

        # Generate image
        config = Config(
            num_inference_steps=2,  # Minimal steps for speed
            height=512,
            width=512,
            guidance=0.0,
        )
        image = zimage.generate_image(seed=42, prompt="test", config=config)

        # Delete image and cleanup
        del image
        gc.collect()
        mx.clear_cache()
        mem_after = mx.get_active_memory()

        # Generation tensors should be released
        # Allow 20% tolerance for MLX internal caching
        assert mem_after < mem_baseline * 1.2, (
            f"Generation tensors not released: baseline={mem_baseline / 1e9:.2f}GB, "
            f"after={mem_after / 1e9:.2f}GB (expected < {mem_baseline * 1.2 / 1e9:.2f}GB)"
        )

        # Model should still be in memory
        assert mem_after > mem_baseline * 0.5, (
            f"Model was unexpectedly released: baseline={mem_baseline / 1e9:.2f}GB, after={mem_after / 1e9:.2f}GB"
        )
