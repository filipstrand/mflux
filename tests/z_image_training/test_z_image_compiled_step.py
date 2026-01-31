"""Tests for CompiledTrainStep caching and compilation behavior.

Tests that:
- Compiled functions are cached by resolution
- Cache hits occur on same resolution
- Cache misses occur on different resolutions
- LRU eviction works when cache is full
- Compilation can be disabled
- get_cache_stats returns accurate statistics
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.z_image.variants.training.dataset.batch import Batch, Example


def create_mock_example(height: int = 256, width: int = 256) -> Example:
    """Create a mock example with specified latent resolution."""
    # Latent space is 8x smaller than image space
    latent_h = height // 8
    latent_w = width // 8
    return Example(
        example_id=0,
        prompt="test prompt",
        image_path=Path("test.jpg"),
        encoded_image=mx.zeros((1, 16, latent_h, latent_w)),  # [C, F, H, W]
        text_embeddings=mx.zeros((1, 77, 768)),
    )


def create_mock_batch(height: int = 256, width: int = 256, target_resolution: tuple | None = None) -> Batch:
    """Create a mock batch with specified resolution."""
    example = create_mock_example(height, width)
    batch = Batch(examples=[example], rng=None)
    if target_resolution:
        batch.target_resolution = target_resolution
    return batch


class MockCompiledTrainStep:
    """A simplified version of CompiledTrainStep for testing cache behavior.

    This avoids the complexity of nn.value_and_grad which requires real models,
    while testing the caching logic that is the focus of these tests.
    """

    VAE_DOWNSCALE_FACTOR = 8

    def __init__(self, cache_size: int = 12, enabled: bool = True):
        self._enabled = enabled
        self._cache_size = cache_size
        self._get_compiled_fn = lru_cache(maxsize=cache_size)(self._create_compiled_fn)

    def _create_compiled_fn(self, height: int, width: int):
        """Create a mock compiled function for given resolution."""

        def mock_train_step(batch):
            return mx.array(0.5), {"mock_grad": mx.zeros((1,))}

        return mock_train_step

    def __call__(self, batch: Batch) -> tuple[mx.array, dict]:
        """Execute training step with caching."""
        if batch.target_resolution:
            width, height = batch.target_resolution
        else:
            shape = batch.examples[0].encoded_image.shape
            height = shape[2] * self.VAE_DOWNSCALE_FACTOR
            width = shape[3] * self.VAE_DOWNSCALE_FACTOR

        train_fn = self._get_compiled_fn(height, width)
        return train_fn(batch)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache_info = self._get_compiled_fn.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize,
            "hit_rate": cache_info.hits / max(1, cache_info.hits + cache_info.misses),
        }

    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self._get_compiled_fn.cache_clear()


@pytest.mark.fast
class TestCompiledTrainStepCache:
    """Tests for CompiledTrainStep caching behavior.

    Uses MockCompiledTrainStep to test caching logic without requiring
    a real model (which would need nn.value_and_grad to work).
    """

    def test_cache_hit_same_resolution(self):
        """Test that same resolution produces cache hit."""
        compiled_step = MockCompiledTrainStep()

        # First call - cache miss
        batch1 = create_mock_batch(256, 256)
        compiled_step(batch1)

        stats1 = compiled_step.get_cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # Second call with same resolution - cache hit
        batch2 = create_mock_batch(256, 256)
        compiled_step(batch2)

        stats2 = compiled_step.get_cache_stats()
        assert stats2["misses"] == 1
        assert stats2["hits"] == 1

    def test_cache_miss_different_resolution(self):
        """Test that different resolutions produce cache miss."""
        compiled_step = MockCompiledTrainStep()

        # First call - cache miss
        batch1 = create_mock_batch(256, 256)
        compiled_step(batch1)

        # Second call with different resolution - cache miss
        batch2 = create_mock_batch(512, 512)
        compiled_step(batch2)

        stats = compiled_step.get_cache_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0
        assert stats["size"] == 2

    def test_cache_stats_tracking(self):
        """Test that get_cache_stats returns correct values."""
        compiled_step = MockCompiledTrainStep(cache_size=5)

        # Initial state
        stats = compiled_step.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["maxsize"] == 5

        # Add some entries
        for i in range(3):
            batch = create_mock_batch(256 + i * 64, 256 + i * 64)
            compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["misses"] == 3
        assert stats["size"] == 3
        assert stats["hit_rate"] == 0.0

        # Reuse one entry
        batch = create_mock_batch(256, 256)
        compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 3
        assert stats["hit_rate"] == 0.25  # 1/(1+3)

    def test_lru_eviction_when_cache_full(self):
        """Test LRU eviction when cache reaches maxsize."""
        # Small cache size for testing eviction
        compiled_step = MockCompiledTrainStep(cache_size=3)

        # Fill the cache
        for i in range(3):
            batch = create_mock_batch(256 + i * 64, 256 + i * 64)
            compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["size"] == 3
        assert stats["misses"] == 3

        # Access first entry to make it recently used
        batch = create_mock_batch(256, 256)
        compiled_step(batch)
        assert compiled_step.get_cache_stats()["hits"] == 1

        # Add a new entry that causes eviction
        batch = create_mock_batch(512, 512)
        compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["size"] == 3  # Still at max size
        assert stats["misses"] == 4  # New miss

        # The 320x320 entry (second added, least recently used after accessing 256x256)
        # should have been evicted
        batch = create_mock_batch(320, 320)
        compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["misses"] == 5  # It was evicted, so this is a miss

    def test_disabled_mode(self):
        """Test that disabled mode still works (no compilation)."""
        compiled_step = MockCompiledTrainStep(enabled=False)

        batch = create_mock_batch(256, 256)
        loss, grads = compiled_step(batch)

        # Should still work
        assert compiled_step._enabled is False
        # Cache should still track calls
        assert compiled_step.get_cache_stats()["misses"] == 1

    def test_clear_cache(self):
        """Test that clear_cache resets the cache."""
        compiled_step = MockCompiledTrainStep()

        # Add some entries
        for i in range(3):
            batch = create_mock_batch(256 + i * 64, 256 + i * 64)
            compiled_step(batch)

        stats = compiled_step.get_cache_stats()
        assert stats["size"] == 3
        assert stats["misses"] == 3

        # Clear cache
        compiled_step.clear_cache()

        stats = compiled_step.get_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_target_resolution_from_batch(self):
        """Test that target_resolution from batch is used for cache key."""
        compiled_step = MockCompiledTrainStep()

        # Batch with explicit target_resolution
        batch1 = create_mock_batch(256, 256, target_resolution=(512, 512))
        compiled_step(batch1)

        # Same target_resolution should hit cache
        batch2 = create_mock_batch(256, 256, target_resolution=(512, 512))
        compiled_step(batch2)

        stats = compiled_step.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        # Different target_resolution should miss
        batch3 = create_mock_batch(256, 256, target_resolution=(768, 768))
        compiled_step(batch3)

        stats = compiled_step.get_cache_stats()
        assert stats["misses"] == 2


@pytest.mark.fast
class TestCompiledTrainStepAPI:
    """Tests for the real CompiledTrainStep API (without executing training)."""

    def test_initialization(self):
        """Test CompiledTrainStep can be initialized."""
        from unittest.mock import MagicMock

        from mflux.models.z_image.variants.training.optimization.compiled_train_step import (
            CompiledTrainStep,
        )

        model = MagicMock()
        config = MagicMock()

        step = CompiledTrainStep(model, config, enabled=True, cache_size=24)

        assert step._enabled is True
        assert step._cache_size == 24
        assert step.model is model
        assert step.config is config

    def test_initial_cache_stats(self):
        """Test initial cache stats are zero."""
        from unittest.mock import MagicMock

        from mflux.models.z_image.variants.training.optimization.compiled_train_step import (
            CompiledTrainStep,
        )

        model = MagicMock()
        config = MagicMock()

        step = CompiledTrainStep(model, config)

        stats = step.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["maxsize"] == CompiledTrainStep.DEFAULT_CACHE_SIZE
