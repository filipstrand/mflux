"""Tests for Z-Image prompt caching functionality.

Prompt caching reduces redundant text encoder forward passes
for repeated prompts during batch generation.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mflux.models.z_image.optimization.prompt_cache import (
    ZImagePromptCache,
    create_prompt_cache,
)


class TestZImagePromptCache:
    """Tests for ZImagePromptCache class."""

    def test_init_default(self):
        """Test default initialization."""
        cache = ZImagePromptCache()

        assert len(cache) == 0
        assert cache._max_items == 100

    def test_init_custom_size(self):
        """Test custom cache size."""
        cache = ZImagePromptCache(max_items=50)

        assert cache._max_items == 50

    def test_init_invalid_size(self):
        """Test that size < 1 raises error."""
        with pytest.raises(ValueError, match="max_items must be >= 1"):
            ZImagePromptCache(max_items=0)

    def test_hash_prompt(self):
        """Test prompt hashing."""
        hash1 = ZImagePromptCache._hash_prompt("hello world")
        hash2 = ZImagePromptCache._hash_prompt("hello world")
        hash3 = ZImagePromptCache._hash_prompt("different prompt")

        # Same prompt = same hash
        assert hash1 == hash2

        # Different prompt = different hash
        assert hash1 != hash3

        # Hash is 16 characters
        assert len(hash1) == 16

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = ZImagePromptCache()
        embedding = mx.ones((1, 512))

        cache.put("test prompt", embedding)
        retrieved = cache.get("test prompt")

        assert retrieved is not None
        assert mx.allclose(retrieved, embedding)

    def test_get_miss(self):
        """Test cache miss returns None."""
        cache = ZImagePromptCache()

        result = cache.get("nonexistent prompt")

        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ZImagePromptCache(max_items=3)

        # Fill cache
        for i in range(3):
            cache.put(f"prompt_{i}", mx.array([i]))

        # Access prompt_0 to make it recently used
        cache.get("prompt_0")

        # Add new item, should evict prompt_1 (LRU)
        cache.put("prompt_3", mx.array([3]))

        # prompt_1 should be evicted
        assert cache.get("prompt_1") is None
        # prompt_0 should still exist (was accessed)
        assert cache.get("prompt_0") is not None
        # prompt_2 and prompt_3 should exist
        assert cache.get("prompt_2") is not None
        assert cache.get("prompt_3") is not None

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = ZImagePromptCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Put and hit
        cache.put("test", mx.ones((1,)))
        cache.get("test")
        stats = cache.get_stats()
        assert stats["hits"] == 1

    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        cache = ZImagePromptCache()

        # Add some entries and access them
        cache.put("a", mx.ones((1,)))
        cache.put("b", mx.ones((1,)))

        # 2 hits, 2 misses = 50% hit rate
        cache.get("a")  # hit
        cache.get("b")  # hit
        cache.get("c")  # miss
        cache.get("d")  # miss

        stats = cache.get_stats()
        assert stats["hit_rate_percent"] == 50.0

    def test_clear(self):
        """Test clearing cache."""
        cache = ZImagePromptCache()

        cache.put("a", mx.ones((1,)))
        cache.put("b", mx.ones((1,)))

        count = cache.clear()

        assert count == 2
        assert len(cache) == 0

    def test_len(self):
        """Test __len__ method."""
        cache = ZImagePromptCache()

        assert len(cache) == 0

        cache.put("a", mx.ones((1,)))
        assert len(cache) == 1

        cache.put("b", mx.ones((1,)))
        assert len(cache) == 2


class TestGetOrCompute:
    """Tests for get_or_compute method."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def mock_encoder(self):
        """Create mock text encoder."""
        return MagicMock()

    def test_computes_on_miss(self, mock_tokenizer, mock_encoder):
        """Test that embedding is computed on cache miss."""
        cache = ZImagePromptCache()
        computed_embedding = mx.ones((1, 512))

        def encoder_fn(prompt, tokenizer, encoder):
            return computed_embedding

        result = cache.get_or_compute(
            prompt="test prompt",
            tokenizer=mock_tokenizer,
            text_encoder=mock_encoder,
            encoder_fn=encoder_fn,
        )

        assert mx.allclose(result, computed_embedding)
        assert len(cache) == 1

    def test_returns_cached_on_hit(self, mock_tokenizer, mock_encoder):
        """Test that cached embedding is returned on hit."""
        cache = ZImagePromptCache()
        cached_embedding = mx.ones((1, 512))

        # Pre-populate cache
        cache.put("test prompt", cached_embedding)

        call_count = [0]

        def encoder_fn(prompt, tokenizer, encoder):
            call_count[0] += 1
            return mx.zeros((1, 512))  # Different from cached

        result = cache.get_or_compute(
            prompt="test prompt",
            tokenizer=mock_tokenizer,
            text_encoder=mock_encoder,
            encoder_fn=encoder_fn,
        )

        # Should return cached, not computed
        assert mx.allclose(result, cached_embedding)
        # Encoder should not have been called
        assert call_count[0] == 0

    def test_caches_computed_result(self, mock_tokenizer, mock_encoder):
        """Test that computed result is cached."""
        cache = ZImagePromptCache()

        call_count = [0]

        def encoder_fn(prompt, tokenizer, encoder):
            call_count[0] += 1
            return mx.array([call_count[0]])

        # First call - computes
        result1 = cache.get_or_compute(
            prompt="test",
            tokenizer=mock_tokenizer,
            text_encoder=mock_encoder,
            encoder_fn=encoder_fn,
        )

        # Second call - should use cache
        result2 = cache.get_or_compute(
            prompt="test",
            tokenizer=mock_tokenizer,
            text_encoder=mock_encoder,
            encoder_fn=encoder_fn,
        )

        # Both should return same value
        assert mx.allclose(result1, result2)
        # Encoder called only once
        assert call_count[0] == 1


class TestCreatePromptCache:
    """Tests for factory function."""

    def test_create_enabled(self):
        """Test creating enabled cache."""
        cache = create_prompt_cache(enabled=True)

        assert isinstance(cache, ZImagePromptCache)

    def test_create_disabled(self):
        """Test that disabled returns None."""
        cache = create_prompt_cache(enabled=False)

        assert cache is None

    def test_create_custom_size(self):
        """Test creating with custom size."""
        cache = create_prompt_cache(enabled=True, max_items=50)

        assert cache._max_items == 50


class TestPromptCacheIntegration:
    """Integration tests for prompt cache usage patterns."""

    def test_batch_generation_pattern(self):
        """Test cache usage in batch generation scenario."""
        cache = ZImagePromptCache(max_items=10)

        # Simulate batch generation with repeated prompts
        prompts = ["sunset"] * 4 + ["mountain"] * 4 + ["sunset"] * 4

        encoder_calls = [0]

        def mock_encode(prompt, tok, enc):
            encoder_calls[0] += 1
            return mx.array([hash(prompt)])

        for prompt in prompts:
            cache.get_or_compute(
                prompt=prompt,
                tokenizer=MagicMock(),
                text_encoder=MagicMock(),
                encoder_fn=mock_encode,
            )

        # Should only call encoder twice (once per unique prompt)
        assert encoder_calls[0] == 2

        stats = cache.get_stats()
        # 12 total, 2 misses = 10 hits
        assert stats["hits"] == 10
        assert stats["misses"] == 2
