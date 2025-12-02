"""Pytest fixtures for Z-Image tests with basic memory management.

This module provides cleanup fixtures to prevent memory accumulation
across test classes when testing large models like S3DiT and Qwen3Encoder.
Make no mistake, though: this will hammer systems with less than 48GB RAM!
"""

import gc

import mlx.core as mx
import pytest


def _clear_mlx_cache():
    """Clear MLX cache using current API."""
    # Use new API if available, fall back to deprecated
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif mx.metal.is_available():
        mx.metal.clear_cache()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up MLX memory after each test.

    This runs automatically after every test to prevent memory accumulation.
    """
    yield
    # Force garbage collection
    gc.collect()
    # Clear MLX cache
    _clear_mlx_cache()


@pytest.fixture(scope="class", autouse=True)
def cleanup_after_class():
    """Clean up MLX memory after each test class.

    More aggressive cleanup after all tests in a class complete.
    """
    yield
    # Force multiple GC passes to clean up circular references
    gc.collect()
    gc.collect()
    gc.collect()
    # Clear MLX cache
    _clear_mlx_cache()


@pytest.fixture
def memory_tracker():
    """Fixture to track memory usage during a test.

    Usage:
        def test_something(memory_tracker):
            memory_tracker.start()
            # ... do stuff ...
            peak_mb = memory_tracker.get_peak_mb()
            assert peak_mb < 1000, f"Used {peak_mb}MB, expected < 1000MB"
    """

    class MemoryTracker:
        def start(self):
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            elif mx.metal.is_available():
                mx.metal.reset_peak_memory()

        def get_active_mb(self):
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            elif mx.metal.is_available():
                return mx.metal.get_active_memory() / (1024 * 1024)
            return 0

        def get_peak_mb(self):
            if hasattr(mx, "get_peak_memory"):
                return mx.get_peak_memory() / (1024 * 1024)
            elif mx.metal.is_available():
                return mx.metal.get_peak_memory() / (1024 * 1024)
            return 0

        def get_cache_mb(self):
            if hasattr(mx, "get_cache_memory"):
                return mx.get_cache_memory() / (1024 * 1024)
            elif mx.metal.is_available():
                return mx.metal.get_cache_memory() / (1024 * 1024)
            return 0

    return MemoryTracker()
