"""Stress tests for Qwen3-VL optimization stability.

These tests verify that optimizations don't introduce:
- Memory leaks over many iterations
- Timing inconsistencies
- Crashes or hangs under load

Usage:
    # Run all stress tests (short)
    python -m pytest tests/test_optimization_stress.py -v

    # Run extended stress test
    python tests/test_optimization_stress.py --extended

    # Run memory profiling
    python tests/test_optimization_stress.py --memory-profile
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from PIL import Image


class StressTestResult(NamedTuple):
    """Results from a stress test run."""

    iterations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_growth_pct: float


def create_test_image(size: int = 224) -> Image.Image:
    """Create a test image for stress testing."""
    try:
        import numpy as np
        from PIL import Image

        # Create random image
        img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    except ImportError:
        pytest.skip("PIL not available")


class TestOptimizationStress:
    """Stress tests for optimization stability."""

    @pytest.fixture
    def embedding_scorer(self):
        """Create MLX Qwen Embedding scorer."""
        try:
            from mflux.models.qwen.variants.training.validation.clip_scorer import (
                MLXQwenEmbeddingScorer,
            )

            scorer = MLXQwenEmbeddingScorer()
            yield scorer
            scorer.clear()
        except ImportError:
            pytest.skip("MLX Qwen Embedding not available")

    @pytest.fixture
    def reranker_scorer(self):
        """Create MLX Qwen Reranker scorer."""
        try:
            from mflux.models.qwen.variants.training.validation.clip_scorer import (
                MLXQwenRerankerScorer,
            )

            scorer = MLXQwenRerankerScorer()
            yield scorer
            scorer.clear()
        except ImportError:
            pytest.skip("MLX Qwen Reranker not available")

    def test_timing_consistency(self, embedding_scorer, iterations: int = 20):
        """Timing should be consistent (low variance).

        Checks that performance is stable across multiple runs.
        High variance indicates potential issues with compilation,
        memory management, or resource contention.
        """
        image = create_test_image()
        prompt = "a test image for timing measurement"

        # Warmup
        for _ in range(3):
            _ = embedding_scorer.compute_score(image, prompt)

        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = embedding_scorer.compute_score(image, prompt)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        cv = std_time / mean_time if mean_time > 0 else 0  # Coefficient of variation

        print(f"\nTiming results ({iterations} iterations):")
        print(f"  Mean: {mean_time:.1f}ms")
        print(f"  Std:  {std_time:.1f}ms")
        print(f"  CV:   {cv:.2%}")
        print(f"  Min:  {min(times):.1f}ms")
        print(f"  Max:  {max(times):.1f}ms")

        # CV should be < 15% for stable performance
        assert cv < 0.15, f"High timing variance: CV={cv:.2%} (std={std_time:.1f}ms, mean={mean_time:.1f}ms)"

    def test_timing_consistency_reranker(self, reranker_scorer, iterations: int = 20):
        """Timing consistency for reranker."""
        image = create_test_image()
        prompt = "a test image for timing measurement"

        # Warmup
        for _ in range(3):
            _ = reranker_scorer.compute_score(image, prompt)

        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = reranker_scorer.compute_score(image, prompt)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        cv = std_time / mean_time if mean_time > 0 else 0

        print(f"\nReranker timing results ({iterations} iterations):")
        print(f"  Mean: {mean_time:.1f}ms")
        print(f"  Std:  {std_time:.1f}ms")
        print(f"  CV:   {cv:.2%}")

        assert cv < 0.15, f"High timing variance: CV={cv:.2%}"

    def test_memory_stability_short(self, embedding_scorer, iterations: int = 50):
        """No memory leaks over moderate iterations (quick test)."""
        try:
            import tracemalloc
        except ImportError:
            pytest.skip("tracemalloc not available")

        image = create_test_image()
        prompt = "a test image for memory testing"

        # Warmup
        for _ in range(3):
            _ = embedding_scorer.compute_score(image, prompt)

        # Start memory tracking
        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        for i in range(iterations):
            _ = embedding_scorer.compute_score(image, prompt)

            # Check periodically
            if (i + 1) % 10 == 0:
                current_memory = tracemalloc.get_traced_memory()[0]
                growth = (current_memory - initial_memory) / initial_memory if initial_memory > 0 else 0

                # Memory shouldn't grow more than 50% over baseline
                assert growth < 0.5, f"Memory grew {growth * 100:.1f}% after {i + 1} iterations"

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        total_growth = (final_memory - initial_memory) / initial_memory if initial_memory > 0 else 0
        print(f"\nMemory results ({iterations} iterations):")
        print(f"  Initial: {initial_memory / 1024 / 1024:.1f}MB")
        print(f"  Final:   {final_memory / 1024 / 1024:.1f}MB")
        print(f"  Growth:  {total_growth * 100:.1f}%")

    def test_no_crashes_varied_inputs(self, embedding_scorer):
        """Model handles varied inputs without crashing."""
        prompts = [
            "a simple test",
            "a very long prompt " * 50,  # Long prompt
            "",  # Empty prompt
            "unicode: 日本語 中文 한국어",  # Unicode
            "special chars: @#$%^&*()",  # Special characters
            "numbers: 12345 67890",  # Numbers
        ]

        # Test each prompt type - all should produce valid scores
        for prompt in prompts:
            image = create_test_image()
            score = embedding_scorer.compute_score(image, prompt)

            # Should return valid score
            assert isinstance(score, (int, float)), f"Invalid score type for prompt: {prompt[:30]}"
            assert 0 <= score <= 100, f"Score {score} out of range for prompt: {prompt[:30]}"

    def test_batch_processing_stability(self, embedding_scorer):
        """Batch processing works correctly."""
        images = [create_test_image() for _ in range(5)]
        prompts = [f"test prompt {i}" for i in range(5)]

        scores = embedding_scorer.compute_scores_batch(images, prompts)

        assert len(scores) == 5
        for i, score in enumerate(scores):
            assert isinstance(score, float), f"Score {i} is not float"
            assert 0 <= score <= 100, f"Score {i} ({score}) out of range"


def run_extended_stress_test(iterations: int = 500):
    """Run extended stress test with detailed reporting."""
    from mflux.models.qwen.variants.training.validation.clip_scorer import (
        MLXQwenEmbeddingScorer,
        MLXQwenRerankerScorer,
    )

    print(f"\n{'=' * 60}")
    print(f"Extended Stress Test ({iterations} iterations)")
    print(f"{'=' * 60}")

    # Test embedding scorer
    print("\nTesting MLX Qwen Embedding...")
    embedding_scorer = MLXQwenEmbeddingScorer()

    try:
        result = _run_stress_test(embedding_scorer, iterations)
        print(f"  Mean time:       {result.mean_time_ms:.1f}ms")
        print(f"  Std time:        {result.std_time_ms:.1f}ms")
        print(f"  Range:           {result.min_time_ms:.1f}-{result.max_time_ms:.1f}ms")
        print(f"  Memory growth:   {result.memory_growth_pct:.1f}%")

        # Validate results
        cv = result.std_time_ms / result.mean_time_ms if result.mean_time_ms > 0 else 0
        if cv < 0.10:
            print("  Status:          PASS (stable timing)")
        elif cv < 0.15:
            print("  Status:          WARN (moderate variance)")
        else:
            print("  Status:          FAIL (high variance)")

    finally:
        embedding_scorer.clear()

    # Test reranker scorer
    print("\nTesting MLX Qwen Reranker...")
    reranker_scorer = MLXQwenRerankerScorer()

    try:
        result = _run_stress_test(reranker_scorer, iterations)
        print(f"  Mean time:       {result.mean_time_ms:.1f}ms")
        print(f"  Std time:        {result.std_time_ms:.1f}ms")
        print(f"  Range:           {result.min_time_ms:.1f}-{result.max_time_ms:.1f}ms")
        print(f"  Memory growth:   {result.memory_growth_pct:.1f}%")

        cv = result.std_time_ms / result.mean_time_ms if result.mean_time_ms > 0 else 0
        if cv < 0.10:
            print("  Status:          PASS (stable timing)")
        elif cv < 0.15:
            print("  Status:          WARN (moderate variance)")
        else:
            print("  Status:          FAIL (high variance)")

    finally:
        reranker_scorer.clear()

    print(f"\n{'=' * 60}")
    print("Extended stress test complete")


def _run_stress_test(scorer, iterations: int) -> StressTestResult:
    """Run stress test on a scorer."""
    import tracemalloc

    image = create_test_image()
    prompt = "a test image for stress testing"

    # Warmup
    for _ in range(5):
        _ = scorer.compute_score(image, prompt)

    # Start memory tracking
    gc.collect()
    tracemalloc.start()
    initial_memory = tracemalloc.get_traced_memory()[0]

    # Run iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = scorer.compute_score(image, prompt)
        times.append((time.perf_counter() - start) * 1000)

    final_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    memory_growth = (final_memory - initial_memory) / initial_memory * 100 if initial_memory > 0 else 0

    return StressTestResult(
        iterations=iterations,
        mean_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        memory_growth_pct=memory_growth,
    )


def run_memory_profile(iterations: int = 100):
    """Run detailed memory profiling."""
    try:
        import tracemalloc
    except ImportError:
        print("tracemalloc not available")
        return

    from mflux.models.qwen.variants.training.validation.clip_scorer import (
        MLXQwenEmbeddingScorer,
    )

    print(f"\n{'=' * 60}")
    print(f"Memory Profile ({iterations} iterations)")
    print(f"{'=' * 60}")

    scorer = MLXQwenEmbeddingScorer()
    image = create_test_image()
    prompt = "test prompt for memory profiling"

    # Warmup
    for _ in range(5):
        _ = scorer.compute_score(image, prompt)

    gc.collect()
    tracemalloc.start()

    memory_samples = []
    for i in range(iterations):
        _ = scorer.compute_score(image, prompt)

        if (i + 1) % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append((i + 1, current / 1024 / 1024, peak / 1024 / 1024))
            print(f"  Iter {i + 1:4d}: current={current / 1024 / 1024:.1f}MB, peak={peak / 1024 / 1024:.1f}MB")

    tracemalloc.stop()
    scorer.clear()

    print(f"\n{'=' * 60}")
    print("Memory profile complete")


if __name__ == "__main__":
    if "--extended" in sys.argv:
        iterations = 500
        if "--iterations" in sys.argv:
            idx = sys.argv.index("--iterations")
            if idx + 1 < len(sys.argv):
                iterations = int(sys.argv[idx + 1])
        run_extended_stress_test(iterations)
    elif "--memory-profile" in sys.argv:
        iterations = 100
        if "--iterations" in sys.argv:
            idx = sys.argv.index("--iterations")
            if idx + 1 < len(sys.argv):
                iterations = int(sys.argv[idx + 1])
        run_memory_profile(iterations)
    else:
        pytest.main([__file__, "-v"])
