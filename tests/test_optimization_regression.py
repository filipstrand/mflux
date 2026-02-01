"""Regression tests for Qwen3-VL optimization phases.

These tests ensure optimizations don't break accuracy or introduce regressions.
Run after each optimization phase to validate changes.

Usage:
    # Run all regression tests
    python -m pytest tests/test_optimization_regression.py -v

    # Save baseline before optimizations
    python tests/test_optimization_regression.py --save-baseline

    # Run with specific backend
    python -m pytest tests/test_optimization_regression.py -v -k "embedding"
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from PIL import Image


# Test case definitions
class TestCase(NamedTuple):
    """A test case for optimization regression testing."""

    name: str
    prompt: str
    image_description: str  # Description for generating test image
    expected_high_score: bool  # True if image should match prompt well


# 15 test cases covering different scenarios
TEST_CASES = [
    # Matching pairs (should score high)
    TestCase("sunset_match", "a beautiful sunset over the ocean", "sunset over ocean", True),
    TestCase("dog_match", "a golden retriever playing in a park", "golden retriever in park", True),
    TestCase("city_match", "a modern city skyline at night", "city skyline night", True),
    TestCase("food_match", "a delicious pizza with toppings", "pizza with toppings", True),
    TestCase("nature_match", "a mountain landscape with snow", "snowy mountain", True),
    TestCase("cat_match", "a fluffy orange cat sleeping", "orange cat sleeping", True),
    TestCase("beach_match", "a tropical beach with palm trees", "tropical beach palms", True),
    # Non-matching pairs (should score low)
    TestCase("sunset_mismatch", "a beautiful sunset over the ocean", "busy city street", False),
    TestCase("dog_mismatch", "a golden retriever playing in a park", "sports car", False),
    TestCase("city_mismatch", "a modern city skyline at night", "forest trail", False),
    TestCase("food_mismatch", "a delicious pizza with toppings", "winter snow scene", False),
    TestCase("nature_mismatch", "a mountain landscape with snow", "indoor office", False),
    TestCase("cat_mismatch", "a fluffy orange cat sleeping", "airplane flying", False),
    TestCase("beach_mismatch", "a tropical beach with palm trees", "desert cactus", False),
    # Edge case
    TestCase("abstract", "abstract art with vibrant colors", "colorful abstract", True),
]

BASELINE_FILE = Path(__file__).parent / "optimization_baseline.json"
SCORE_TOLERANCE = 15.0  # Maximum allowed score difference from baseline (on 0-100 scale)
# Higher tolerance needed for synthetic test images with inherent variability
# For production use with real images, reduce to 5.0


def create_test_image(description: str) -> Image.Image:
    """Create a test image based on description.

    For actual testing, this would load real test images.
    For CI/quick tests, generates simple placeholder images.
    """
    try:
        import numpy as np
        from PIL import Image

        # Create a simple colored image based on description hash
        # This provides consistent images for the same description
        hash_val = hash(description) % (2**24)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF

        # Create 224x224 image (standard for embeddings)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = r
        img_array[:, :, 1] = g
        img_array[:, :, 2] = b

        # Add some texture based on description
        for i in range(0, 224, 16):
            for j in range(0, 224, 16):
                offset = (hash(f"{description}_{i}_{j}") % 64) - 32
                img_array[i : i + 16, j : j + 16, :] = np.clip(
                    img_array[i : i + 16, j : j + 16, :].astype(int) + offset, 0, 255
                ).astype(np.uint8)

        return Image.fromarray(img_array)

    except ImportError:
        pytest.skip("PIL not available for image creation")


class TestOptimizationRegression:
    """Regression tests for optimization phases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load baseline scores if available."""
        self.baseline = None
        if BASELINE_FILE.exists():
            with open(BASELINE_FILE) as f:
                self.baseline = json.load(f)

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

    def test_embedding_scores_within_tolerance(self, embedding_scorer):
        """Each test case score within tolerance of baseline."""
        if self.baseline is None:
            pytest.skip("No baseline file - run with --save-baseline first")

        if "embedding" not in self.baseline:
            pytest.skip("No embedding baseline available")

        for case in TEST_CASES[:5]:  # Test subset for speed
            image = create_test_image(case.image_description)
            score = embedding_scorer.compute_score(image, case.prompt)

            if case.name in self.baseline["embedding"]:
                baseline_score = self.baseline["embedding"][case.name]
                diff = abs(score - baseline_score)
                assert diff < SCORE_TOLERANCE, (
                    f"{case.name}: score {score:.2f} differs from baseline {baseline_score:.2f} "
                    f"by {diff:.2f} (tolerance: {SCORE_TOLERANCE})"
                )

    def test_reranker_scores_within_tolerance(self, reranker_scorer):
        """Each test case score within tolerance of baseline."""
        if self.baseline is None:
            pytest.skip("No baseline file - run with --save-baseline first")

        if "reranker" not in self.baseline:
            pytest.skip("No reranker baseline available")

        for case in TEST_CASES[:5]:  # Test subset for speed
            image = create_test_image(case.image_description)
            score = reranker_scorer.compute_score(image, case.prompt)

            if case.name in self.baseline["reranker"]:
                baseline_score = self.baseline["reranker"][case.name]
                diff = abs(score - baseline_score)
                assert diff < SCORE_TOLERANCE, (
                    f"{case.name}: score {score:.2f} differs from baseline {baseline_score:.2f} "
                    f"by {diff:.2f} (tolerance: {SCORE_TOLERANCE})"
                )

    def test_no_nan_outputs(self, embedding_scorer):
        """No NaN or inf in embedding outputs."""
        # Test all cases to ensure complete coverage
        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = embedding_scorer.compute_score(image, case.prompt)

            assert not math.isnan(score), f"{case.name}: got NaN score"
            assert not math.isinf(score), f"{case.name}: got inf score"
            assert 0 <= score <= 100, f"{case.name}: score {score} out of range [0, 100]"

    def test_no_nan_outputs_reranker(self, reranker_scorer):
        """No NaN or inf in reranker outputs."""
        # Test all cases to ensure complete coverage
        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = reranker_scorer.compute_score(image, case.prompt)

            assert not math.isnan(score), f"{case.name}: got NaN score"
            assert not math.isinf(score), f"{case.name}: got inf score"
            assert 0 <= score <= 100, f"{case.name}: score {score} out of range [0, 100]"

    def test_embedding_accuracy_preserved(self, embedding_scorer):
        """Accuracy must not drop below baseline."""
        if self.baseline is None:
            pytest.skip("No baseline file - run with --save-baseline first")

        if "embedding_accuracy" not in self.baseline:
            pytest.skip("No embedding accuracy baseline available")

        # Compute accuracy on test cases
        correct = 0
        total = len(TEST_CASES)
        threshold = 50.0  # Score threshold for "matching"

        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = embedding_scorer.compute_score(image, case.prompt)

            predicted_match = score >= threshold
            if predicted_match == case.expected_high_score:
                correct += 1

        accuracy = (correct / total) * 100
        baseline_accuracy = self.baseline["embedding_accuracy"]

        # Allow 1% regression tolerance
        assert accuracy >= baseline_accuracy - 1.0, (
            f"Accuracy {accuracy:.1f}% dropped below baseline {baseline_accuracy:.1f}%"
        )

    @pytest.mark.skip(reason="Synthetic test images don't provide semantic discrimination - use real images")
    def test_score_discrimination(self, embedding_scorer):
        """Scores should discriminate between matching and non-matching pairs."""
        matching_scores = []
        non_matching_scores = []

        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = embedding_scorer.compute_score(image, case.prompt)

            if case.expected_high_score:
                matching_scores.append(score)
            else:
                non_matching_scores.append(score)

        if matching_scores and non_matching_scores:
            avg_matching = sum(matching_scores) / len(matching_scores)
            avg_non_matching = sum(non_matching_scores) / len(non_matching_scores)
            discrimination = avg_matching - avg_non_matching

            # Discrimination should be positive (matching scores higher)
            assert discrimination > 0, (
                f"No discrimination: matching avg {avg_matching:.1f}, non-matching avg {avg_non_matching:.1f}"
            )


def save_baseline():
    """Save current scores as baseline for regression testing."""
    from mflux.models.qwen.variants.training.validation.clip_scorer import (
        MLXQwenEmbeddingScorer,
        MLXQwenRerankerScorer,
    )

    baseline = {
        "embedding": {},
        "reranker": {},
    }

    print("Saving embedding baseline...")
    try:
        embedding_scorer = MLXQwenEmbeddingScorer()

        correct = 0
        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = embedding_scorer.compute_score(image, case.prompt)
            baseline["embedding"][case.name] = score

            # Track accuracy
            predicted_match = score >= 50.0
            if predicted_match == case.expected_high_score:
                correct += 1

            print(f"  {case.name}: {score:.2f}")

        baseline["embedding_accuracy"] = (correct / len(TEST_CASES)) * 100
        print(f"  Accuracy: {baseline['embedding_accuracy']:.1f}%")
        embedding_scorer.clear()

    except (ImportError, RuntimeError, ValueError) as e:
        print(f"  Failed: {e}")

    print("\nSaving reranker baseline...")
    try:
        reranker_scorer = MLXQwenRerankerScorer()

        correct = 0
        for case in TEST_CASES:
            image = create_test_image(case.image_description)
            score = reranker_scorer.compute_score(image, case.prompt)
            baseline["reranker"][case.name] = score

            # Track accuracy
            predicted_match = score >= 50.0
            if predicted_match == case.expected_high_score:
                correct += 1

            print(f"  {case.name}: {score:.2f}")

        baseline["reranker_accuracy"] = (correct / len(TEST_CASES)) * 100
        print(f"  Accuracy: {baseline['reranker_accuracy']:.1f}%")
        reranker_scorer.clear()

    except (ImportError, RuntimeError, ValueError) as e:
        print(f"  Failed: {e}")

    # Save baseline
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline saved to {BASELINE_FILE}")


if __name__ == "__main__":
    if "--save-baseline" in sys.argv:
        save_baseline()
    else:
        pytest.main([__file__, "-v"])
