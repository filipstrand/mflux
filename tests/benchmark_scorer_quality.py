"""Benchmark Qwen3-VL vs CLIP scoring quality.

Compares scoring backends for image-text alignment:
- CLIP ViT-B/32: Fast (61ms), reliable, ~600MB
- Qwen3-VL-Reranker-2B: Slower (2.5s), good discrimination, ~4GB
- Qwen3-VL-Embedding-2B: Slower (2.7s), calibration issues, ~4GB
- MLX Qwen Embedding: Native MLX inference (target: 200-400ms), ~4GB
- MLX Qwen Reranker: Native MLX inference (target: 200-400ms), ~4GB

Results (January 2025):
    | Backend             | Accuracy | Discrimination | Speed    |
    |---------------------|----------|----------------|----------|
    | CLIP ViT-B/32       | 100%     | 34.9           | 61ms     |
    | Qwen3-VL-Reranker-2B| 100%     | 57.4           | 2509ms   |
    | Qwen3-VL-Embedding-2B| 73%     | 59.4           | 2691ms   |

Recommendations:
    - Speed-critical: Use CLIP (40x faster)
    - Quality-critical: Use Qwen3-VL-Reranker (65% better discrimination)
    - MLX backends: Use for Apple Silicon native inference (5-10x faster than PyTorch)

Requirements:
    pip install qwen-vl-utils>=0.0.14
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git (in mlx-mflux root)

Usage:
    python tests/benchmark_scorer_quality.py
    python tests/benchmark_scorer_quality.py --backends clip  # Run single backend
    python tests/benchmark_scorer_quality.py --backends mlx-qwen-embedding  # Run MLX backend
    python tests/benchmark_scorer_quality.py --verbose  # Show detailed output
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from PIL import Image

# Add src to path for local development - must be before imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add Qwen3-VL-Embedding to path if available
QWEN_EMBEDDING_PATH = Path(__file__).parent.parent.parent / "Qwen3-VL-Embedding" / "src"
if QWEN_EMBEDDING_PATH.exists():
    sys.path.insert(0, str(QWEN_EMBEDDING_PATH))

# Import after path setup (noqa: E402 required for conditional path setup)
from mflux.models.qwen.variants.training.validation.clip_scorer import QwenCLIPScorer  # noqa: E402


class ImageTextScorer(Protocol):
    """Protocol for image-text scorers."""

    def compute_score(self, image: Image.Image, prompt: str) -> float:
        """Compute score between image and prompt (0-100 scale)."""
        ...

    def clear(self) -> None:
        """Release model resources."""
        ...


class OfficialQwenEmbeddingScorer:
    """Wrapper for official Qwen3VLEmbedder using cosine similarity."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            from models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            raise ImportError(
                "Qwen3-VL-Embedding not found. Clone it to mlx-mflux root:\n"
                "  git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git"
            )

        self._model = Qwen3VLEmbedder(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            torch_dtype="float32",  # MPS compatibility
        )

    def compute_score(self, image: Image.Image, prompt: str) -> float:
        """Compute cosine similarity between image and text embeddings."""
        self._ensure_loaded()

        # Get embeddings for image and text
        image_inputs = [{"image": image, "instruction": "Represent this image for retrieval."}]
        text_inputs = [{"text": prompt, "instruction": "Represent this text for retrieval."}]

        image_emb = self._model.process(image_inputs).cpu().numpy()
        text_emb = self._model.process(text_inputs).cpu().numpy()

        # Cosine similarity (already normalized)
        similarity = float(np.dot(image_emb.flatten(), text_emb.flatten()))

        # Scale from [-1, 1] to [0, 100] (typical range is 0.2-0.8)
        score = max(0, min(100, (similarity - 0.2) / 0.6 * 100))
        return score

    def clear(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


class OfficialQwenRerankerScorer:
    """Wrapper for official Qwen3VLReranker for direct relevance scoring."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            from models.qwen3_vl_reranker import Qwen3VLReranker
        except ImportError:
            raise ImportError(
                "Qwen3-VL-Embedding not found. Clone it to mlx-mflux root:\n"
                "  git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git"
            )

        self._model = Qwen3VLReranker(
            model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
            torch_dtype="float32",  # MPS compatibility
        )

    def compute_score(self, image: Image.Image, prompt: str) -> float:
        """Compute relevance score between image and text."""
        self._ensure_loaded()

        inputs = {
            "instruction": "Retrieve images that match the user's query.",
            "query": {"text": prompt},
            "documents": [{"image": image}],
        }

        scores = self._model.process(inputs)
        if not scores:
            return 0.0

        # Reranker outputs 0-1, scale to 0-100
        return float(scores[0] * 100)

    def clear(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


# Test resources directory
RESOURCES = Path(__file__).parent / "resources"


@dataclass
class TestCase:
    """A single test case for scoring evaluation."""

    image_name: str
    prompt: str
    expected: Literal["high", "medium", "low"]
    category: str = ""


@dataclass
class ScoringResult:
    """Result of scoring a single test case."""

    test_case: TestCase
    score: float
    elapsed_ms: float
    matches_expectation: bool


@dataclass
class BackendResults:
    """Aggregated results for a scoring backend."""

    name: str
    results: list[ScoringResult] = field(default_factory=list)
    load_time_s: float = 0.0

    @property
    def accuracy(self) -> float:
        """Percentage of test cases matching expectations."""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.matches_expectation)
        return 100 * correct / len(self.results)

    @property
    def avg_time_ms(self) -> float:
        """Average scoring time in milliseconds."""
        if not self.results:
            return 0.0
        return sum(r.elapsed_ms for r in self.results) / len(self.results)

    @property
    def high_scores(self) -> list[float]:
        """Scores for 'high' expectation test cases."""
        return [r.score for r in self.results if r.test_case.expected == "high"]

    @property
    def low_scores(self) -> list[float]:
        """Scores for 'low' expectation test cases."""
        return [r.score for r in self.results if r.test_case.expected == "low"]

    @property
    def discrimination(self) -> float:
        """Gap between mean high and low scores (higher = better)."""
        high = self.high_scores
        low = self.low_scores
        if not high or not low:
            return 0.0
        return sum(high) / len(high) - sum(low) / len(low)


# Test cases organized by category
# NOTE: Prompts are based on actual image content (verified visually)
TEST_CASES = [
    # High alignment - should score 45+
    # reference_qwen_txt2img.png: fancy dessert (panna cotta style)
    TestCase(
        image_name="reference_qwen_txt2img.png",
        prompt="elegant dessert on a white plate, fine dining, luxury food",
        expected="high",
        category="matching",
    ),
    # reference_dev.png: fried seafood (shrimp/calamari) with vegetables
    TestCase(
        image_name="reference_dev.png",
        prompt="Gourmet fried seafood with vegetables on a plate",
        expected="high",
        category="matching",
    ),
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="tall buildings city skyline urban architecture",
        expected="high",
        category="matching",
    ),
    # reference_z_image_turbo.png: cat astronaut in space station
    TestCase(
        image_name="reference_z_image_turbo.png",
        prompt="a cat in a spacesuit floating in a space station",
        expected="high",
        category="matching",
    ),
    # Medium alignment - should score 35-55
    TestCase(
        image_name="reference_qwen_txt2img.png",
        prompt="food on a plate",
        expected="medium",
        category="partial",
    ),
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="modern architecture",
        expected="medium",
        category="partial",
    ),
    TestCase(
        image_name="reference_dev.png",
        prompt="a meal being served",
        expected="medium",
        category="partial",
    ),
    TestCase(
        image_name="reference_z_image_turbo.png",
        prompt="pet floating in zero gravity",
        expected="medium",
        category="partial",
    ),
    # Low alignment - should score <35
    TestCase(
        image_name="reference_qwen_txt2img.png",
        prompt="a cat sleeping on a red couch",
        expected="low",
        category="mismatch",
    ),
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="underwater coral reef with tropical fish",
        expected="low",
        category="mismatch",
    ),
    TestCase(
        image_name="reference_dev.png",
        prompt="snowy mountain peak at sunrise with hikers",
        expected="low",
        category="mismatch",
    ),
    TestCase(
        image_name="reference_z_image_turbo.png",
        prompt="a busy city street with cars and people",
        expected="low",
        category="mismatch",
    ),
    # Semantic similarity tests - same image, different phrasings
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="skyscrapers",
        expected="high",
        category="semantic",
    ),
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="tall office buildings",
        expected="high",
        category="semantic",
    ),
    TestCase(
        image_name="skyscrapers.jpg",
        prompt="downtown metropolitan area",
        expected="medium",
        category="semantic",
    ),
]


def check_expectation(score: float, expected: Literal["high", "medium", "low"]) -> bool:
    """Check if score matches expectation.

    Thresholds calibrated for CLIP ViT-B/32:
    - CLIP cosine similarities typically range 0.2-0.4 (raw)
    - After scaling to 0-100: high=45+, medium=35-70, low=<35

    The key metric is discrimination (gap between high and low),
    not absolute scores. Medium range is wide to accommodate
    semantic similarity where partial descriptions still match well.
    """
    if expected == "high":
        return score >= 45  # CLIP high matches typically 45-70
    elif expected == "medium":
        return 35 <= score < 70  # Wide range for partial matches
    else:  # low
        return score < 35  # Mismatches typically <35


def create_backend(name: str) -> ImageTextScorer:
    """Create a scorer backend by name."""
    if name == "clip":
        return QwenCLIPScorer()
    elif name == "qwen-vl-embedding":
        return OfficialQwenEmbeddingScorer()
    elif name == "qwen-vl-reranker":
        return OfficialQwenRerankerScorer()
    elif name == "mlx-qwen-embedding":
        from mflux.models.qwen.variants.training.validation.clip_scorer import (
            MLXQwenEmbeddingScorer,
        )

        return MLXQwenEmbeddingScorer()
    elif name == "mlx-qwen-reranker":
        from mflux.models.qwen.variants.training.validation.clip_scorer import (
            MLXQwenRerankerScorer,
        )

        return MLXQwenRerankerScorer()
    else:
        raise ValueError(f"Unknown backend: {name}")


def run_benchmark(
    backends: list[str],
    verbose: bool = False,
) -> dict[str, BackendResults]:
    """Run the full benchmark suite.

    Args:
        backends: List of backend names to test
        verbose: Whether to print detailed output

    Returns:
        Dictionary mapping backend name to results
    """
    results: dict[str, BackendResults] = {}

    # Filter test cases to only those with existing images
    valid_cases = []
    for tc in TEST_CASES:
        img_path = RESOURCES / tc.image_name
        if img_path.exists():
            valid_cases.append(tc)
        elif verbose:
            print(f"  Skipping {tc.image_name} (not found)")

    if not valid_cases:
        print("ERROR: No valid test images found!")
        print(f"Looked in: {RESOURCES}")
        return results

    print(f"Found {len(valid_cases)} valid test cases")

    for backend_name in backends:
        print(f"\n{'=' * 60}")
        print(f"Testing: {backend_name}")
        print("=" * 60)

        backend_results = BackendResults(name=backend_name)

        try:
            # Create and load scorer (measure load time)
            load_start = time.time()
            scorer = create_backend(backend_name)

            # Force model load by scoring a dummy image
            dummy_path = RESOURCES / valid_cases[0].image_name
            dummy_img = Image.open(dummy_path)
            _ = scorer.compute_score(dummy_img, "test")

            backend_results.load_time_s = time.time() - load_start
            print(f"Model loaded in {backend_results.load_time_s:.1f}s")

        except (ImportError, RuntimeError, OSError, ValueError) as e:
            print(f"ERROR: Failed to load {backend_name}: {e}")
            results[backend_name] = backend_results
            continue

        # Run all test cases
        print(f"\n{'Image':<25} | {'Prompt':<35} | {'Score':>6} | {'Expect':<6} | Status")
        print("-" * 90)

        for tc in valid_cases:
            img_path = RESOURCES / tc.image_name
            image = Image.open(img_path)

            start = time.time()
            try:
                score = scorer.compute_score(image, tc.prompt)
            except (ValueError, TypeError, RuntimeError, OSError) as e:
                print(f"  ERROR scoring {tc.image_name}: {e}")
                continue
            elapsed_ms = (time.time() - start) * 1000

            matches = check_expectation(score, tc.expected)
            status = "✓" if matches else "✗"

            result = ScoringResult(
                test_case=tc,
                score=score,
                elapsed_ms=elapsed_ms,
                matches_expectation=matches,
            )
            backend_results.results.append(result)

            if verbose or not matches:
                # Always show failures, optionally show all
                prompt_display = tc.prompt[:33] + ".." if len(tc.prompt) > 35 else tc.prompt
                print(f"{tc.image_name:<25} | {prompt_display:<35} | {score:>5.1f} | {tc.expected:<6} | {status}")

        # Print summary for this backend
        print(f"\n{backend_name} Summary:")
        print(
            f"  Accuracy: {backend_results.accuracy:.0f}% ({sum(1 for r in backend_results.results if r.matches_expectation)}/{len(backend_results.results)})"
        )
        print(f"  Discrimination (high-low gap): {backend_results.discrimination:.1f}")
        print(f"  Avg scoring time: {backend_results.avg_time_ms:.0f}ms")

        # Cleanup
        scorer.clear()
        results[backend_name] = backend_results

    return results


def print_comparison(results: dict[str, BackendResults]) -> None:
    """Print comparison table across all backends."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    headers = ["Backend", "Accuracy", "Discrimination", "Avg Time", "Load Time"]
    print(f"{headers[0]:<25} | {headers[1]:>10} | {headers[2]:>14} | {headers[3]:>10} | {headers[4]:>10}")
    print("-" * 70)

    for name, r in results.items():
        print(
            f"{name:<25} | {r.accuracy:>9.0f}% | {r.discrimination:>14.1f} | {r.avg_time_ms:>8.0f}ms | {r.load_time_s:>8.1f}s"
        )

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")

    # Find best by accuracy
    best_accuracy = max(results.values(), key=lambda x: x.accuracy)
    print(f"  Best accuracy: {best_accuracy.name} ({best_accuracy.accuracy:.0f}%)")

    # Find best by discrimination
    best_disc = max(results.values(), key=lambda x: x.discrimination)
    print(f"  Best discrimination: {best_disc.name} ({best_disc.discrimination:.1f})")

    # Find fastest
    fastest = min(results.values(), key=lambda x: x.avg_time_ms if x.avg_time_ms > 0 else float("inf"))
    print(f"  Fastest: {fastest.name} ({fastest.avg_time_ms:.0f}ms)")

    # Overall recommendation
    print("\n  Overall:")
    if best_accuracy.name == best_disc.name:
        winner = best_accuracy.name
        print(f"    → {winner} is the clear winner (best accuracy AND discrimination)")
    else:
        # Score each backend (accuracy weight + discrimination weight)
        def score_backend(r: BackendResults) -> float:
            # Normalize: accuracy 0-100, discrimination 0-100
            acc_norm = r.accuracy
            disc_norm = min(100, max(0, r.discrimination))
            return acc_norm * 0.6 + disc_norm * 0.4

        best_overall = max(results.values(), key=score_backend)
        print(f"    → {best_overall.name} is recommended (balanced accuracy + discrimination)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL vs CLIP scoring quality")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["clip", "qwen-vl-embedding", "qwen-vl-reranker"],
        choices=[
            "clip",
            "qwen-vl-embedding",
            "qwen-vl-reranker",
            "mlx-qwen-embedding",
            "mlx-qwen-reranker",
        ],
        help="Backends to test (default: 2B PyTorch models + clip, use mlx-* for native MLX)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for all test cases",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3-VL vs CLIP Scoring Quality Benchmark")
    print("=" * 70)
    print(f"Test resources: {RESOURCES}")
    print(f"Backends: {', '.join(args.backends)}")

    results = run_benchmark(args.backends, verbose=args.verbose)
    print_comparison(results)


if __name__ == "__main__":
    main()
