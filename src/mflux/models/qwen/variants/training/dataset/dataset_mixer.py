"""
Dataset Mixer for Multi-Dataset Training.

Mixes multiple datasets with configurable weights for large-scale
training on diverse data sources.

Features:
- Weighted sampling across multiple datasets
- Configurable per-dataset weights
- Iterative batch generation with mixed sources
- Support for StreamingDataset and other dataset types
- Thread-safe sampling with locks

Note: This module is designed for single-threaded use by default.
For multi-threaded DataLoaders, each worker should have its own
mixer instance with a different seed.
"""

import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mflux.models.qwen.variants.training.optimization.qwen_loss import (
    QwenTrainingBatch,
    QwenTrainingExample,
)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in a mix."""

    path: Path | str
    weight: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if self.name is None:
            self.name = Path(self.path).stem


class DatasetMixer:
    """
    Mixes multiple datasets with configurable weights.

    Samples examples from multiple datasets according to specified
    weights. Useful for training on diverse data sources like
    WikiArt + CivitAI + custom data.

    Args:
        datasets_with_weights: List of (dataset, weight) tuples.
            Each dataset must support __len__ and __getitem__.
        seed: Random seed for reproducibility

    Raises:
        ValueError: If no datasets provided, weights are invalid, or any dataset is empty
        TypeError: If any dataset doesn't support required methods

    Example:
        mixer = DatasetMixer([
            (wikiart_dataset, 0.3),
            (civitai_dataset, 0.5),
            (custom_dataset, 0.2),
        ])

        # Sample examples according to weights
        for step in range(num_steps):
            example = mixer.sample()

        # Or get a full batch
        batch = mixer.get_batch(batch_size=4)
    """

    def __init__(
        self,
        datasets_with_weights: list[tuple[Any, float]],
        seed: int = 42,
    ) -> None:
        if not datasets_with_weights:
            raise ValueError("At least one dataset is required")

        self.datasets = [d for d, _ in datasets_with_weights]
        self.weights = [w for _, w in datasets_with_weights]

        # Validate dataset types
        for i, dataset in enumerate(self.datasets):
            if not hasattr(dataset, "__len__"):
                raise TypeError(f"Dataset at index {i} must support __len__ (have a length)")
            if not hasattr(dataset, "__getitem__"):
                raise TypeError(f"Dataset at index {i} must support __getitem__ (be indexable)")

        # Validate datasets are non-empty
        self._dataset_lengths: list[int] = []
        for i, dataset in enumerate(self.datasets):
            length = len(dataset)
            if length == 0:
                raise ValueError(f"Dataset at index {i} is empty")
            self._dataset_lengths.append(length)

        # Validate weights
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative")

        # Normalize weights
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("Total weight must be positive (at least one weight > 0)")
        self.weights = [w / total for w in self.weights]

        # Build cumulative weights for sampling
        # Use strict < comparison to avoid float precision issues
        self._cumulative: list[float] = []
        cumsum = 0.0
        for w in self.weights:
            cumsum += w
            self._cumulative.append(cumsum)

        # Ensure last cumulative weight is exactly 1.0 to avoid float precision issues
        if self._cumulative:
            self._cumulative[-1] = 1.0

        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        # Total virtual length
        self._total_len = sum(self._dataset_lengths)

    @property
    def num_datasets(self) -> int:
        """Number of datasets in the mixer."""
        return len(self.datasets)

    def __len__(self) -> int:
        """Total number of examples across all datasets."""
        return self._total_len

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DatasetMixer(num_datasets={len(self.datasets)}, "
            f"weights={[f'{w:.2%}' for w in self.weights]}, "
            f"total_examples={self._total_len})"
        )

    def sample(self) -> QwenTrainingExample:
        """
        Sample one example according to weights.

        Thread-safe: uses internal lock for RNG and dataset access.

        Returns:
            A training example from one of the datasets
        """
        with self._lock:
            r = self._rng.random()
            dataset_idx = self._select_dataset(r)
            idx = self._rng.randint(0, self._dataset_lengths[dataset_idx] - 1)
            # Access dataset inside lock to ensure thread safety
            result = self.datasets[dataset_idx][idx]

        return result

    def _select_dataset(self, r: float) -> int:
        """Select dataset index based on random value r in [0, 1)."""
        # Use < instead of <= to handle float precision at boundaries
        for i, cum in enumerate(self._cumulative):
            if r < cum:
                return i
        # Fallback to last dataset (should rarely happen due to cum[-1] = 1.0)
        return len(self.datasets) - 1

    def sample_with_source(self) -> tuple[QwenTrainingExample, int]:
        """
        Sample one example and return (example, dataset_index).

        Thread-safe: uses internal lock for RNG and dataset access.

        Returns:
            Tuple of (training example, index of source dataset)
        """
        with self._lock:
            r = self._rng.random()
            dataset_idx = self._select_dataset(r)
            idx = self._rng.randint(0, self._dataset_lengths[dataset_idx] - 1)
            # Access dataset inside lock to ensure thread safety
            result = self.datasets[dataset_idx][idx]

        return result, dataset_idx

    def get_batch(self, batch_size: int) -> QwenTrainingBatch:
        """
        Sample a batch according to weights.

        Args:
            batch_size: Number of examples in the batch (must be positive)

        Returns:
            A training batch with the specified number of examples

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        examples = [self.sample() for _ in range(batch_size)]

        with self._lock:
            batch_seed = self._rng.randint(0, 2**31 - 1)

        rng = random.Random(batch_seed)
        return QwenTrainingBatch(examples=examples, rng=rng)

    def get_batch_with_sources(self, batch_size: int) -> tuple[QwenTrainingBatch, list[int]]:
        """
        Sample a batch and return (batch, list of dataset indices).

        Args:
            batch_size: Number of examples in the batch (must be positive)

        Returns:
            Tuple of (training batch, list of source dataset indices)

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        examples = []
        sources = []
        for _ in range(batch_size):
            ex, src = self.sample_with_source()
            examples.append(ex)
            sources.append(src)

        with self._lock:
            batch_seed = self._rng.randint(0, 2**31 - 1)

        rng = random.Random(batch_seed)
        return QwenTrainingBatch(examples=examples, rng=rng), sources

    def get_dataset_stats(self) -> dict[str, Any]:
        """
        Get statistics about the mixed datasets.

        Returns:
            Dictionary with num_datasets, weights, sizes, and total_examples
        """
        return {
            "num_datasets": len(self.datasets),
            "weights": self.weights,
            "sizes": self._dataset_lengths,
            "total_examples": self._total_len,
        }


class MixedDatasetIterator:
    """
    Iterator for mixed dataset that generates batches according to weights.

    Yields batches indefinitely (use with step-based training).
    This iterator never raises StopIteration - use a step counter
    to limit training.

    Args:
        mixer: DatasetMixer instance
        batch_size: Batch size (must be positive)
        track_sources: Whether to track which dataset each example came from

    Raises:
        ValueError: If batch_size <= 0
    """

    def __init__(
        self,
        mixer: DatasetMixer,
        batch_size: int = 1,
        track_sources: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.mixer = mixer
        self.batch_size = batch_size
        self.track_sources = track_sources

        self._step = 0
        self._source_counts: dict[int, int] = {i: 0 for i in range(mixer.num_datasets)}

    @property
    def step(self) -> int:
        """Current step count (number of batches generated)."""
        return self._step

    @property
    def source_distribution(self) -> dict[int, float]:
        """
        Actual distribution of examples from each dataset.

        Returns:
            Dictionary mapping dataset index to fraction of total examples
        """
        total = sum(self._source_counts.values())
        if total == 0:
            return {i: 0.0 for i in range(self.mixer.num_datasets)}
        return {i: count / total for i, count in self._source_counts.items()}

    def get_source_counts(self) -> dict[int, int]:
        """
        Get raw source counts (number of examples from each dataset).

        Returns:
            Dictionary mapping dataset index to count
        """
        return dict(self._source_counts)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"MixedDatasetIterator(mixer={self.mixer}, batch_size={self.batch_size}, step={self._step})"

    def __iter__(self) -> Iterator[QwenTrainingBatch]:
        return self

    def __next__(self) -> QwenTrainingBatch:
        if self.track_sources:
            batch, sources = self.mixer.get_batch_with_sources(self.batch_size)
            for src in sources:
                self._source_counts[src] += 1
        else:
            batch = self.mixer.get_batch(self.batch_size)

        self._step += 1
        return batch


class BalancedMixer:
    """
    Alternative mixer that ensures balanced sampling across epochs.

    Instead of pure weighted random sampling, this ensures each
    dataset is fully traversed each "epoch" with examples shuffled.
    All datasets are given equal weight per-example.

    Args:
        datasets: List of datasets (each must support __len__ and __getitem__)
        seed: Random seed

    Raises:
        ValueError: If no datasets provided or all datasets are empty
        TypeError: If any dataset doesn't support required methods
    """

    def __init__(self, datasets: list[Any], seed: int = 42) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required")

        # Validate dataset types
        for i, dataset in enumerate(datasets):
            if not hasattr(dataset, "__len__"):
                raise TypeError(f"Dataset at index {i} must support __len__ (have a length)")
            if not hasattr(dataset, "__getitem__"):
                raise TypeError(f"Dataset at index {i} must support __getitem__ (be indexable)")

        self.datasets = datasets
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        # Flatten all indices
        self._all_indices: list[tuple[int, int]] = []  # (dataset_idx, item_idx)
        for d_idx, dataset in enumerate(datasets):
            dataset_len = len(dataset)
            for i_idx in range(dataset_len):
                self._all_indices.append((d_idx, i_idx))

        if not self._all_indices:
            raise ValueError("BalancedMixer requires at least one non-empty dataset")

        self._shuffled_indices: list[tuple[int, int]] = []
        self._current_idx = 0
        self._epoch = 0
        self._reshuffle()

    @property
    def epoch(self) -> int:
        """Current epoch number (1-indexed, incremented after each full pass)."""
        return self._epoch

    def __len__(self) -> int:
        """Total number of examples across all datasets."""
        return len(self._all_indices)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BalancedMixer(num_datasets={len(self.datasets)}, "
            f"total_examples={len(self._all_indices)}, epoch={self._epoch})"
        )

    def _reshuffle(self) -> None:
        """Reshuffle indices for new epoch."""
        self._shuffled_indices = list(self._all_indices)
        self._rng.shuffle(self._shuffled_indices)
        self._current_idx = 0
        self._epoch += 1

    def sample(self) -> QwenTrainingExample:
        """
        Get next example (reshuffles at epoch boundary).

        Thread-safe: uses internal lock for index and dataset access.

        Returns:
            The next training example in the shuffled order
        """
        with self._lock:
            if self._current_idx >= len(self._shuffled_indices):
                self._reshuffle()

            d_idx, i_idx = self._shuffled_indices[self._current_idx]
            self._current_idx += 1
            # Access dataset inside lock to ensure thread safety
            result = self.datasets[d_idx][i_idx]

        return result

    def get_batch(self, batch_size: int) -> QwenTrainingBatch:
        """
        Get next batch.

        Args:
            batch_size: Number of examples in the batch (must be positive)

        Returns:
            A training batch with the specified number of examples

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        examples = [self.sample() for _ in range(batch_size)]

        with self._lock:
            batch_seed = self._rng.randint(0, 2**31 - 1)

        rng = random.Random(batch_seed)
        return QwenTrainingBatch(examples=examples, rng=rng)
