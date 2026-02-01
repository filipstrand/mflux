"""
Prefetch Iterator for MLX Training.

Background thread prefetches next batch while GPU trains on current batch.
Overlaps data loading with GPU computation for 10-20% speedup.

Features:
- Background prefetching with configurable queue depth
- Thread-safe with proper exception handling
- Graceful shutdown
- Works with any iterable

IMPORTANT: MLX is not thread-safe for GPU operations. Data loading
(disk I/O) happens in background thread, but mx.eval() must happen
in main thread before data is used for training.
"""

import random
import threading
from queue import Empty, Queue
from typing import Any, Callable, Iterator

from mflux.models.qwen.variants.training.optimization.qwen_loss import (
    QwenTrainingBatch,
    QwenTrainingExample,
)


class PrefetchIterator:
    """
    Wraps an iterator to prefetch items in background thread.

    Overlaps data loading with GPU computation for improved throughput.

    Args:
        base_iterator: Iterator that yields batch indices
        load_fn: Function to load examples from indices
        prefetch_count: Number of batches to prefetch (default: 2)

    Example:
        def load_batch(indices):
            return [dataset[idx] for idx in indices]

        base_iter = StreamingIterator(dataset, batch_size=4)
        prefetch_iter = PrefetchIterator(base_iter, load_fn=load_batch)

        for batch in prefetch_iter:
            loss, grads = train_step(batch)
    """

    def __init__(
        self,
        base_iterator: Iterator[list[int]],
        load_fn: Callable[[list[int]], list[QwenTrainingExample]],
        prefetch_count: int = 2,
    ):
        self.base_iterator = base_iterator
        self.load_fn = load_fn
        self.prefetch_count = prefetch_count

        self._queue: Queue[QwenTrainingBatch | StopIteration | Exception] = Queue(maxsize=prefetch_count)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None
        self._started = False

    def __iter__(self) -> Iterator[QwenTrainingBatch]:
        self._start_prefetch_thread()
        return self

    def __next__(self) -> QwenTrainingBatch:
        if self._exception is not None:
            raise self._exception

        try:
            item = self._queue.get(timeout=60.0)  # 60s timeout
        except Empty:
            raise RuntimeError("Prefetch queue timed out - data loading may be stuck")

        if isinstance(item, StopIteration):
            raise StopIteration

        if isinstance(item, Exception):
            self._exception = item
            raise item

        return item

    def _start_prefetch_thread(self) -> None:
        """Start background prefetch thread."""
        if self._started:
            return

        self._stop_event.clear()
        self._exception = None
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        self._started = True

    def _prefetch_loop(self) -> None:
        """Background thread that prefetches batches."""
        try:
            for batch_indices in self.base_iterator:
                if self._stop_event.is_set():
                    break

                try:
                    # Load examples (disk I/O happens here)
                    examples = self.load_fn(batch_indices)

                    # Create batch with random state
                    rng = random.Random(random.randint(0, 2**32 - 1))
                    batch = QwenTrainingBatch(examples=examples, rng=rng)

                    # Put in queue (blocks if queue full)
                    self._queue.put(batch)

                except Exception as e:
                    self._queue.put(e)
                    return

            # Signal end of iteration
            self._queue.put(StopIteration())

        except Exception as e:
            self._queue.put(e)

    def stop(self) -> None:
        """Stop prefetch thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False

    def __del__(self):
        self.stop()


class PrefetchDataLoader:
    """
    Higher-level prefetch data loader that combines streaming and prefetching.

    Args:
        dataset: StreamingDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle each epoch
        prefetch_count: Number of batches to prefetch
        seed: Random seed

    Example:
        dataloader = PrefetchDataLoader(
            dataset=streaming_dataset,
            batch_size=4,
            prefetch_count=2,
        )

        for epoch in range(num_epochs):
            for batch in dataloader:
                loss, grads = train_step(batch)

            dataloader.reset()  # Reset for next epoch
    """

    def __init__(
        self,
        dataset: Any,  # StreamingDataset
        batch_size: int = 1,
        shuffle: bool = True,
        prefetch_count: int = 2,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_count = prefetch_count
        self.seed = seed

        self._rng = random.Random(seed)
        self._indices: list[int] = list(range(len(dataset)))
        self._epoch = 0
        self._current_iterator: PrefetchIterator | None = None

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self._indices) // self.batch_size

    def __iter__(self) -> Iterator[QwenTrainingBatch]:
        # Shuffle if needed
        if self.shuffle:
            self._rng.shuffle(self._indices)

        # Create batch index generator
        def batch_indices_generator():
            for i in range(0, len(self._indices), self.batch_size):
                batch_indices = self._indices[i : i + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    yield batch_indices

        # Create load function
        def load_fn(indices: list[int]) -> list[QwenTrainingExample]:
            return [self.dataset[idx] for idx in indices]

        # Create prefetch iterator
        self._current_iterator = PrefetchIterator(
            base_iterator=batch_indices_generator(),
            load_fn=load_fn,
            prefetch_count=self.prefetch_count,
        )

        return iter(self._current_iterator)

    def reset(self) -> None:
        """Reset for next epoch."""
        if self._current_iterator is not None:
            self._current_iterator.stop()
            self._current_iterator = None
        self._epoch += 1

    def stop(self) -> None:
        """Stop any running prefetch threads."""
        if self._current_iterator is not None:
            self._current_iterator.stop()
            self._current_iterator = None


class SimplePrefetchIterator:
    """
    Simpler prefetch iterator that works with pre-loaded examples.

    For cases where you have examples in memory but want to overlap
    batch creation with training.

    Args:
        examples: List of QwenTrainingExample
        batch_size: Batch size
        shuffle: Whether to shuffle
        prefetch_count: Number of batches to prefetch
    """

    def __init__(
        self,
        examples: list[QwenTrainingExample],
        batch_size: int = 1,
        shuffle: bool = True,
        prefetch_count: int = 2,
        seed: int = 42,
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_count = prefetch_count

        self._rng = random.Random(seed)
        self._queue: Queue[QwenTrainingBatch | None] = Queue(maxsize=prefetch_count)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __iter__(self) -> Iterator[QwenTrainingBatch]:
        self._start_thread()
        return self

    def __next__(self) -> QwenTrainingBatch:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item

    def _start_thread(self) -> None:
        """Start background thread."""
        self._stop_event.clear()

        indices = list(range(len(self.examples)))
        if self.shuffle:
            self._rng.shuffle(indices)

        self._thread = threading.Thread(
            target=self._prefetch_loop,
            args=(indices,),
            daemon=True,
        )
        self._thread.start()

    def _prefetch_loop(self, indices: list[int]) -> None:
        """Background thread loop."""
        try:
            for i in range(0, len(indices), self.batch_size):
                if self._stop_event.is_set():
                    break

                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue

                examples = [self.examples[idx] for idx in batch_indices]
                rng = random.Random(self._rng.randint(0, 2**32 - 1))
                batch = QwenTrainingBatch(examples=examples, rng=rng)

                self._queue.put(batch)

            self._queue.put(None)  # Signal end

        except Exception:
            self._queue.put(None)

    def stop(self) -> None:
        """Stop background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
