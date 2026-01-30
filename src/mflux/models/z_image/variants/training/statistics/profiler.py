"""Training profiler for performance analysis and optimization.

This module provides timing instrumentation for identifying training
bottlenecks and measuring optimization effectiveness.

Usage:
    profiler = TrainingProfiler(enabled=True)

    with profiler.time_section("forward"):
        loss, grads = train_step(batch)

    with profiler.time_section("backward"):
        optimizer.update(model, grads)

    # Get timing report
    profiler.report()

Security Note:
    Profiler data contains detailed timing information that could reveal
    information about model architecture, data sizes, or processing patterns.
    Do not expose profiler output to untrusted parties.
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimingStats:
    """Statistics for a timed section."""

    name: str
    total_time: float
    call_count: int
    min_time: float
    max_time: float
    times: list[float]

    @property
    def avg_time(self) -> float:
        """Average time per call in seconds."""
        return self.total_time / max(1, self.call_count)

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.avg_time * 1000

    @property
    def total_time_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_time * 1000

    @property
    def min_time_ms(self) -> float:
        """Minimum time in milliseconds."""
        return self.min_time * 1000

    @property
    def max_time_ms(self) -> float:
        """Maximum time in milliseconds."""
        return self.max_time * 1000


class TrainingProfiler:
    """Profiler for training performance analysis.

    Provides timing instrumentation for different training phases:
    - forward: Forward pass and loss computation
    - backward: Gradient computation
    - optimizer: Parameter updates
    - sync: GPU synchronization
    - data: Data loading and preprocessing

    The profiler can be enabled/disabled without code changes,
    making it safe to leave in production code.

    Attributes:
        enabled: Whether profiling is active
        timings: Dictionary of timing data per section
        max_history_size: Maximum number of timing entries to keep per section
    """

    # Standard training section names
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    SYNC = "sync"
    DATA = "data"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"

    # Default maximum history size to prevent unbounded memory growth
    DEFAULT_MAX_HISTORY = 10_000

    def __init__(
        self,
        enabled: bool = False,
        keep_history: bool = True,
        max_history_size: int = DEFAULT_MAX_HISTORY,
    ):
        """Initialize the training profiler.

        Args:
            enabled: Whether profiling is active (default False)
            keep_history: Whether to keep individual timing history
                         (uses more memory but enables percentile analysis)
            max_history_size: Maximum timing entries per section (default 10000).
                            Older entries are discarded when limit is reached.
        """
        self.enabled = enabled
        self.keep_history = keep_history
        self.max_history_size = max_history_size
        # Use deque with maxlen for O(1) bounded append - auto-evicts oldest entries
        self._timings: dict[str, deque[float]] = {}
        self._total_times: dict[str, float] = defaultdict(float)
        self._call_counts: dict[str, int] = defaultdict(int)
        # Both min and max use regular dicts with explicit initialization for consistency
        self._min_times: dict[str, float] = {}
        self._max_times: dict[str, float] = {}

    @contextmanager
    def time_section(self, name: str) -> Iterator[None]:
        """Context manager for timing a code section.

        Args:
            name: Name of the section (e.g., "forward", "optimizer")

        Yields:
            None (context manager)

        Example:
            with profiler.time_section("forward"):
                loss = model(batch)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record_timing(name, elapsed)

    def _record_timing(self, name: str, elapsed: float) -> None:
        """Record a timing measurement.

        Args:
            name: Section name
            elapsed: Elapsed time in seconds

        Note:
            History is bounded by max_history_size using deque with maxlen.
            Oldest entries are automatically evicted in O(1) time.
        """
        if self.keep_history:
            # Lazily create deque with maxlen for each section
            if name not in self._timings:
                self._timings[name] = deque(maxlen=self.max_history_size)
            # deque with maxlen auto-evicts oldest entries in O(1)
            self._timings[name].append(elapsed)

        self._total_times[name] += elapsed
        self._call_counts[name] += 1

        # Track min/max with consistent initialization pattern
        is_first = name not in self._min_times
        if is_first:
            self._min_times[name] = elapsed
            self._max_times[name] = elapsed
        else:
            self._min_times[name] = min(self._min_times[name], elapsed)
            self._max_times[name] = max(self._max_times[name], elapsed)

    def start_section(self, name: str) -> float:
        """Start timing a section manually.

        Use this when context manager syntax isn't convenient.
        Returns the start time to pass to end_section.

        Args:
            name: Section name

        Returns:
            Start timestamp
        """
        if not self.enabled:
            return 0.0
        return time.perf_counter()

    def end_section(self, name: str, start_time: float) -> float:
        """End timing a section manually.

        Args:
            name: Section name
            start_time: Start timestamp from start_section

        Returns:
            Elapsed time in seconds
        """
        if not self.enabled:
            return 0.0
        elapsed = time.perf_counter() - start_time
        self._record_timing(name, elapsed)
        return elapsed

    def get_stats(self, name: str) -> TimingStats | None:
        """Get statistics for a specific section.

        Args:
            name: Section name

        Returns:
            TimingStats object, or None if section not recorded
        """
        if name not in self._call_counts:
            return None

        # Convert deque to list for TimingStats (expected by callers)
        times_list = list(self._timings.get(name, [])) if self.keep_history else []
        return TimingStats(
            name=name,
            total_time=self._total_times[name],
            call_count=self._call_counts[name],
            min_time=self._min_times.get(name, 0.0),
            max_time=self._max_times[name],
            times=times_list,
        )

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get statistics for all recorded sections.

        Returns:
            Dictionary mapping section names to TimingStats
        """
        result: dict[str, TimingStats] = {}
        for name in self._call_counts:
            stats = self.get_stats(name)
            if stats is not None:
                result[name] = stats
        return result

    def report(self, include_percentiles: bool = False) -> str:
        """Generate a formatted timing report.

        Args:
            include_percentiles: Whether to include p50/p90/p99 stats
                               (requires keep_history=True)

        Returns:
            Formatted string report
        """
        if not self._call_counts:
            return "No timings recorded"

        lines = ["Training Profiler Report", "=" * 60]

        # Calculate total time across all sections
        total_all = sum(self._total_times.values())

        # Sort sections by total time (descending)
        sorted_sections = sorted(
            self._call_counts.keys(),
            key=lambda x: self._total_times[x],
            reverse=True,
        )

        for name in sorted_sections:
            stats = self.get_stats(name)
            if stats is None:
                continue

            pct = (stats.total_time / total_all * 100) if total_all > 0 else 0

            lines.append(f"\n{name}:")
            lines.append(f"  Calls:     {stats.call_count:,}")
            lines.append(f"  Total:     {stats.total_time_ms:,.2f}ms ({pct:.1f}%)")
            lines.append(f"  Avg:       {stats.avg_time_ms:.2f}ms")
            lines.append(f"  Min/Max:   {stats.min_time_ms:.2f}ms / {stats.max_time_ms:.2f}ms")

            if include_percentiles and self.keep_history and stats.times:
                sorted_times = sorted(stats.times)
                n = len(sorted_times)
                p50 = sorted_times[n // 2] * 1000
                p90 = sorted_times[int(n * 0.9)] * 1000
                p99 = sorted_times[int(n * 0.99)] * 1000
                lines.append(f"  P50/P90/P99: {p50:.2f}ms / {p90:.2f}ms / {p99:.2f}ms")

        lines.append(f"\n{'=' * 60}")
        lines.append(f"Total recorded time: {total_all * 1000:,.2f}ms")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all timing data."""
        self._timings.clear()
        self._total_times.clear()
        self._call_counts.clear()
        self._min_times.clear()
        self._max_times.clear()

    def to_dict(self) -> dict:
        """Export timing data as dictionary.

        Returns:
            Dictionary with timing data for serialization
        """
        return {
            "sections": {
                name: {
                    "total_time_ms": self._total_times[name] * 1000,
                    "call_count": self._call_counts[name],
                    "avg_time_ms": self._total_times[name] / max(1, self._call_counts[name]) * 1000,
                    "min_time_ms": self._min_times.get(name, 0) * 1000,
                    "max_time_ms": self._max_times[name] * 1000,
                }
                for name in self._call_counts
            }
        }


class NullProfiler(TrainingProfiler):
    """No-op profiler for when profiling is disabled.

    This class provides the same interface as TrainingProfiler
    but with zero overhead when profiling is not needed.
    """

    def __init__(self):
        super().__init__(enabled=False, keep_history=False)

    @contextmanager
    def time_section(self, name: str) -> Iterator[None]:
        yield

    def _record_timing(self, name: str, elapsed: float) -> None:
        pass

    def start_section(self, name: str) -> float:
        return 0.0

    def end_section(self, name: str, start_time: float) -> float:
        return 0.0


def create_profiler(
    enabled: bool = False,
    keep_history: bool = True,
    max_history_size: int = TrainingProfiler.DEFAULT_MAX_HISTORY,
) -> TrainingProfiler:
    """Factory function to create appropriate profiler.

    Args:
        enabled: Whether profiling is active
        keep_history: Whether to keep timing history
        max_history_size: Maximum timing entries per section (default 10000)

    Returns:
        TrainingProfiler or NullProfiler instance
    """
    if not enabled:
        return NullProfiler()
    return TrainingProfiler(
        enabled=True,
        keep_history=keep_history,
        max_history_size=max_history_size,
    )
