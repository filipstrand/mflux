"""Memory monitoring for Z-Image training on Apple Silicon.

Provides real-time memory tracking and early OOM detection using MLX's
Metal memory APIs. Designed for Mac Studio M3 Ultra with 512GB unified memory.

Features:
- Real-time memory snapshots via mx.metal.get_active_memory()
- Warning thresholds for proactive intervention
- Batch size reduction suggestions based on memory pressure
- Training divergence detection
"""

import logging
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot.

    Attributes:
        active_bytes: Currently allocated memory in bytes
        peak_bytes: Peak memory usage since last reset
        cache_bytes: Memory in MLX cache (reclaimable)
        total_available_bytes: Total unified memory available
        utilization: Fraction of available memory in use (0.0-1.0)
        status: "ok", "warning", or "critical"
    """

    active_bytes: int
    peak_bytes: int
    cache_bytes: int
    total_available_bytes: int
    utilization: float
    status: str

    @property
    def active_gb(self) -> float:
        """Active memory in GB."""
        return self.active_bytes / (1024**3)

    @property
    def peak_gb(self) -> float:
        """Peak memory in GB."""
        return self.peak_bytes / (1024**3)

    @property
    def cache_gb(self) -> float:
        """Cache memory in GB."""
        return self.cache_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        """Total available memory in GB."""
        return self.total_available_bytes / (1024**3)


class MemoryMonitor:
    """Monitor memory usage during training with configurable thresholds.

    Usage:
        monitor = MemoryMonitor(
            warning_threshold=0.85,
            critical_threshold=0.95,
        )

        # In training loop:
        snapshot = monitor.check()
        if snapshot.status == "critical":
            logger.error("Memory critical!")
        elif snapshot.status == "warning":
            new_batch = monitor.suggest_batch_size_reduction(current_batch)
    """

    # Default thresholds tuned for 512GB systems
    DEFAULT_WARNING_THRESHOLD = 0.85
    DEFAULT_CRITICAL_THRESHOLD = 0.95

    def __init__(
        self,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        total_memory_gb: float | None = None,
        on_warning: Callable[[MemorySnapshot], None] | None = None,
        on_critical: Callable[[MemorySnapshot], None] | None = None,
    ):
        """Initialize memory monitor.

        Args:
            warning_threshold: Memory utilization threshold for warnings (0.0-1.0)
            critical_threshold: Memory utilization threshold for critical alerts
            total_memory_gb: Override total memory detection (for testing)
            on_warning: Callback when warning threshold exceeded
            on_critical: Callback when critical threshold exceeded
        """
        if not 0 < warning_threshold < critical_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < warning < critical <= 1.0, "
                f"got warning={warning_threshold}, critical={critical_threshold}"
            )

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.on_warning = on_warning
        self.on_critical = on_critical

        # Detect total memory
        if total_memory_gb is not None:
            self._total_memory_bytes = int(total_memory_gb * (1024**3))
        else:
            self._total_memory_bytes = self._detect_total_memory()

        # Statistics tracking
        self._check_count = 0
        self._warning_count = 0
        self._critical_count = 0
        self._peak_utilization = 0.0

    @staticmethod
    def _detect_total_memory() -> int:
        """Detect total system memory in bytes."""
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip())
        except Exception:  # noqa: BLE001 - Intentional: fallback for memory detection
            # Fallback to 512GB (Mac Studio M3 Ultra default)
            return 512 * (1024**3)

    def check(self) -> MemorySnapshot:
        """Take a memory snapshot and evaluate status.

        Returns:
            MemorySnapshot with current memory state and status.
        """
        self._check_count += 1

        # Get MLX memory stats
        active = mx.metal.get_active_memory()
        peak = mx.metal.get_peak_memory()
        cache = mx.metal.get_cache_memory()

        utilization = active / self._total_memory_bytes
        self._peak_utilization = max(self._peak_utilization, utilization)

        # Determine status
        if utilization >= self.critical_threshold:
            status = "critical"
            self._critical_count += 1
        elif utilization >= self.warning_threshold:
            status = "warning"
            self._warning_count += 1
        else:
            status = "ok"

        snapshot = MemorySnapshot(
            active_bytes=active,
            peak_bytes=peak,
            cache_bytes=cache,
            total_available_bytes=self._total_memory_bytes,
            utilization=utilization,
            status=status,
        )

        # Fire callbacks
        if status == "critical" and self.on_critical:
            self.on_critical(snapshot)
        elif status == "warning" and self.on_warning:
            self.on_warning(snapshot)

        return snapshot

    def suggest_batch_size_reduction(self, current_batch_size: int) -> int:
        """Suggest reduced batch size based on current memory pressure.

        Uses proportional reduction: if memory is at X% utilization and
        critical threshold is Y%, reduce batch by (X - target) / (Y - target)
        where target is slightly below warning threshold.

        Args:
            current_batch_size: Current training batch size

        Returns:
            Suggested new batch size (minimum 1)
        """
        if current_batch_size <= 1:
            return 1

        snapshot = self.check()
        target_utilization = self.warning_threshold * 0.9  # 10% below warning

        if snapshot.utilization <= target_utilization:
            return current_batch_size

        # Calculate reduction factor
        # Higher utilization = more aggressive reduction
        excess_utilization = snapshot.utilization - target_utilization
        max_excess = self.critical_threshold - target_utilization

        if max_excess <= 0:
            return 1

        reduction_factor = min(1.0, excess_utilization / max_excess)
        new_batch = int(current_batch_size * (1.0 - reduction_factor * 0.5))

        return max(1, new_batch)

    def clear_cache(self) -> int:
        """Clear MLX memory cache to reclaim memory.

        Returns:
            Bytes freed from cache.
        """
        before = mx.metal.get_cache_memory()
        mx.metal.clear_cache()
        after = mx.metal.get_cache_memory()
        freed = before - after

        if freed > 0:
            logger.info(f"Cleared {freed / (1024**3):.2f} GB from MLX cache")

        return freed

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        mx.metal.reset_peak_memory()
        self._peak_utilization = 0.0

    def get_stats(self) -> dict:
        """Get monitoring statistics.

        Returns:
            Dictionary with check counts, warnings, and peak utilization.
        """
        return {
            "check_count": self._check_count,
            "warning_count": self._warning_count,
            "critical_count": self._critical_count,
            "peak_utilization": self._peak_utilization,
            "total_memory_gb": self._total_memory_bytes / (1024**3),
        }

    def log_status(self) -> None:
        """Log current memory status."""
        snapshot = self.check()
        logger.info(
            f"Memory: {snapshot.active_gb:.1f}GB / {snapshot.available_gb:.1f}GB "
            f"({snapshot.utilization * 100:.1f}%) [{snapshot.status}]"
        )


def create_memory_monitor(
    enabled: bool = True,
    warning_threshold: float = MemoryMonitor.DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = MemoryMonitor.DEFAULT_CRITICAL_THRESHOLD,
) -> MemoryMonitor | None:
    """Factory function to create memory monitor.

    Args:
        enabled: Whether to create a real monitor (False returns None)
        warning_threshold: Warning threshold (0.0-1.0)
        critical_threshold: Critical threshold (0.0-1.0)

    Returns:
        MemoryMonitor instance or None if disabled.
    """
    if not enabled:
        return None

    return MemoryMonitor(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
