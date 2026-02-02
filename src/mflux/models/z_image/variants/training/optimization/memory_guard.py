"""Memory guard that auto-pauses training when memory exceeds threshold.

Provides a hard memory limit safety net for Z-Image training on Apple Silicon.
When active memory exceeds the configured limit (default 340GB), training
automatically pauses, clears cache, and waits until memory drops below the
resume threshold (default 300GB) before continuing.

This is a safety mechanism that preserves maximum training speed while
preventing OOM crashes. Unlike activation checkpointing which trades speed
for memory, the guard only intervenes when actually needed.

Usage:
    guard = MemoryGuard(config)

    for batch in batches:
        guard.check_and_wait()  # Pauses if needed
        loss, grads = train_step(batch)
"""

import logging
import time
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


class MemoryGuardTimeoutError(RuntimeError):
    """Raised when memory guard times out waiting for memory to be freed."""

    pass


@dataclass
class MemoryGuardConfig:
    """Configuration for memory guard.

    Attributes:
        hard_limit_gb: Memory threshold that triggers pause (default 340GB)
        resume_threshold_gb: Memory level to resume at (default 300GB)
        check_interval_steps: Check every N steps to minimize overhead (default 10)
        max_wait_seconds: Log warning if waiting longer than this (default 60)
        hard_timeout_seconds: Raise exception after this many seconds (default 300)
        poll_interval_seconds: How often to check memory while waiting (default 0.5)
        clear_cache_on_pause: Whether to clear MLX cache when pausing (default True)
    """

    hard_limit_gb: float = 340.0
    resume_threshold_gb: float = 300.0
    check_interval_steps: int = 10
    max_wait_seconds: float = 60.0
    hard_timeout_seconds: float = 300.0  # 5 minutes hard timeout
    poll_interval_seconds: float = 0.5
    clear_cache_on_pause: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.hard_limit_gb <= self.resume_threshold_gb:
            raise ValueError(
                f"hard_limit_gb ({self.hard_limit_gb}) must be greater than "
                f"resume_threshold_gb ({self.resume_threshold_gb})"
            )
        if self.resume_threshold_gb <= 0:
            raise ValueError(f"resume_threshold_gb must be positive, got {self.resume_threshold_gb}")
        if self.check_interval_steps <= 0:
            raise ValueError(f"check_interval_steps must be positive, got {self.check_interval_steps}")
        if self.hard_timeout_seconds <= self.max_wait_seconds:
            raise ValueError(
                f"hard_timeout_seconds ({self.hard_timeout_seconds}) must be greater than "
                f"max_wait_seconds ({self.max_wait_seconds})"
            )


class MemoryGuard:
    """Guards against OOM by pausing training when memory limit exceeded.

    The guard checks memory periodically (every check_interval_steps) and
    pauses training if active memory exceeds hard_limit_gb. It waits until
    memory drops below resume_threshold_gb before allowing training to continue.

    This provides a hard safety net while preserving maximum training speed:
    - No overhead when memory is within limits
    - Minimal check overhead (every 10 steps by default)
    - Automatic recovery when memory pressure subsides
    - Hard timeout to prevent infinite hangs (raises MemoryGuardTimeoutError)

    Usage:
        guard = MemoryGuard(config)

        for batch in batches:
            guard.check_and_wait()  # Pauses if needed, raises on timeout
            loss, grads = train_step(batch)

        # Log statistics at end of training
        stats = guard.get_stats()
        if stats["pause_count"] > 0:
            print(f"Training paused {stats['pause_count']} times")

    Raises:
        MemoryGuardTimeoutError: If memory doesn't drop below resume threshold
            within hard_timeout_seconds.
    """

    def __init__(self, config: MemoryGuardConfig | None = None):
        """Initialize memory guard.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or MemoryGuardConfig()
        self._pause_count = 0
        self._total_wait_time = 0.0
        self._step_count = 0

    @staticmethod
    def _bytes_to_gb(bytes_value: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_value / (1024**3)

    def check_and_wait(self) -> bool:
        """Check memory and pause if over limit.

        Call this at the start of each training step. The check is performed
        every check_interval_steps to minimize overhead.

        Returns:
            True if training was paused and resumed, False if no pause was needed.

        Raises:
            MemoryGuardTimeoutError: If memory doesn't drop below resume threshold
                within hard_timeout_seconds.
        """
        self._step_count += 1

        if self._step_count % self.config.check_interval_steps != 0:
            return False

        active_gb = self._bytes_to_gb(mx.get_active_memory())

        if active_gb < self.config.hard_limit_gb:
            return False

        # Over limit - pause and wait
        self._pause_count += 1
        logger.warning(
            f"Memory guard triggered: {active_gb:.1f}GB > {self.config.hard_limit_gb:.0f}GB limit. "
            f"Pausing until memory drops below {self.config.resume_threshold_gb:.0f}GB..."
        )

        if self.config.clear_cache_on_pause:
            mx.clear_cache()
            mx.synchronize()

        wait_start = time.time()
        warning_logged = False

        while True:
            time.sleep(self.config.poll_interval_seconds)
            active_gb = self._bytes_to_gb(mx.get_active_memory())

            if active_gb < self.config.resume_threshold_gb:
                wait_time = time.time() - wait_start
                self._total_wait_time += wait_time
                logger.info(f"Memory guard: Resumed after {wait_time:.1f}s (now at {active_gb:.1f}GB)")
                return True

            elapsed = time.time() - wait_start

            # Hard timeout - raise exception to prevent infinite hang
            if elapsed > self.config.hard_timeout_seconds:
                raise MemoryGuardTimeoutError(
                    f"Memory guard timed out after {elapsed:.0f}s. "
                    f"Memory at {active_gb:.1f}GB, threshold is {self.config.resume_threshold_gb:.0f}GB. "
                    f"Consider reducing batch size, enabling activation checkpointing, or checking for memory leaks."
                )

            # Warning after max_wait_seconds (but continue waiting up to hard timeout)
            if elapsed > self.config.max_wait_seconds and not warning_logged:
                logger.error(
                    f"Memory guard: Waited {elapsed:.0f}s but memory still at {active_gb:.1f}GB. "
                    f"Will timeout after {self.config.hard_timeout_seconds:.0f}s total. "
                    f"Consider reducing batch size or enabling activation checkpointing."
                )
                warning_logged = True

    def get_stats(self) -> dict:
        """Get guard statistics.

        Returns:
            Dictionary with pause count, total wait time, and steps checked.
        """
        return {
            "pause_count": self._pause_count,
            "total_wait_time_seconds": self._total_wait_time,
            "steps_checked": self._step_count,
        }


def create_memory_guard(
    enabled: bool = True,
    hard_limit_gb: float = 340.0,
    resume_threshold_gb: float = 300.0,
) -> MemoryGuard | None:
    """Factory function to create memory guard.

    Args:
        enabled: Whether to create a guard (False returns None)
        hard_limit_gb: Memory threshold that triggers pause
        resume_threshold_gb: Memory level to resume at

    Returns:
        MemoryGuard instance or None if disabled.
    """
    if not enabled:
        return None
    return MemoryGuard(
        MemoryGuardConfig(
            hard_limit_gb=hard_limit_gb,
            resume_threshold_gb=resume_threshold_gb,
        )
    )
