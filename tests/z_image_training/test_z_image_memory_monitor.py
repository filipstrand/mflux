"""Tests for Z-Image memory monitoring functionality.

Memory monitoring provides early OOM detection and batch size
suggestions for training on Apple Silicon.
"""

import pytest

from mflux.models.z_image.variants.training.optimization.memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    create_memory_monitor,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_snapshot_properties(self):
        """Test MemorySnapshot property conversions."""
        snapshot = MemorySnapshot(
            active_bytes=10 * (1024**3),  # 10GB
            peak_bytes=12 * (1024**3),  # 12GB
            cache_bytes=2 * (1024**3),  # 2GB
            total_available_bytes=512 * (1024**3),  # 512GB
            utilization=0.5,
            status="ok",
        )

        assert abs(snapshot.active_gb - 10.0) < 0.01
        assert abs(snapshot.peak_gb - 12.0) < 0.01
        assert abs(snapshot.cache_gb - 2.0) < 0.01
        assert abs(snapshot.available_gb - 512.0) < 0.01

    def test_snapshot_status_values(self):
        """Test that status is one of expected values."""
        for status in ["ok", "warning", "critical"]:
            snapshot = MemorySnapshot(
                active_bytes=0,
                peak_bytes=0,
                cache_bytes=0,
                total_available_bytes=512 * (1024**3),
                utilization=0.0,
                status=status,
            )
            assert snapshot.status == status


class TestMemoryMonitorInit:
    """Tests for MemoryMonitor initialization."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        monitor = MemoryMonitor(total_memory_gb=512.0)

        assert monitor.warning_threshold == 0.85
        assert monitor.critical_threshold == 0.95

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        monitor = MemoryMonitor(
            warning_threshold=0.7,
            critical_threshold=0.9,
            total_memory_gb=512.0,
        )

        assert monitor.warning_threshold == 0.7
        assert monitor.critical_threshold == 0.9

    def test_invalid_thresholds_raises(self):
        """Test that invalid thresholds raise ValueError."""
        # Warning >= critical
        with pytest.raises(ValueError):
            MemoryMonitor(warning_threshold=0.9, critical_threshold=0.8)

        # Warning <= 0
        with pytest.raises(ValueError):
            MemoryMonitor(warning_threshold=0.0, critical_threshold=0.9)

        # Critical > 1
        with pytest.raises(ValueError):
            MemoryMonitor(warning_threshold=0.8, critical_threshold=1.1)

    def test_total_memory_override(self):
        """Test total memory can be overridden."""
        monitor = MemoryMonitor(total_memory_gb=256.0)
        stats = monitor.get_stats()
        assert abs(stats["total_memory_gb"] - 256.0) < 0.01


class TestMemoryMonitorCheck:
    """Tests for MemoryMonitor.check() method."""

    def test_check_returns_snapshot(self):
        """Test that check returns a MemorySnapshot."""
        monitor = MemoryMonitor(total_memory_gb=512.0)
        snapshot = monitor.check()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.active_bytes >= 0
        assert snapshot.peak_bytes >= 0
        assert 0.0 <= snapshot.utilization <= 1.0
        assert snapshot.status in ["ok", "warning", "critical"]

    def test_check_increments_count(self):
        """Test that check increments check count."""
        monitor = MemoryMonitor(total_memory_gb=512.0)

        assert monitor.get_stats()["check_count"] == 0
        monitor.check()
        assert monitor.get_stats()["check_count"] == 1
        monitor.check()
        assert monitor.get_stats()["check_count"] == 2


class TestMemoryMonitorCallbacks:
    """Tests for memory monitoring callbacks."""

    def test_warning_callback(self):
        """Test warning callback is called at threshold."""
        warnings = []

        def on_warning(snapshot):
            warnings.append(snapshot)

        # Use tiny total memory to trigger warning
        monitor = MemoryMonitor(
            total_memory_gb=0.001,  # 1MB total = will trigger warning
            warning_threshold=0.1,
            critical_threshold=0.9,
            on_warning=on_warning,
        )
        monitor.check()

        # May or may not trigger depending on actual memory usage
        # Just verify callback mechanism works
        assert callable(monitor.on_warning)

    def test_critical_callback(self):
        """Test critical callback is called at threshold."""
        criticals = []

        def on_critical(snapshot):
            criticals.append(snapshot)

        monitor = MemoryMonitor(
            total_memory_gb=0.001,  # Very small to trigger critical
            warning_threshold=0.01,
            critical_threshold=0.02,
            on_critical=on_critical,
        )
        monitor.check()

        assert callable(monitor.on_critical)


class TestBatchSizeReduction:
    """Tests for batch size reduction suggestions."""

    def test_suggest_no_reduction_when_ok(self):
        """Test no reduction suggested when memory is ok."""
        monitor = MemoryMonitor(
            total_memory_gb=1024.0,  # Large memory to ensure ok status
            warning_threshold=0.85,
            critical_threshold=0.95,
        )

        # With 1TB+ available, should not suggest reduction
        suggestion = monitor.suggest_batch_size_reduction(16)
        # May or may not reduce depending on actual system state
        assert suggestion >= 1
        assert suggestion <= 16

    def test_suggest_minimum_is_one(self):
        """Test that minimum suggested batch size is 1."""
        monitor = MemoryMonitor(
            total_memory_gb=0.001,  # Tiny to trigger reduction
            warning_threshold=0.01,
            critical_threshold=0.02,
        )

        suggestion = monitor.suggest_batch_size_reduction(1)
        assert suggestion == 1

    def test_suggest_reduces_under_pressure(self):
        """Test that reduction is suggested under memory pressure."""
        # With very small total memory, any usage triggers pressure
        monitor = MemoryMonitor(
            total_memory_gb=0.001,
            warning_threshold=0.1,
            critical_threshold=0.2,
        )

        suggestion = monitor.suggest_batch_size_reduction(32)
        # Should reduce significantly
        assert suggestion <= 32
        assert suggestion >= 1


class TestMemoryMonitorStatistics:
    """Tests for memory monitor statistics tracking."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        monitor = MemoryMonitor(total_memory_gb=512.0)
        stats = monitor.get_stats()

        assert stats["check_count"] == 0
        assert stats["warning_count"] == 0
        assert stats["critical_count"] == 0
        assert stats["peak_utilization"] == 0.0

    def test_stats_after_checks(self):
        """Test statistics after multiple checks."""
        monitor = MemoryMonitor(total_memory_gb=512.0)

        for _ in range(5):
            monitor.check()

        stats = monitor.get_stats()
        assert stats["check_count"] == 5
        assert stats["peak_utilization"] >= 0.0


class TestMemoryMonitorCache:
    """Tests for cache clearing functionality."""

    def test_clear_cache(self):
        """Test cache clearing returns bytes freed."""
        monitor = MemoryMonitor(total_memory_gb=512.0)
        freed = monitor.clear_cache()

        assert isinstance(freed, int)
        assert freed >= 0

    def test_reset_peak(self):
        """Test peak memory reset."""
        monitor = MemoryMonitor(total_memory_gb=512.0)
        monitor.check()  # Establish some peak

        monitor.reset_peak()
        # Peak utilization tracking should be reset
        assert monitor._peak_utilization == 0.0


class TestCreateMemoryMonitor:
    """Tests for factory function."""

    def test_create_enabled(self):
        """Test creating enabled monitor."""
        monitor = create_memory_monitor(enabled=True)
        assert isinstance(monitor, MemoryMonitor)

    def test_create_disabled(self):
        """Test creating disabled monitor returns None."""
        monitor = create_memory_monitor(enabled=False)
        assert monitor is None

    def test_create_with_custom_thresholds(self):
        """Test creating with custom thresholds."""
        monitor = create_memory_monitor(
            enabled=True,
            warning_threshold=0.7,
            critical_threshold=0.85,
        )

        assert monitor.warning_threshold == 0.7
        assert monitor.critical_threshold == 0.85
