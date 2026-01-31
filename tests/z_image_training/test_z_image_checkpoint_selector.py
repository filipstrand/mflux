"""Tests for Z-Image best checkpoint selection functionality.

Best checkpoint selection automatically keeps the N best checkpoints
by validation metrics (loss, CLIP score) and deletes worse ones.
"""

from pathlib import Path

import pytest

from mflux.models.z_image.variants.training.state.checkpoint_selector import (
    BestCheckpointSelector,
    CheckpointRecord,
    create_checkpoint_selector,
)
from mflux.models.z_image.variants.training.state.training_spec import SaveSpec


class TestCheckpointRecord:
    """Tests for CheckpointRecord dataclass."""

    def test_basic_record(self):
        """Test creating a basic checkpoint record."""
        record = CheckpointRecord(
            step=1000,
            path=Path("/checkpoints/step_1000.zip"),
            validation_loss=0.05,
        )

        assert record.step == 1000
        assert record.validation_loss == 0.05
        assert record.clip_score is None

    def test_get_metric_validation_loss(self):
        """Test getting validation_loss metric."""
        record = CheckpointRecord(
            step=1000,
            path=Path("/test.zip"),
            validation_loss=0.05,
        )

        assert record.get_metric("validation_loss") == 0.05

    def test_get_metric_clip_score(self):
        """Test getting clip_score metric."""
        record = CheckpointRecord(
            step=1000,
            path=Path("/test.zip"),
            clip_score=0.85,
        )

        assert record.get_metric("clip_score") == 0.85

    def test_get_metric_none(self):
        """Test getting metric that's not set."""
        record = CheckpointRecord(
            step=1000,
            path=Path("/test.zip"),
        )

        assert record.get_metric("validation_loss") is None
        assert record.get_metric("unknown_metric") is None


class TestBestCheckpointSelector:
    """Tests for BestCheckpointSelector."""

    def test_init_default(self):
        """Test default initialization."""
        selector = BestCheckpointSelector(keep_best_n=3)

        assert selector.keep_best_n == 3
        assert selector.metric == "validation_loss"
        assert selector.higher_is_better is False
        assert len(selector.checkpoints) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        selector = BestCheckpointSelector(
            keep_best_n=5,
            metric="clip_score",
            higher_is_better=True,
        )

        assert selector.keep_best_n == 5
        assert selector.metric == "clip_score"
        assert selector.higher_is_better is True

    def test_init_invalid_keep_best_n(self):
        """Test that keep_best_n < 1 raises error."""
        with pytest.raises(ValueError, match="keep_best_n must be >= 1"):
            BestCheckpointSelector(keep_best_n=0)

    def test_init_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            BestCheckpointSelector(keep_best_n=3, metric="invalid")

    def test_record_single_checkpoint(self):
        """Test recording a single checkpoint."""
        selector = BestCheckpointSelector(keep_best_n=3)

        to_delete = selector.record_checkpoint(
            step=1000,
            path=Path("/checkpoints/step_1000.zip"),
            validation_loss=0.05,
        )

        assert len(to_delete) == 0
        assert len(selector.checkpoints) == 1

    def test_record_within_limit(self):
        """Test recording checkpoints within keep_best_n limit."""
        selector = BestCheckpointSelector(keep_best_n=3)

        for i in range(3):
            to_delete = selector.record_checkpoint(
                step=i * 1000,
                path=Path(f"/checkpoints/step_{i * 1000}.zip"),
                validation_loss=0.05 - i * 0.01,  # Improving loss
            )
            assert len(to_delete) == 0

        assert len(selector.checkpoints) == 3

    def test_record_beyond_limit_deletes_worst(self):
        """Test that worst checkpoint is deleted when exceeding limit."""
        selector = BestCheckpointSelector(
            keep_best_n=2,
            metric="validation_loss",
            higher_is_better=False,
        )

        # Add checkpoints with different losses
        selector.record_checkpoint(
            step=1000,
            path=Path("/checkpoints/step_1000.zip"),
            validation_loss=0.10,  # Worst
        )
        selector.record_checkpoint(
            step=2000,
            path=Path("/checkpoints/step_2000.zip"),
            validation_loss=0.05,  # Best
        )

        # Third checkpoint should trigger deletion of worst
        to_delete = selector.record_checkpoint(
            step=3000,
            path=Path("/checkpoints/step_3000.zip"),
            validation_loss=0.07,  # Middle
        )

        assert len(to_delete) == 1
        assert to_delete[0] == Path("/checkpoints/step_1000.zip")  # Worst deleted
        assert len(selector.checkpoints) == 2

    def test_record_higher_is_better(self):
        """Test checkpoint selection with higher_is_better=True."""
        selector = BestCheckpointSelector(
            keep_best_n=2,
            metric="clip_score",
            higher_is_better=True,
        )

        selector.record_checkpoint(step=1000, path=Path("/a.zip"), clip_score=0.70)
        selector.record_checkpoint(step=2000, path=Path("/b.zip"), clip_score=0.90)

        to_delete = selector.record_checkpoint(step=3000, path=Path("/c.zip"), clip_score=0.80)

        # Should delete lowest clip score
        assert len(to_delete) == 1
        assert to_delete[0] == Path("/a.zip")

    def test_get_best_checkpoint(self):
        """Test getting the best checkpoint."""
        selector = BestCheckpointSelector(keep_best_n=5)

        selector.record_checkpoint(step=1000, path=Path("/a.zip"), validation_loss=0.10)
        selector.record_checkpoint(step=2000, path=Path("/b.zip"), validation_loss=0.05)
        selector.record_checkpoint(step=3000, path=Path("/c.zip"), validation_loss=0.08)

        best = selector.get_best_checkpoint()

        assert best is not None
        assert best.path == Path("/b.zip")
        assert best.validation_loss == 0.05

    def test_get_best_checkpoint_empty(self):
        """Test getting best checkpoint when empty."""
        selector = BestCheckpointSelector(keep_best_n=3)

        assert selector.get_best_checkpoint() is None

    def test_handles_none_metrics(self):
        """Test that None metrics are handled (deprioritized)."""
        selector = BestCheckpointSelector(keep_best_n=2)

        selector.record_checkpoint(step=1000, path=Path("/a.zip"), validation_loss=None)
        selector.record_checkpoint(step=2000, path=Path("/b.zip"), validation_loss=0.05)

        # Add third checkpoint - should delete the one with None
        to_delete = selector.record_checkpoint(step=3000, path=Path("/c.zip"), validation_loss=0.08)

        assert Path("/a.zip") in to_delete


class TestCheckpointSelectorSerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict(self):
        """Test serializing selector to dict."""
        selector = BestCheckpointSelector(keep_best_n=3, metric="validation_loss")
        selector.record_checkpoint(
            step=1000,
            path=Path("/test.zip"),
            validation_loss=0.05,
        )

        data = selector.to_dict()

        assert data["keep_best_n"] == 3
        assert data["metric"] == "validation_loss"
        assert data["higher_is_better"] is False
        assert len(data["checkpoints"]) == 1
        assert data["checkpoints"][0]["step"] == 1000

    def test_from_dict(self):
        """Test restoring selector from dict."""
        data = {
            "keep_best_n": 5,
            "metric": "clip_score",
            "higher_is_better": True,
            "checkpoints": [
                {
                    "step": 1000,
                    "path": "/test.zip",
                    "validation_loss": None,
                    "clip_score": 0.85,
                    "timestamp": None,
                }
            ],
        }

        selector = BestCheckpointSelector.from_dict(data)

        assert selector.keep_best_n == 5
        assert selector.metric == "clip_score"
        assert selector.higher_is_better is True
        assert len(selector.checkpoints) == 1
        assert selector.checkpoints[0].clip_score == 0.85

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = BestCheckpointSelector(keep_best_n=3)
        original.record_checkpoint(step=1000, path=Path("/a.zip"), validation_loss=0.1)
        original.record_checkpoint(step=2000, path=Path("/b.zip"), validation_loss=0.05)

        data = original.to_dict()
        restored = BestCheckpointSelector.from_dict(data)

        assert restored.keep_best_n == original.keep_best_n
        assert restored.metric == original.metric
        assert len(restored.checkpoints) == len(original.checkpoints)


class TestCreateCheckpointSelector:
    """Tests for factory function."""

    def test_create_enabled(self):
        """Test creating enabled selector."""
        selector = create_checkpoint_selector(keep_best_n=3)

        assert selector is not None
        assert selector.keep_best_n == 3

    def test_create_disabled_none(self):
        """Test that None returns None."""
        selector = create_checkpoint_selector(keep_best_n=None)
        assert selector is None

    def test_create_disabled_zero(self):
        """Test that 0 returns None."""
        selector = create_checkpoint_selector(keep_best_n=0)
        assert selector is None

    def test_create_auto_detect_loss(self):
        """Test auto-detect higher_is_better for loss."""
        selector = create_checkpoint_selector(keep_best_n=3, metric="validation_loss")

        assert selector.higher_is_better is False  # Lower loss is better

    def test_create_auto_detect_clip(self):
        """Test auto-detect higher_is_better for CLIP score."""
        selector = create_checkpoint_selector(keep_best_n=3, metric="clip_score")

        assert selector.higher_is_better is True  # Higher CLIP is better


class TestSaveSpecIntegration:
    """Tests for SaveSpec integration."""

    def test_save_spec_with_best_checkpoint(self):
        """Test SaveSpec with best checkpoint settings."""
        spec = SaveSpec(
            checkpoint_frequency=100,
            output_path="output",  # Use relative path
            keep_best_n_checkpoints=3,
            best_checkpoint_metric="validation_loss",
        )

        assert spec.keep_best_n_checkpoints == 3
        assert spec.best_checkpoint_metric == "validation_loss"

    def test_save_spec_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="best_checkpoint_metric must be"):
            SaveSpec(
                checkpoint_frequency=100,
                output_path="output",  # Use relative path
                best_checkpoint_metric="invalid_metric",
            )

    def test_save_spec_default_values(self):
        """Test default values for checkpoint selection."""
        spec = SaveSpec(checkpoint_frequency=100, output_path="output")  # Use relative path

        assert spec.keep_best_n_checkpoints == 0  # Disabled by default
        assert spec.best_checkpoint_metric == "validation_loss"
