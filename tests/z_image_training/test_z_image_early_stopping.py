"""Tests for early stopping functionality in Z-Image training.

Tests that:
- EarlyStoppingSpec validates configuration properly
- EarlyStoppingState correctly tracks validation loss
- Early stopping triggers after patience exceeded
- Early stopping respects min_delta threshold
- State can be serialized and restored
"""

import pytest

from mflux.models.z_image.variants.training.state.training_spec import EarlyStoppingSpec
from mflux.models.z_image.variants.training.state.training_state import EarlyStoppingState


@pytest.mark.fast
class TestEarlyStoppingSpec:
    """Tests for EarlyStoppingSpec configuration."""

    def test_default_values(self):
        """Test default values for EarlyStoppingSpec."""
        spec = EarlyStoppingSpec()
        assert spec.enabled is False
        assert spec.patience == 5
        assert spec.min_delta == 0.0

    def test_custom_values(self):
        """Test custom values for EarlyStoppingSpec."""
        spec = EarlyStoppingSpec(enabled=True, patience=10, min_delta=0.001)
        assert spec.enabled is True
        assert spec.patience == 10
        assert spec.min_delta == 0.001

    def test_patience_validation(self):
        """Test that patience must be >= 1."""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStoppingSpec(patience=0)

        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStoppingSpec(patience=-1)

    def test_min_delta_validation(self):
        """Test that min_delta must be >= 0."""
        with pytest.raises(ValueError, match="min_delta must be >= 0"):
            EarlyStoppingSpec(min_delta=-0.1)


@pytest.mark.fast
class TestEarlyStoppingState:
    """Tests for EarlyStoppingState behavior."""

    def test_initialization(self):
        """Test EarlyStoppingState initialization."""
        state = EarlyStoppingState(patience=5, min_delta=0.001)
        assert state.patience == 5
        assert state.min_delta == 0.001
        assert state.best_val_loss == float("inf")
        assert state.patience_counter == 0
        assert state.should_stop is False

    def test_improvement_resets_counter(self):
        """Test that loss improvement resets patience counter."""
        state = EarlyStoppingState(patience=3, min_delta=0.0)

        # First validation - sets baseline
        state.check(1.0)
        assert state.best_val_loss == 1.0
        assert state.patience_counter == 0
        assert state.should_stop is False

        # No improvement - counter increments
        state.check(1.0)
        assert state.patience_counter == 1

        # Improvement - counter resets
        state.check(0.8)
        assert state.best_val_loss == 0.8
        assert state.patience_counter == 0

    def test_triggers_after_patience_exceeded(self):
        """Test that early stopping triggers after patience exceeded."""
        state = EarlyStoppingState(patience=3, min_delta=0.0)

        # Set baseline
        state.check(1.0)
        assert state.should_stop is False

        # Three validations without improvement
        state.check(1.0)  # patience_counter = 1
        assert state.should_stop is False

        state.check(1.1)  # patience_counter = 2
        assert state.should_stop is False

        state.check(1.0)  # patience_counter = 3 -> triggers
        assert state.should_stop is True

    def test_min_delta_threshold(self):
        """Test that min_delta threshold is respected."""
        state = EarlyStoppingState(patience=3, min_delta=0.01)

        # Set baseline
        state.check(1.0)
        assert state.best_val_loss == 1.0

        # Small improvement (less than min_delta) - not counted
        state.check(0.995)  # Only 0.005 improvement, less than 0.01
        assert state.best_val_loss == 1.0  # Not updated
        assert state.patience_counter == 1

        # Larger improvement (more than min_delta) - counted
        state.check(0.98)  # 0.02 improvement, more than 0.01
        assert state.best_val_loss == 0.98
        assert state.patience_counter == 0

    def test_check_returns_should_stop(self):
        """Test that check() returns the should_stop value."""
        state = EarlyStoppingState(patience=2, min_delta=0.0)

        result = state.check(1.0)
        assert result is False

        result = state.check(1.1)
        assert result is False

        result = state.check(1.2)
        assert result is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = EarlyStoppingState(patience=5, min_delta=0.001)
        state.check(0.5)
        state.check(0.6)  # No improvement

        data = state.to_dict()

        assert data["best_val_loss"] == 0.5
        assert data["patience_counter"] == 1
        assert data["should_stop"] is False

    def test_from_dict(self):
        """Test restoration from dictionary."""
        data = {
            "best_val_loss": 0.3,
            "patience_counter": 2,
            "should_stop": False,
        }

        state = EarlyStoppingState.from_dict(data, patience=5, min_delta=0.01)

        assert state.patience == 5
        assert state.min_delta == 0.01
        assert state.best_val_loss == 0.3
        assert state.patience_counter == 2
        assert state.should_stop is False

    def test_serialization_round_trip(self):
        """Test that serialization and deserialization preserves state."""
        original = EarlyStoppingState(patience=7, min_delta=0.005)
        original.check(1.0)
        original.check(0.9)
        original.check(0.95)  # No improvement

        data = original.to_dict()
        restored = EarlyStoppingState.from_dict(data, patience=7, min_delta=0.005)

        assert restored.best_val_loss == original.best_val_loss
        assert restored.patience_counter == original.patience_counter
        assert restored.should_stop == original.should_stop

        # Both should behave the same way
        original.check(0.96)
        restored.check(0.96)
        assert restored.patience_counter == original.patience_counter

    def test_from_dict_with_missing_keys(self):
        """Test restoration handles missing keys gracefully."""
        # Empty dict - uses defaults
        state = EarlyStoppingState.from_dict({}, patience=5, min_delta=0.0)
        assert state.best_val_loss == float("inf")
        assert state.patience_counter == 0
        assert state.should_stop is False

    def test_from_dict_with_invalid_values(self):
        """Test restoration handles invalid/corrupted values gracefully."""
        # Invalid best_val_loss (NaN)
        state = EarlyStoppingState.from_dict(
            {"best_val_loss": float("nan"), "patience_counter": 2, "should_stop": False},
            patience=5,
            min_delta=0.0,
        )
        assert state.best_val_loss == float("inf")  # Reset to default

        # Negative patience_counter
        state = EarlyStoppingState.from_dict(
            {"best_val_loss": 0.5, "patience_counter": -1, "should_stop": False},
            patience=5,
            min_delta=0.0,
        )
        assert state.patience_counter == 0  # Clamped to 0

        # patience_counter exceeds patience (clamped)
        state = EarlyStoppingState.from_dict(
            {"best_val_loss": 0.5, "patience_counter": 100, "should_stop": False},
            patience=5,
            min_delta=0.0,
        )
        assert state.patience_counter == 5  # Clamped to patience

        # Invalid type for should_stop (truthy value)
        state = EarlyStoppingState.from_dict(
            {"best_val_loss": 0.5, "patience_counter": 2, "should_stop": "yes"},
            patience=5,
            min_delta=0.0,
        )
        assert state.should_stop is True  # Converted to bool

    def test_does_not_trigger_when_improving(self):
        """Test that early stopping never triggers when loss keeps improving."""
        state = EarlyStoppingState(patience=3, min_delta=0.0)

        # Continuous improvement
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            result = state.check(loss)
            assert result is False
            assert state.patience_counter == 0

        assert state.best_val_loss == 0.3
        assert state.should_stop is False

    def test_once_triggered_stays_triggered(self):
        """Test that once should_stop is True, it stays True."""
        state = EarlyStoppingState(patience=2, min_delta=0.0)

        state.check(1.0)
        state.check(1.1)
        state.check(1.2)
        assert state.should_stop is True

        # Even with improvement, should_stop stays True
        state.check(0.5)
        assert state.should_stop is True
