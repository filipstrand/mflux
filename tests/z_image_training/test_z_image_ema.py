"""Tests for Z-Image EMA (Exponential Moving Average) implementation.

Tests that EMA:
- Properly initializes shadow weights
- Updates shadow weights correctly
- Applies/restores weights correctly
- Handles state serialization
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.z_image.variants.training.optimization.ema import (
    EMAModel,
    NoOpEMA,
    create_ema,
)


class SimpleModel(nn.Module):
    """Simple model for testing EMA."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def __call__(self, x):
        return self.linear(x)


@pytest.mark.fast
def test_ema_initialization():
    """Test that EMA initializes shadow weights from model."""
    model = SimpleModel()
    ema = EMAModel(model, decay=0.9999)

    # Shadow weights should be copies of model weights
    model_flat = tree_flatten(model.parameters())
    shadow_flat = tree_flatten(ema.shadow)

    assert len(model_flat) == len(shadow_flat)

    for (name, model_p), (_, shadow_p) in zip(model_flat, shadow_flat):
        np.testing.assert_array_almost_equal(np.array(model_p), np.array(shadow_p), decimal=5)


@pytest.mark.fast
def test_ema_update():
    """Test that EMA update follows correct formula."""
    model = SimpleModel()
    decay = 0.9

    # Initialize EMA
    ema = EMAModel(model, decay=decay)

    # Get initial shadow weights (flat list)
    initial_shadow_flat = tree_flatten(ema.shadow)
    initial_shadow = {k: v * 1.0 for k, v in initial_shadow_flat}
    # Force materialization (mx.eval is MLX's lazy computation trigger)
    mx.eval(initial_shadow)

    # Modify model weights - add ones to each parameter
    model.linear.weight = model.linear.weight + mx.ones_like(model.linear.weight)
    model.linear.bias = model.linear.bias + mx.ones_like(model.linear.bias)
    mx.eval(model.parameters())

    # Update EMA
    ema.update(model)

    # Check update formula: shadow = decay * old_shadow + (1 - decay) * new_weights
    model_flat = tree_flatten(model.parameters())
    shadow_flat = tree_flatten(ema.shadow)

    for (name, new_weight), (_, actual) in zip(model_flat, shadow_flat):
        old_shadow = initial_shadow.get(name)
        if old_shadow is None:
            continue
        expected = decay * old_shadow + (1.0 - decay) * new_weight

        np.testing.assert_array_almost_equal(np.array(actual), np.array(expected), decimal=5)


@pytest.mark.fast
def test_ema_apply_and_restore():
    """Test that apply_shadow and restore work correctly."""
    from mlx.utils import tree_map

    model = SimpleModel()
    ema = EMAModel(model, decay=0.999)

    # Get original weights (flatten for comparison)
    original_flat = tree_flatten(model.parameters())
    original_weights = {k: v * 1.0 for k, v in original_flat}
    mx.eval(original_weights)

    # Modify shadow weights to be different
    ema._shadow = tree_map(lambda v: v + mx.ones_like(v), ema._shadow)
    mx.eval(ema._shadow)

    # Apply shadow weights
    ema.apply_shadow(model)
    mx.eval(model.parameters())

    # Model should now have shadow weights
    model_flat = tree_flatten(model.parameters())
    shadow_flat = tree_flatten(ema.shadow)
    for (_, model_val), (_, shadow_val) in zip(model_flat, shadow_flat):
        np.testing.assert_array_almost_equal(np.array(model_val), np.array(shadow_val), decimal=5)

    # Restore original weights
    ema.restore(model)
    mx.eval(model.parameters())

    # Model should have original weights back
    model_flat = tree_flatten(model.parameters())
    for name, model_val in model_flat:
        original_val = original_weights[name]
        np.testing.assert_array_almost_equal(np.array(model_val), np.array(original_val), decimal=5)


@pytest.mark.fast
def test_ema_apply_without_restore_raises():
    """Test that calling apply_shadow twice without restore raises error."""
    model = SimpleModel()
    ema = EMAModel(model, decay=0.999)

    ema.apply_shadow(model)

    with pytest.raises(RuntimeError, match="backup exists"):
        ema.apply_shadow(model)


@pytest.mark.fast
def test_ema_restore_without_apply_raises():
    """Test that calling restore without apply_shadow raises error."""
    model = SimpleModel()
    ema = EMAModel(model, decay=0.999)

    with pytest.raises(RuntimeError, match="without prior"):
        ema.restore(model)


@pytest.mark.fast
def test_ema_copy_to():
    """Test that copy_to permanently copies shadow weights."""
    from mlx.utils import tree_map

    model = SimpleModel()
    ema = EMAModel(model, decay=0.999)

    # Modify shadow weights
    ema._shadow = tree_map(lambda v: v + mx.ones_like(v) * 5.0, ema._shadow)
    mx.eval(ema._shadow)

    # Copy to model permanently
    ema.copy_to(model)
    mx.eval(model.parameters())

    # Model should have shadow weights
    model_flat = tree_flatten(model.parameters())
    shadow_flat = tree_flatten(ema.shadow)
    for (_, model_val), (_, shadow_val) in zip(model_flat, shadow_flat):
        np.testing.assert_array_almost_equal(np.array(model_val), np.array(shadow_val), decimal=5)


@pytest.mark.fast
def test_ema_state_dict():
    """Test state_dict and load_state_dict."""
    model = SimpleModel()
    ema = EMAModel(model, decay=0.9999)

    # Save state
    state = ema.state_dict()

    # Create new EMA and load state
    ema2 = EMAModel(model, decay=0.5)  # Different decay
    ema2.load_state_dict(state)

    # Should have same decay
    assert ema2.decay == ema.decay

    # Should have same shadow weights
    shadow1_flat = tree_flatten(ema.shadow)
    shadow2_flat = tree_flatten(ema2.shadow)
    for (_, v1), (_, v2) in zip(shadow1_flat, shadow2_flat):
        np.testing.assert_array_almost_equal(np.array(v1), np.array(v2), decimal=5)


@pytest.mark.fast
def test_ema_save_and_load():
    """Test save and load functionality."""
    from mlx.utils import tree_map

    model = SimpleModel()
    ema = EMAModel(model, decay=0.9999)

    # Modify shadow
    ema._shadow = tree_map(lambda v: v + mx.ones_like(v), ema._shadow)
    mx.eval(ema._shadow)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ema.safetensors"

        # Save
        ema.save(path)

        # Load into new EMA
        ema2 = EMAModel.load(path, model, decay=0.9999)

        # Should have same shadow weights
        shadow1_flat = tree_flatten(ema.shadow)
        shadow2_flat = tree_flatten(ema2.shadow)
        for (name1, v1), (name2, v2) in zip(shadow1_flat, shadow2_flat):
            np.testing.assert_array_almost_equal(np.array(v1), np.array(v2), decimal=5)


@pytest.mark.fast
def test_ema_decay_validation():
    """Test that invalid decay values are rejected."""
    model = SimpleModel()

    with pytest.raises(ValueError, match="decay must be in"):
        EMAModel(model, decay=-0.1)

    with pytest.raises(ValueError, match="decay must be in"):
        EMAModel(model, decay=1.5)


@pytest.mark.fast
def test_noop_ema():
    """Test that NoOpEMA does nothing."""
    model = SimpleModel()
    original_flat = tree_flatten(model.parameters())
    original_weights = {k: v * 1.0 for k, v in original_flat}
    mx.eval(original_weights)

    noop = NoOpEMA(model, decay=0.999)

    # These should all be no-ops
    noop.update(model)
    noop.apply_shadow(model)
    noop.restore(model)
    noop.copy_to(model)

    # Model weights should be unchanged
    model_flat = tree_flatten(model.parameters())
    for name, model_val in model_flat:
        np.testing.assert_array_almost_equal(np.array(model_val), np.array(original_weights[name]), decimal=5)

    # State dict should be empty
    assert noop.state_dict() == {}


@pytest.mark.fast
def test_create_ema_factory():
    """Test create_ema factory function."""
    model = SimpleModel()

    # Enabled should return EMAModel
    ema = create_ema(model, enabled=True, decay=0.999)
    assert isinstance(ema, EMAModel)
    assert ema.decay == 0.999

    # Disabled should return NoOpEMA
    noop = create_ema(model, enabled=False, decay=0.999)
    assert isinstance(noop, NoOpEMA)


@pytest.mark.fast
def test_ema_spec_warns_unusual_decay():
    """Test that EMASpec warns about unusual decay values."""
    import warnings

    from mflux.models.z_image.variants.training.state.training_spec import EMASpec

    # Normal decay should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EMASpec(enabled=True, decay=0.9999)
        assert len(w) == 0

    # Very low decay should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EMASpec(enabled=True, decay=0.5)
        assert len(w) == 1
        assert "outside typical range" in str(w[0].message)

    # Very high decay should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EMASpec(enabled=True, decay=0.999999)
        assert len(w) == 1
        assert "outside typical range" in str(w[0].message)

    # Disabled EMA should not warn even with unusual decay
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EMASpec(enabled=False, decay=0.5)
        assert len(w) == 0
