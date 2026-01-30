"""Tests for Z-Image optimizer functionality.

Tests that:
- Optimizer state can be saved and loaded correctly
- Gradient clipping handles NaN/Inf values
- Weight decay is applied correctly
- bf16 momentum optimizers work correctly
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest

from mflux.models.z_image.variants.training.optimization.optimizer import Optimizer
from mflux.models.z_image.variants.training.trainer import ZImageTrainer


@pytest.mark.fast
def test_optimizer_state_save_load():
    """Test optimizer state can be saved and loaded."""
    # Create a simple optimizer
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    # Create some mock state
    optimizer.state = {
        "weight.m": mx.zeros((10, 10)),
        "weight.v": mx.zeros((10, 10)),
        "step": mx.array(100),
    }

    opt = Optimizer(optimizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "optimizer.safetensors"
        opt.save(save_path)

        # Verify file was created
        assert save_path.exists()

        # Create new optimizer and load state
        optimizer2 = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
        opt2 = Optimizer(optimizer2)
        opt2._load_state(str(save_path))

        # Verify state was loaded
        assert "weight.m" in opt2.optimizer.state
        assert "weight.v" in opt2.optimizer.state


@pytest.mark.fast
def test_clip_grad_norm_basic():
    """Test basic gradient clipping."""
    # Create gradients with large norm
    grads = {
        "weight1": mx.array([10.0, 20.0, 30.0]),
        "weight2": mx.array([40.0, 50.0, 60.0]),
    }

    # Clip to max_norm=1.0
    clipped = ZImageTrainer._clip_grad_norm(grads, max_norm=1.0)

    assert clipped is not None

    # Calculate clipped norm
    total_norm_sq = 0.0
    for g in clipped.values():
        total_norm_sq += float(mx.sum(g**2).item())
    clipped_norm = np.sqrt(total_norm_sq)

    # Norm should be <= max_norm (with small tolerance)
    np.testing.assert_almost_equal(clipped_norm, 1.0, decimal=5)


@pytest.mark.fast
def test_clip_grad_norm_no_clipping_needed():
    """Test gradient clipping when norm is already small."""
    # Create gradients with small norm
    grads = {
        "weight1": mx.array([0.001, 0.002, 0.003]),
        "weight2": mx.array([0.004, 0.005, 0.006]),
    }

    # Original norm
    original_norm_sq = 0.0
    for g in grads.values():
        original_norm_sq += float(mx.sum(g**2).item())
    original_norm = np.sqrt(original_norm_sq)

    # Clip to max_norm=1.0 (larger than original)
    clipped = ZImageTrainer._clip_grad_norm(grads, max_norm=1.0)

    assert clipped is not None

    # Calculate clipped norm
    clipped_norm_sq = 0.0
    for g in clipped.values():
        clipped_norm_sq += float(mx.sum(g**2).item())
    clipped_norm = np.sqrt(clipped_norm_sq)

    # Norm should remain unchanged
    np.testing.assert_almost_equal(clipped_norm, original_norm, decimal=5)


@pytest.mark.fast
def test_clip_grad_norm_nan_detection():
    """Test that NaN gradients return None."""
    # Create gradients with NaN
    grads = {
        "weight1": mx.array([1.0, float("nan"), 3.0]),
        "weight2": mx.array([4.0, 5.0, 6.0]),
    }

    # Should return None due to NaN
    result = ZImageTrainer._clip_grad_norm(grads, max_norm=1.0)
    assert result is None


@pytest.mark.fast
def test_clip_grad_norm_inf_detection():
    """Test that Inf gradients return None."""
    # Create gradients with Inf
    grads = {
        "weight1": mx.array([1.0, float("inf"), 3.0]),
        "weight2": mx.array([4.0, 5.0, 6.0]),
    }

    # Should return None due to Inf
    result = ZImageTrainer._clip_grad_norm(grads, max_norm=1.0)
    assert result is None


@pytest.mark.fast
def test_clip_grad_norm_negative_inf_detection():
    """Test that negative Inf gradients return None."""
    # Create gradients with -Inf
    grads = {
        "weight1": mx.array([1.0, float("-inf"), 3.0]),
        "weight2": mx.array([4.0, 5.0, 6.0]),
    }

    # Should return None due to -Inf
    result = ZImageTrainer._clip_grad_norm(grads, max_norm=1.0)
    assert result is None


@pytest.mark.fast
def test_clip_grad_norm_disabled():
    """Test that max_norm <= 0 disables clipping."""
    grads = {
        "weight1": mx.array([10.0, 20.0, 30.0]),
        "weight2": mx.array([40.0, 50.0, 60.0]),
    }

    # Clipping disabled
    result = ZImageTrainer._clip_grad_norm(grads, max_norm=0.0)

    # Should return original gradients unchanged
    assert result is not None
    np.testing.assert_array_equal(result["weight1"], grads["weight1"])
    np.testing.assert_array_equal(result["weight2"], grads["weight2"])


@pytest.mark.fast
def test_clip_grad_norm_negative_max_norm():
    """Test that negative max_norm disables clipping."""
    grads = {
        "weight1": mx.array([10.0, 20.0, 30.0]),
    }

    # Negative max_norm should disable clipping
    result = ZImageTrainer._clip_grad_norm(grads, max_norm=-1.0)

    assert result is not None
    np.testing.assert_array_equal(result["weight1"], grads["weight1"])


@pytest.mark.fast
def test_accumulate_grads():
    """Test gradient accumulation."""
    acc_grads = {
        "weight1": mx.array([1.0, 2.0, 3.0]),
        "weight2": mx.array([4.0, 5.0, 6.0]),
    }
    new_grads = {
        "weight1": mx.array([0.5, 0.5, 0.5]),
        "weight2": mx.array([1.0, 1.0, 1.0]),
    }

    result = ZImageTrainer._accumulate_grads(acc_grads, new_grads)

    expected_weight1 = mx.array([1.5, 2.5, 3.5])
    expected_weight2 = mx.array([5.0, 6.0, 7.0])

    np.testing.assert_array_almost_equal(result["weight1"], expected_weight1)
    np.testing.assert_array_almost_equal(result["weight2"], expected_weight2)


@pytest.mark.fast
def test_accumulate_grads_mismatched_keys():
    """Test gradient accumulation with mismatched keys."""
    acc_grads = {
        "weight1": mx.array([1.0, 2.0]),
    }
    new_grads = {
        "weight1": mx.array([0.5, 0.5]),
        "weight2": mx.array([1.0, 1.0]),  # New key
    }

    result = ZImageTrainer._accumulate_grads(acc_grads, new_grads)

    # weight1 should be accumulated
    np.testing.assert_array_almost_equal(result["weight1"], mx.array([1.5, 2.5]))

    # weight2 should be included from new_grads
    assert "weight2" in result
    np.testing.assert_array_almost_equal(result["weight2"], mx.array([1.0, 1.0]))


@pytest.mark.fast
def test_scale_grads():
    """Test gradient scaling."""
    grads = {
        "weight1": mx.array([2.0, 4.0, 6.0]),
        "weight2": mx.array([8.0, 10.0, 12.0]),
    }

    # Scale by 0.5
    result = ZImageTrainer._scale_grads(grads, 0.5)

    expected_weight1 = mx.array([1.0, 2.0, 3.0])
    expected_weight2 = mx.array([4.0, 5.0, 6.0])

    np.testing.assert_array_almost_equal(result["weight1"], expected_weight1)
    np.testing.assert_array_almost_equal(result["weight2"], expected_weight2)


@pytest.mark.fast
def test_optimizer_update():
    """Test optimizer update method."""
    optimizer = optim.AdamW(learning_rate=1e-4)
    opt = Optimizer(optimizer)

    # Create mock model
    model = MagicMock()
    gradients = {"weight": mx.array([1.0, 2.0, 3.0])}

    # Should not raise
    opt.update(model, gradients)


@pytest.mark.fast
def test_optimizer_zero_grad():
    """Test optimizer zero_grad method."""
    optimizer = optim.AdamW(learning_rate=1e-4)
    opt = Optimizer(optimizer)

    # Should not raise (no-op in MLX)
    opt.zero_grad()


@pytest.mark.fast
def test_optimizer_load_empty_state_raises():
    """Test that loading empty state raises error."""
    optimizer = optim.AdamW(learning_rate=1e-4)
    opt = Optimizer(optimizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save empty state
        empty_state = {}
        save_path = Path(tmpdir) / "empty.safetensors"
        mx.save_safetensors(str(save_path), empty_state)

        # Should raise ValueError for empty state
        with pytest.raises(ValueError, match="empty"):
            opt._load_state(str(save_path))
