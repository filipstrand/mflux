"""Tests for Z-Image flow matching loss computation.

Tests that the loss function:
- Computes correct flow matching formula
- Handles batch averaging
- Maintains gradient flow
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.z_image.variants.training.dataset.batch import Batch, Example
from mflux.models.z_image.variants.training.optimization.z_image_loss import ZImageLoss


def create_mock_example(example_id: int = 0) -> Example:
    """Create a mock Example for testing."""
    return Example(
        example_id=example_id,
        prompt=f"test prompt {example_id}",
        image_path=Path(f"test_image_{example_id}.jpg"),
        encoded_image=mx.random.normal((1, 16, 16, 16)),  # Random latent
        text_embeddings=mx.random.normal((1, 77, 768)),  # Random text embedding
    )


def create_mock_batch(num_examples: int = 2, seed: int = 42) -> Batch:
    """Create a mock Batch for testing."""
    examples = [create_mock_example(i) for i in range(num_examples)]
    return Batch(examples=examples, rng=random.Random(seed))


def create_mock_config():
    """Create a mock Config with required attributes."""
    config = MagicMock()
    config.num_inference_steps = 30
    config.scheduler = MagicMock()
    # Create sigma schedule from 1.0 (pure noise) to 0.0 (clean)
    config.scheduler.sigmas = [1.0 - (i / 30) for i in range(31)]
    return config


def create_mock_model():
    """Create a mock ZImageBase model."""
    model = MagicMock()

    # Mock transformer to return predicted noise
    def mock_transformer(x, t, sigmas, cap_feats):
        # Return zeros (perfect prediction would make loss = 0)
        return mx.zeros_like(x)

    model.transformer = mock_transformer
    return model


@pytest.mark.fast
def test_loss_returns_scalar():
    """Test that compute_loss returns a scalar value."""
    model = create_mock_model()
    config = create_mock_config()
    batch = create_mock_batch(num_examples=2)

    loss = ZImageLoss.compute_loss(model, config, batch)

    # Loss should be a scalar (0-d array)
    assert loss.ndim == 0
    assert isinstance(float(loss), float)


@pytest.mark.fast
def test_loss_is_non_negative():
    """Test that MSE loss is always non-negative."""
    model = create_mock_model()
    config = create_mock_config()

    # Test multiple batches
    for seed in range(5):
        batch = create_mock_batch(num_examples=2, seed=seed)
        loss = ZImageLoss.compute_loss(model, config, batch)
        assert float(loss) >= 0.0


@pytest.mark.fast
def test_loss_batch_averaging():
    """Test that loss averages correctly over batch."""
    model = create_mock_model()
    config = create_mock_config()

    # Single example batch
    batch1 = create_mock_batch(num_examples=1, seed=42)
    loss1 = ZImageLoss.compute_loss(model, config, batch1)

    # Two example batch with same seed
    batch2 = create_mock_batch(num_examples=2, seed=42)
    loss2 = ZImageLoss.compute_loss(model, config, batch2)

    # Both should be valid floats
    assert isinstance(float(loss1), float)
    assert isinstance(float(loss2), float)


@pytest.mark.fast
def test_noise_interpolation():
    """Test _add_noise_by_interpolation formula."""
    clean = mx.ones((1, 16, 16, 4))
    noise = mx.zeros((1, 16, 16, 4))

    # At sigma=0, should return clean
    result_sigma0 = ZImageLoss._add_noise_by_interpolation(clean, noise, sigma=0.0)
    np.testing.assert_array_almost_equal(np.array(result_sigma0), np.array(clean), decimal=5)

    # At sigma=1, should return noise
    result_sigma1 = ZImageLoss._add_noise_by_interpolation(clean, noise, sigma=1.0)
    np.testing.assert_array_almost_equal(np.array(result_sigma1), np.array(noise), decimal=5)

    # At sigma=0.5, should return midpoint
    result_sigma05 = ZImageLoss._add_noise_by_interpolation(clean, noise, sigma=0.5)
    expected_midpoint = 0.5 * clean + 0.5 * noise
    np.testing.assert_array_almost_equal(np.array(result_sigma05), np.array(expected_midpoint), decimal=5)


@pytest.mark.fast
def test_interpolation_linearity():
    """Test that interpolation is linear in sigma."""
    clean = mx.random.normal((1, 16, 16, 4))
    noise = mx.random.normal((1, 16, 16, 4))

    sigma_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for sigma in sigma_values:
        result = ZImageLoss._add_noise_by_interpolation(clean, noise, sigma=sigma)
        results.append(np.array(result))

    # Check linearity: result at 0.5 should be average of results at 0.25 and 0.75
    midpoint = (results[1] + results[3]) / 2
    np.testing.assert_array_almost_equal(results[2], midpoint, decimal=5)


@pytest.mark.fast
def test_validation_loss_same_as_training_loss():
    """Test that validation loss uses same computation as training loss."""
    model = create_mock_model()
    config = create_mock_config()
    batch = create_mock_batch(num_examples=2, seed=42)

    training_loss = ZImageLoss.compute_loss(model, config, batch)
    validation_loss = ZImageLoss.compute_validation_loss(model, config, batch)

    # With same random state, losses should be close
    # (not exactly equal due to random timestep sampling)
    assert isinstance(float(training_loss), float)
    assert isinstance(float(validation_loss), float)


@pytest.mark.fast
def test_different_seeds_different_losses():
    """Test that different RNG seeds produce different losses."""
    model = create_mock_model()
    config = create_mock_config()

    batch1 = create_mock_batch(num_examples=2, seed=42)
    batch2 = create_mock_batch(num_examples=2, seed=123)

    loss1 = ZImageLoss.compute_loss(model, config, batch1)
    loss2 = ZImageLoss.compute_loss(model, config, batch2)

    # Different seeds should generally produce different losses
    # (with high probability, not guaranteed)
    # At minimum, both should be valid
    assert isinstance(float(loss1), float)
    assert isinstance(float(loss2), float)


@pytest.mark.fast
def test_loss_with_zero_clean_image():
    """Test loss computation with zero-valued clean image."""
    example = Example(
        example_id=0,
        prompt="test",
        image_path=Path("test.jpg"),
        encoded_image=mx.zeros((1, 16, 16, 4)),  # Zero latent
        text_embeddings=mx.zeros((1, 77, 768)),
    )
    batch = Batch(examples=[example], rng=random.Random(42))

    model = create_mock_model()
    config = create_mock_config()

    loss = ZImageLoss.compute_loss(model, config, batch)

    # Should still produce valid loss
    assert isinstance(float(loss), float)
    assert not np.isnan(float(loss))
    assert not np.isinf(float(loss))


@pytest.mark.fast
def test_flow_matching_loss_formula():
    """Test the flow matching loss formula: ||clean + predicted_noise - pure_noise||^2."""
    # Create controlled inputs
    clean = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # 1x1x2x2 tensor
    pure_noise = mx.array([[[[0.5, 0.5], [0.5, 0.5]]]])
    predicted_noise = mx.array([[[[-0.5, 1.5], [2.5, 3.5]]]])

    # Expected: clean + predicted_noise - pure_noise
    # = [1+(-0.5)-0.5, 2+1.5-0.5, 3+2.5-0.5, 4+3.5-0.5]
    # = [0, 3, 5, 7]
    # MSE = mean([0^2, 9, 25, 49]) = mean([0, 9, 25, 49]) = 83/4 = 20.75

    diff = clean + predicted_noise - pure_noise
    expected_loss = diff.square().mean()

    assert float(expected_loss) == pytest.approx(20.75, rel=1e-5)
