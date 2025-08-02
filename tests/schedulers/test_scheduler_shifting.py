import mlx.core as mx
import numpy as np

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig


def test_sigma_shifting_for_dev_model():
    """Test that sigma shifting is applied correctly for dev models."""
    # Create config for dev model (which requires shifting)
    config = Config(num_inference_steps=4, width=1024, height=1024)
    model_config = ModelConfig.from_name("dev", base_model=None)

    runtime_config = RuntimeConfig(config, model_config)

    # Dev model should have shifted sigmas
    assert model_config.requires_sigma_shift is True

    # Get the sigmas
    sigmas = runtime_config.sigmas

    # Verify shifting was applied
    # For dev model, sigmas are transformed using sigmoid-like function
    linear_sigmas = np.linspace(1.0, 1.0 / 4, 4)
    linear_sigmas = np.concatenate([linear_sigmas, np.zeros(1)])
    linear_sigmas = mx.array(linear_sigmas).astype(mx.float32)

    # Sigmas should be different from linear due to shifting
    assert not mx.allclose(sigmas, linear_sigmas, atol=1e-3)


def test_no_sigma_shifting_for_schnell_model():
    """Test that sigma shifting is NOT applied for schnell models."""
    # Create config for schnell model (which doesn't require shifting)
    config = Config(num_inference_steps=4, width=1024, height=1024)
    model_config = ModelConfig.from_name("schnell", base_model=None)

    runtime_config = RuntimeConfig(config, model_config)

    # Schnell model should not have shifted sigmas
    assert model_config.requires_sigma_shift is False

    # Get the sigmas
    sigmas = runtime_config.sigmas

    # Verify no shifting was applied - should be linear
    expected_sigmas = mx.linspace(1.0, 1.0 / 4, 4)
    expected_sigmas = mx.concatenate([expected_sigmas, mx.zeros(1)])

    assert mx.allclose(sigmas, expected_sigmas, atol=1e-6)


def test_shift_sigmas_computation():
    """Test the actual sigma shifting computation."""
    # Test the shifting formula directly
    sigmas = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])
    width = 1024
    height = 1024

    # Compute mu based on resolution (from RuntimeConfig._shift_sigmas)
    y1 = 0.5
    x1 = 256
    m = (1.15 - y1) / (4096 - x1)
    b = y1 - m * x1
    mu = m * width * height / 256 + b
    mu = mx.array(mu)

    # Apply shifting formula
    shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas[:-1] - 1))
    shifted_sigmas = mx.concatenate([shifted_sigmas, mx.zeros(1)])

    # Verify properties of shifted sigmas
    # Should still start close to 1 and end at 0
    assert shifted_sigmas[-1] == 0
    # When sigma=1.0, the formula gives exactly 1.0 (no shift at the extremes)
    assert mx.allclose(shifted_sigmas[0], mx.array(1.0), atol=1e-6)
    # Middle values should be shifted
    assert shifted_sigmas[1] > sigmas[1]  # Should be shifted up from 0.75
    assert shifted_sigmas[2] > sigmas[2]  # Should be shifted up from 0.5

    # Should still be monotonically decreasing
    diffs = shifted_sigmas[1:] - shifted_sigmas[:-1]
    assert mx.all(diffs <= 0)


def test_shift_sigmas_different_resolutions():
    """Test that shifting varies with image resolution."""
    base_sigmas = mx.array([1.0, 0.75, 0.5, 0.25])

    # Helper function to compute mu
    def compute_mu(width, height):
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        return m * width * height / 256 + b

    # Test different resolutions
    resolutions = [(512, 512), (1024, 1024), (2048, 2048)]
    mus = [compute_mu(w, h) for w, h in resolutions]

    # Higher resolution should give higher mu
    assert mus[0] < mus[1] < mus[2]

    # This affects the shifting - higher mu means less aggressive shifting
    shifted_results = []
    for mu in mus:
        mu_array = mx.array(mu)
        shifted = mx.exp(mu_array) / (mx.exp(mu_array) + (1 / base_sigmas - 1))
        shifted_results.append(shifted)

    # First sigma (1.0) always stays 1.0 after shifting
    # Check the second sigma instead - it should show the resolution effect
    assert shifted_results[0][1] < shifted_results[1][1] < shifted_results[2][1]

    # Lower resolution (smaller mu) produces more aggressive shifting (values closer to original)
    # Higher resolution (larger mu) produces less aggressive shifting (values closer to 1)


def test_runtime_config_sigma_creation():
    """Test the complete sigma creation pipeline in RuntimeConfig."""
    # Test both model types
    for model_name in ["schnell", "dev"]:
        config = Config(num_inference_steps=10, width=1024, height=1024)
        model_config = ModelConfig.from_name(model_name, base_model=None)
        runtime_config = RuntimeConfig(config, model_config)

        # Should have correct number of sigmas
        assert runtime_config.sigmas.shape == (11,)  # num_steps + 1

        # Should start high and end at 0
        assert runtime_config.sigmas[-1] == 0
        assert runtime_config.sigmas[0] > 0

        # Should be monotonically decreasing
        diffs = runtime_config.sigmas[1:] - runtime_config.sigmas[:-1]
        assert mx.all(diffs <= 0)
