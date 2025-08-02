import mlx.core as mx
import numpy as np

from mflux.schedulers.linear_scheduler import LinearScheduler


def test_linear_scheduler_initialization():
    """Test that the scheduler initializes with correct default values."""
    scheduler = LinearScheduler()

    assert scheduler.num_train_timesteps == 1000
    assert scheduler.shift == 1.0
    assert scheduler.use_dynamic_shifting is False
    assert scheduler.timesteps is None
    assert scheduler.sigmas is None


def test_linear_scheduler_set_timesteps():
    """Test setting timesteps creates correct sigma schedule."""
    scheduler = LinearScheduler()
    num_steps = 4

    scheduler.set_timesteps(num_steps)

    # Check timesteps
    assert scheduler.num_inference_steps == num_steps
    assert mx.array_equal(scheduler.timesteps, mx.array([0, 1, 2, 3]))

    # Check sigmas match original mflux implementation
    expected_sigmas = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])
    assert scheduler.sigmas.shape == (num_steps + 1,)
    assert mx.allclose(scheduler.sigmas, expected_sigmas, atol=1e-6)


def test_linear_scheduler_sigma_values():
    """Test that sigma values match the original mflux linear schedule."""
    scheduler = LinearScheduler()

    # Test with different step counts
    for num_steps in [2, 4, 10, 50]:
        scheduler.set_timesteps(num_steps)

        # First sigma should be 1.0
        assert mx.allclose(scheduler.sigmas[0], mx.array(1.0), atol=1e-6)

        # Last sigma should be 0.0
        assert mx.allclose(scheduler.sigmas[-1], mx.array(0.0), atol=1e-6)

        # Second to last should be 1/num_steps
        assert mx.allclose(scheduler.sigmas[-2], mx.array(1.0 / num_steps), atol=1e-6)

        # Check linearity
        diffs = scheduler.sigmas[1:-1] - scheduler.sigmas[:-2]
        # All differences should be equal (linear)
        assert mx.allclose(diffs, diffs[0], atol=1e-6)


def test_linear_scheduler_step():
    """Test the Euler integration step."""
    scheduler = LinearScheduler()
    scheduler.set_timesteps(4)

    # Create test data
    sample = mx.ones((2, 4, 64, 64))
    noise = mx.ones_like(sample) * 0.1

    # Test step 0
    result = scheduler.step(noise, 0, sample)
    dt = scheduler.sigmas[1] - scheduler.sigmas[0]  # Should be -0.25
    expected = sample + noise * dt
    assert mx.allclose(result, expected, atol=1e-6)

    # Test last step (should return sample unchanged)
    result_last = scheduler.step(noise, 4, sample)
    assert mx.array_equal(result_last, sample)


def test_linear_scheduler_dt_computation():
    """Test that dt values are computed correctly."""
    scheduler = LinearScheduler()
    scheduler.set_timesteps(4)

    # Compute all dt values
    dts = []
    for t in range(4):
        dt = scheduler.sigmas[t + 1] - scheduler.sigmas[t]
        dts.append(dt)

    # For linear schedule from 1.0 to 0.25 in 4 steps: [1.0, 0.75, 0.5, 0.25, 0.0]
    # dt values should be: [-0.25, -0.25, -0.25, -0.25]
    expected_dt = -0.25
    for dt in dts:
        assert mx.allclose(dt, mx.array(expected_dt), atol=1e-6)


def test_linear_scheduler_scale_noise():
    """Test that scale_noise returns sample unchanged (no scaling in linear scheduler)."""
    scheduler = LinearScheduler()

    sample = mx.random.normal((2, 4, 64, 64))
    timestep = 0

    result = scheduler.scale_noise(sample, timestep)
    assert mx.array_equal(result, sample)


def test_linear_scheduler_custom_sigmas():
    """Test that custom sigmas can be provided."""
    scheduler = LinearScheduler()

    custom_sigmas = mx.array([2.0, 1.5, 1.0, 0.5, 0.0])
    scheduler.set_timesteps(4, sigmas=custom_sigmas)

    assert mx.array_equal(scheduler.sigmas, custom_sigmas)


def test_linear_scheduler_edge_cases():
    """Test edge cases for the scheduler."""
    scheduler = LinearScheduler()

    # Test with 1 step
    scheduler.set_timesteps(1)
    assert scheduler.sigmas.shape == (2,)
    assert mx.allclose(scheduler.sigmas, mx.array([1.0, 0.0]), atol=1e-6)

    # Test with very large number of steps
    scheduler.set_timesteps(1000)
    assert scheduler.sigmas.shape == (1001,)
    assert mx.allclose(scheduler.sigmas[0], mx.array(1.0), atol=1e-6)
    assert mx.allclose(scheduler.sigmas[-1], mx.array(0.0), atol=1e-6)
    assert mx.allclose(scheduler.sigmas[-2], mx.array(0.001), atol=1e-6)


def test_matches_original_mflux_implementation():
    """Test that our scheduler exactly matches the original mflux sigma generation."""
    scheduler = LinearScheduler()

    # Test cases from original mflux
    num_inference_steps = 4
    scheduler.set_timesteps(num_inference_steps)

    # Original mflux: np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    # Then concatenate with zeros(1)
    original_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    original_sigmas = np.concatenate([original_sigmas, np.zeros(1)])
    original_sigmas = mx.array(original_sigmas).astype(mx.float32)

    assert mx.allclose(scheduler.sigmas, original_sigmas, atol=1e-6)
