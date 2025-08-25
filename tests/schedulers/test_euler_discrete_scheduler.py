import mlx.core as mx

from mflux.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler


def test_euler_discrete_initialization():
    """Test Euler Discrete scheduler initialization."""
    scheduler = EulerDiscreteScheduler()
    assert scheduler.num_train_timesteps == 1000
    assert scheduler.prediction_type == "epsilon"
    assert scheduler.use_karras_sigmas is False


def test_euler_discrete_sigma_conversion():
    """Test conversion from beta schedule to sigma schedule for correctness."""
    scheduler = EulerDiscreteScheduler(num_train_timesteps=10)

    # The initial sigmas should be in ascending order.
    expected_sigmas = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5

    assert mx.allclose(scheduler.sigmas, expected_sigmas, atol=1e-6)


def test_euler_discrete_set_timesteps():
    """Test setting timesteps with linear spacing."""
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    assert scheduler.num_inference_steps == 50
    assert len(scheduler.timesteps) == 50
    # Sigmas should have N+1 values
    assert len(scheduler.sigmas) == 51


def test_euler_discrete_karras_sigmas():
    """Test Karras sigma schedule."""
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000, use_karras_sigmas=True, sigma_min=0.002, sigma_max=80.0
    )
    scheduler.set_timesteps(10)

    rho = 7.0
    u = mx.linspace(0, 1, 10)
    expected = (80.0 ** (1 / rho) + u * (0.002 ** (1 / rho) - 80.0 ** (1 / rho))) ** rho
    expected = mx.concatenate([expected, mx.array([0.0])])

    assert mx.allclose(scheduler.sigmas, expected, atol=1e-5)


def test_euler_discrete_scale_model_input():
    """Test model input scaling."""
    scheduler = EulerDiscreteScheduler(sigma_data=1.5)
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    timestep = 10

    scaled = scheduler.scale_model_input(sample, timestep)

    sigma = scheduler.sigmas[timestep]
    expected_scale = 1.0 / ((sigma**2 + scheduler.sigma_data**2) ** 0.5)
    expected = sample * expected_scale

    assert mx.allclose(scaled, expected, atol=1e-6)


def test_euler_discrete_step_epsilon():
    """Test Euler step with epsilon prediction based on corrected logic."""
    scheduler = EulerDiscreteScheduler(prediction_type="epsilon")
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 10

    prev_sample = scheduler.step(noise_pred, timestep, sample)

    # Verify Euler method computation with correct sigma values
    sigma = scheduler.sigmas[timestep]
    sigma_next = scheduler.sigmas[timestep + 1]

    denoised = sample - sigma * noise_pred
    derivative = (sample - denoised) / sigma
    dt = sigma_next - sigma
    expected = sample + derivative * dt

    assert mx.allclose(prev_sample, expected, atol=1e-5)


def test_euler_discrete_sigma_ordering():
    """Test sigma ordering for Euler discrete scheduler."""
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)

    # Test without Karras
    scheduler.set_timesteps(50)
    # Sigmas should be in descending order
    diffs = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    # All differences should be <= 0
    assert mx.all(diffs <= 0)
    # The last sigma should be 0
    assert scheduler.sigmas[-1] == 0.0

    # Test with Karras sigmas
    scheduler_karras = EulerDiscreteScheduler(use_karras_sigmas=True)
    scheduler_karras.set_timesteps(50)
    # Karras sigmas should also be in descending order
    diffs_karras = scheduler_karras.sigmas[1:] - scheduler_karras.sigmas[:-1]
    assert mx.all(diffs_karras < 0)
    assert scheduler_karras.sigmas[-1] == 0.0


def test_euler_discrete_deterministic():
    """Test that Euler discrete is deterministic."""
    scheduler = EulerDiscreteScheduler()
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 10

    result1 = scheduler.step(noise_pred, timestep, sample)
    result2 = scheduler.step(noise_pred, timestep, sample)

    assert mx.array_equal(result1, result2)
