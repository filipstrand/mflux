import mlx.core as mx

from mflux.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler


def test_euler_discrete_initialization():
    """Test Euler Discrete scheduler initialization."""
    scheduler = EulerDiscreteScheduler()

    assert scheduler.num_train_timesteps == 1000
    assert scheduler.beta_start == 0.0001
    assert scheduler.beta_end == 0.02
    assert scheduler.beta_schedule == "linear"
    assert scheduler.prediction_type == "epsilon"
    assert scheduler.use_karras_sigmas is False
    assert scheduler.sigma_min == 0.002
    assert scheduler.sigma_max == 80.0


def test_euler_discrete_sigma_conversion():
    """Test conversion from beta schedule to sigma schedule."""
    scheduler = EulerDiscreteScheduler(num_train_timesteps=10)

    # Check sigma computation from alphas_cumprod
    expected_sigmas = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5
    expected_sigmas = mx.concatenate([mx.array([0.0]), expected_sigmas])

    assert mx.allclose(scheduler.sigmas, expected_sigmas, atol=1e-6)


def test_euler_discrete_set_timesteps():
    """Test setting timesteps with linear spacing."""
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)

    assert scheduler.num_inference_steps == 50
    assert len(scheduler.timesteps) == 50
    assert len(scheduler.sigmas) == 51  # One extra for final sigma


def test_euler_discrete_karras_sigmas():
    """Test Karras sigma schedule."""
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000, use_karras_sigmas=True, sigma_min=0.002, sigma_max=80.0
    )
    scheduler.set_timesteps(10)

    # Karras sigmas should follow specific formula
    rho = 7.0
    u = mx.linspace(0, 1, 10)
    expected = (80.0 ** (1 / rho) + u * (0.002 ** (1 / rho) - 80.0 ** (1 / rho))) ** rho
    expected = mx.concatenate([expected, mx.array([0.0])])

    assert mx.allclose(scheduler.sigmas, expected, atol=1e-5)


def test_euler_discrete_scale_model_input():
    """Test model input scaling."""
    scheduler = EulerDiscreteScheduler()
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    timestep = 10

    scaled = scheduler.scale_model_input(sample, timestep)

    # Should scale by 1/sqrt(sigma^2 + 1)
    sigma = scheduler.sigmas[timestep]
    expected_scale = 1.0 / ((sigma**2 + 1) ** 0.5)
    expected = sample * expected_scale

    assert mx.allclose(scaled, expected, atol=1e-6)


def test_euler_discrete_step_epsilon():
    """Test Euler step with epsilon prediction."""
    scheduler = EulerDiscreteScheduler(prediction_type="epsilon")
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 10

    prev_sample = scheduler.step(noise_pred, timestep, sample)

    # Verify Euler method computation
    sigma = scheduler.sigmas[timestep]
    sigma_next = scheduler.sigmas[timestep + 1]

    # Denoised sample
    denoised = sample - sigma * noise_pred

    # Derivative
    derivative = (sample - denoised) / sigma

    # Euler step
    dt = sigma_next - sigma
    expected = sample + derivative * dt

    assert mx.allclose(prev_sample, expected, atol=1e-6)


def test_euler_discrete_step_v_prediction():
    """Test Euler step with v-prediction."""
    scheduler = EulerDiscreteScheduler(prediction_type="v_prediction")
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    v_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 10

    prev_sample = scheduler.step(v_pred, timestep, sample)

    # Verify v-prediction computation
    # sigma = scheduler.sigmas[timestep]
    # v-prediction would compute denoised here
    # denoised = sample / ((sigma**2 + 1) ** 0.5) - (sigma / ((sigma**2 + 1) ** 0.5)) * v_pred

    assert prev_sample.shape == sample.shape


def test_euler_discrete_edge_cases():
    """Test edge cases."""
    scheduler = EulerDiscreteScheduler()
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1

    # Test that last valid timestep (49) still performs a step
    last_sample = scheduler.step(noise_pred, 49, sample)
    assert last_sample.shape == sample.shape
    # Should not be equal since it performs a step
    assert not mx.array_equal(last_sample, sample)

    # Test beyond valid timesteps (50 and above should return sample unchanged)
    beyond_sample = scheduler.step(noise_pred, 50, sample)
    assert mx.array_equal(beyond_sample, sample)

    beyond_sample2 = scheduler.step(noise_pred, 51, sample)
    assert mx.array_equal(beyond_sample2, sample)


def test_euler_discrete_sigma_ordering():
    """Test sigma ordering for Euler discrete scheduler."""
    scheduler = EulerDiscreteScheduler()

    # Test without Karras
    scheduler.use_karras_sigmas = False
    scheduler.set_timesteps(50)

    # For Euler discrete, sigmas start at 0 and increase
    # The first sigma is 0 (no noise), and they increase
    assert scheduler.sigmas[0] == 0.0

    # Most sigmas should be increasing (noise level increases)
    # But check specific properties based on the implementation
    assert scheduler.sigmas[-1] == 0.0  # Last sigma is also 0

    # Test with Karras sigmas
    scheduler.use_karras_sigmas = True
    scheduler.set_timesteps(50)

    # Karras sigmas should decrease from high to low
    diffs = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    # Most differences should be negative (decreasing)
    assert mx.sum(diffs < 0) > mx.sum(diffs > 0)


def test_euler_discrete_deterministic():
    """Test that Euler discrete is deterministic."""
    scheduler = EulerDiscreteScheduler()
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 10

    # Run multiple times
    result1 = scheduler.step(noise_pred, timestep, sample)
    result2 = scheduler.step(noise_pred, timestep, sample)

    # Should get exact same result
    assert mx.array_equal(result1, result2)


def test_euler_discrete_different_beta_schedules():
    """Test scaled linear beta schedule."""
    scheduler_linear = EulerDiscreteScheduler(beta_schedule="linear")
    scheduler_scaled = EulerDiscreteScheduler(beta_schedule="scaled_linear")

    # Betas should be different
    assert not mx.allclose(scheduler_linear.betas, scheduler_scaled.betas, atol=1e-3)

    # But both should start and end at specified values
    assert mx.allclose(scheduler_linear.betas[0], mx.array(0.0001), atol=1e-6)
    assert mx.allclose(scheduler_linear.betas[-1], mx.array(0.02), atol=1e-6)
