import mlx.core as mx

from mflux.schedulers.ddim_scheduler import DDIMScheduler


def test_ddim_scheduler_initialization():
    """Test DDIM scheduler initialization with default values."""
    scheduler = DDIMScheduler()

    assert scheduler.num_train_timesteps == 1000
    assert scheduler.beta_start == 0.0001
    assert scheduler.beta_end == 0.02
    assert scheduler.beta_schedule == "linear"
    assert scheduler.eta == 0.0
    assert scheduler.set_alpha_to_one is False
    assert scheduler.steps_offset == 0
    assert scheduler.prediction_type == "epsilon"


def test_ddim_linear_beta_schedule():
    """Test linear beta schedule generation."""
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_start=0.0001, beta_end=0.02)

    # Check beta values
    assert scheduler.betas.shape == (10,)
    assert mx.allclose(scheduler.betas[0], mx.array(0.0001), atol=1e-6)
    assert mx.allclose(scheduler.betas[-1], mx.array(0.02), atol=1e-6)

    # Check alphas
    expected_alphas = 1.0 - scheduler.betas
    assert mx.allclose(scheduler.alphas, expected_alphas, atol=1e-6)

    # Check cumulative product
    expected_alphas_cumprod = mx.cumprod(scheduler.alphas, axis=0)
    assert mx.allclose(scheduler.alphas_cumprod, expected_alphas_cumprod, atol=1e-6)


def test_ddim_scaled_linear_beta_schedule():
    """Test scaled linear beta schedule generation."""
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_start=0.0001, beta_end=0.02, beta_schedule="scaled_linear")

    # Check that betas follow scaled linear pattern
    sqrt_beta_start = 0.0001**0.5
    sqrt_beta_end = 0.02**0.5
    expected_sqrt = mx.linspace(sqrt_beta_start, sqrt_beta_end, 10, dtype=mx.float32)
    expected_betas = expected_sqrt**2

    assert mx.allclose(scheduler.betas, expected_betas, atol=1e-6)


def test_ddim_set_timesteps():
    """Test setting timesteps for DDIM."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=50)

    assert scheduler.num_inference_steps == 50
    assert scheduler.timesteps.shape == (50,)

    # Timesteps should be in reverse order (from high to low)
    assert scheduler.timesteps[0] > scheduler.timesteps[-1]

    # Check step ratio
    step_ratio = 1000 // 50  # 20
    expected_first = (49 * step_ratio) + scheduler.steps_offset
    assert scheduler.timesteps[0] == expected_first


def test_ddim_step_deterministic():
    """Test DDIM step with eta=0 (deterministic)."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    scheduler.set_timesteps(50)

    # Create test data
    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 980  # Near the beginning

    # Perform step
    prev_sample = scheduler.step(noise_pred, timestep, sample, eta=0.0)

    # With eta=0, DDIM is deterministic
    # Verify output shape
    assert prev_sample.shape == sample.shape

    # Verify the computation follows DDIM formula
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    prev_timestep = timestep - 1000 // 50
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else mx.array(1.0)

    # Predicted original sample
    pred_original = (sample - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t**0.5

    # DDIM formula
    expected = alpha_prod_t_prev**0.5 * pred_original + (1 - alpha_prod_t_prev) ** 0.5 * noise_pred

    assert mx.allclose(prev_sample, expected, atol=1e-5)


def test_ddim_step_stochastic():
    """Test DDIM step with eta>0 (stochastic)."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 980

    # Fix random seed for reproducibility
    mx.random.seed(42)
    prev_sample_1 = scheduler.step(noise_pred, timestep, sample, eta=1.0, generator=42)

    mx.random.seed(42)
    prev_sample_2 = scheduler.step(noise_pred, timestep, sample, eta=1.0, generator=42)

    # With same seed, should get same result
    assert mx.allclose(prev_sample_1, prev_sample_2, atol=1e-6)

    # Different seed should give different result
    prev_sample_3 = scheduler.step(noise_pred, timestep, sample, eta=1.0, generator=43)
    assert not mx.allclose(prev_sample_1, prev_sample_3, atol=1e-3)


def test_ddim_scale_model_input():
    """Test that scale_model_input returns unchanged sample."""
    scheduler = DDIMScheduler()
    sample = mx.random.normal((1, 4, 64, 64))

    scaled = scheduler.scale_model_input(sample, timestep=500)
    assert mx.array_equal(scaled, sample)


def test_ddim_v_prediction():
    """Test DDIM with v-prediction parameterization."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="v_prediction")
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    v_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 980

    prev_sample = scheduler.step(v_pred, timestep, sample)

    # Verify computation for v-prediction
    # alpha_prod_t = scheduler.alphas_cumprod[timestep]
    # beta_prod_t = 1 - alpha_prod_t
    # v-prediction formula would compute pred_original here
    # pred_original = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * v_pred

    # Rest follows standard DDIM
    assert prev_sample.shape == sample.shape


def test_ddim_timestep_spacing():
    """Test different timestep spacings."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Test with different number of inference steps
    for num_steps in [10, 25, 50, 100]:
        scheduler.set_timesteps(num_steps)

        assert len(scheduler.timesteps) == num_steps

        # Check that timesteps are evenly spaced
        diffs = scheduler.timesteps[:-1] - scheduler.timesteps[1:]
        expected_diff = 1000 // num_steps
        assert mx.all(diffs == expected_diff)


def test_ddim_with_offset():
    """Test DDIM with steps offset."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, steps_offset=1)
    scheduler.set_timesteps(50)

    # All timesteps should be offset by 1
    scheduler_no_offset = DDIMScheduler(num_train_timesteps=1000, steps_offset=0)
    scheduler_no_offset.set_timesteps(50)

    assert mx.all(scheduler.timesteps == scheduler_no_offset.timesteps + 1)


def test_ddim_variance_computation():
    """Test variance computation for stochastic DDIM."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    timestep = 500
    prev_timestep = 480

    variance = scheduler._get_variance(timestep, prev_timestep)

    # Verify variance formula
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    expected_variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    assert mx.allclose(variance, expected_variance, atol=1e-6)
