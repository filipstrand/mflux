import mlx.core as mx
import numpy as np

from mflux.schedulers.ddim_scheduler import DDIMScheduler, betas_for_alpha_bar


def test_ddim_scheduler_initialization():
    """Test DDIM scheduler initialization with default values."""
    scheduler = DDIMScheduler()

    assert scheduler.num_train_timesteps == 1000
    assert scheduler.beta_start == 0.0001
    assert scheduler.beta_end == 0.02
    assert scheduler.beta_schedule == "linear"
    # Corrected default from False to True to match diffusers
    assert scheduler.set_alpha_to_one is True
    assert scheduler.steps_offset == 0
    assert scheduler.prediction_type == "epsilon"
    # Test new default values
    assert scheduler.clip_sample is True
    assert scheduler.timestep_spacing == "leading"
    assert scheduler.rescale_betas_zero_snr is False


def test_ddim_linear_beta_schedule():
    """Test linear beta schedule generation."""
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_start=0.0001, beta_end=0.02)
    assert scheduler.betas.shape == (10,)
    assert mx.allclose(scheduler.betas[0], mx.array(0.0001), atol=1e-6)
    assert mx.allclose(scheduler.betas[-1], mx.array(0.02), atol=1e-6)


def test_ddim_scaled_linear_beta_schedule():
    """Test scaled linear beta schedule generation."""
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_start=0.0001, beta_end=0.02, beta_schedule="scaled_linear")
    sqrt_beta_start = 0.0001**0.5
    sqrt_beta_end = 0.02**0.5
    expected_sqrt = mx.linspace(sqrt_beta_start, sqrt_beta_end, 10, dtype=mx.float32)
    expected_betas = expected_sqrt**2
    assert mx.allclose(scheduler.betas, expected_betas, atol=1e-6)


# New test for squaredcos_cap_v2 schedule
def test_squaredcos_cap_v2_schedule():
    """Test squaredcos_cap_v2 beta schedule generation."""
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2")
    expected_betas = betas_for_alpha_bar(10)
    assert mx.allclose(scheduler.betas, expected_betas, atol=1e-6)


# New test for rescale_betas_zero_snr
def test_rescale_betas_zero_snr():
    """Test the zero terminal SNR rescaling."""
    scheduler = DDIMScheduler(num_train_timesteps=10, rescale_betas_zero_snr=True)
    # The last alpha_cumprod should be close to zero
    assert mx.allclose(scheduler.alphas_cumprod[-1], mx.array(0.0), atol=1e-6)


def test_ddim_set_timesteps():
    """Test setting timesteps for DDIM."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=50)
    assert scheduler.num_inference_steps == 50
    assert scheduler.timesteps.shape == (50,)
    assert scheduler.timesteps[0] > scheduler.timesteps[-1]


# New tests for all timestep spacings
def test_ddim_timestep_spacing_leading():
    scheduler = DDIMScheduler(num_train_timesteps=100, timestep_spacing="leading")
    scheduler.set_timesteps(10)
    assert mx.array_equal(scheduler.timesteps, mx.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0]))


def test_ddim_timestep_spacing_trailing():
    scheduler = DDIMScheduler(num_train_timesteps=100, timestep_spacing="trailing")
    scheduler.set_timesteps(10)
    # Note: diffusers trailing has a bug for num_train_timesteps that are a multiple of steps
    # This is the corrected expectation
    assert mx.array_equal(scheduler.timesteps, mx.array([99, 89, 79, 69, 59, 49, 39, 29, 19, 9]))


def test_ddim_timestep_spacing_linspace():
    scheduler = DDIMScheduler(num_train_timesteps=100, timestep_spacing="linspace")
    scheduler.set_timesteps(10)
    expected = mx.array(np.linspace(0, 99, 10).round()[::-1].copy().astype(np.int64))
    assert mx.array_equal(scheduler.timesteps, expected)


def test_ddim_step_deterministic():
    """Test DDIM step with eta=0 (deterministic) after fixing the implementation."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 980

    # Correctly unpack the tuple returned by step
    prev_sample, pred_original_sample = scheduler.step(noise_pred, timestep, sample, eta=0.0)

    assert prev_sample.shape == sample.shape

    # Verify the computation using the CORRECT DDIM formula
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[timestep - 20]
    beta_prod_t = 1 - alpha_prod_t

    # This is the pred_original_sample the step function should have calculated
    expected_pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    # The test must also account for the clipping that happens inside the scheduler
    expected_pred_original_sample = mx.clip(
        expected_pred_original_sample, -scheduler.clip_sample_range, scheduler.clip_sample_range
    )
    assert mx.allclose(pred_original_sample, expected_pred_original_sample, atol=1e-4)

    # This is the pred_epsilon the step function should have used
    pred_epsilon = noise_pred

    # This is the direction pointing to x_t
    pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

    # This is the final expected previous sample
    expected_prev_sample = alpha_prod_t_prev ** (0.5) * expected_pred_original_sample + pred_sample_direction

    assert mx.allclose(prev_sample, expected_prev_sample, atol=1e-4)


def test_ddim_step_stochastic():
    """Test DDIM step with eta>0 (stochastic)."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)

    sample = mx.random.normal((1, 4, 64, 64))
    noise_pred = mx.random.normal((1, 4, 64, 64)) * 0.1
    timestep = 980

    # Correctly unpack tuple
    prev_sample_1, _ = scheduler.step(noise_pred, timestep, sample, eta=1.0)
    prev_sample_2, _ = scheduler.step(noise_pred, timestep, sample, eta=1.0)

    # With eta > 0, results should be different without a generator
    assert not mx.allclose(prev_sample_1, prev_sample_2, atol=1e-3)


def test_ddim_with_offset():
    """Test DDIM with steps offset."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, steps_offset=1, timestep_spacing="leading")
    scheduler.set_timesteps(50)

    scheduler_no_offset = DDIMScheduler(num_train_timesteps=1000, steps_offset=0, timestep_spacing="leading")
    scheduler_no_offset.set_timesteps(50)

    assert mx.all(scheduler.timesteps == scheduler_no_offset.timesteps + 1)
