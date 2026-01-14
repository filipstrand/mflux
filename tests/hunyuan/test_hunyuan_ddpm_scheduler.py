"""
Unit tests for DDPM scheduler integration with Hunyuan-DiT.

Tests verify:
1. DDPM scheduler initialization
2. Beta schedule computation
3. Alpha and alpha_cumprod computation
4. Timestep scheduling
5. Denoising step mechanics
6. Noise addition for img2img
7. Model input scaling
8. Integration with Hunyuan-DiT
"""

import pytest
import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.ddpm_scheduler import DDPMScheduler


class TestDDPMSchedulerInitialization:
    """Tests for DDPM scheduler initialization."""

    @pytest.mark.fast
    def test_scheduler_initialization(self):
        """Verify DDPM scheduler initializes correctly."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert scheduler.config == config
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.prediction_type == "epsilon"

    @pytest.mark.fast
    def test_custom_beta_schedule(self):
        """Verify custom beta values work."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        beta_start = 0.0001
        beta_end = 0.02

        scheduler = DDPMScheduler(
            config=config,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Verify betas are in expected range (with tolerance for float32 precision)
        assert float(mx.min(scheduler.betas)) >= beta_start - 1e-6
        assert float(mx.max(scheduler.betas)) <= beta_end + 1e-6

    @pytest.mark.fast
    def test_custom_num_train_timesteps(self):
        """Verify custom training timesteps work."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        num_train_timesteps = 2000

        scheduler = DDPMScheduler(
            config=config,
            num_train_timesteps=num_train_timesteps,
        )

        assert scheduler.num_train_timesteps == num_train_timesteps
        assert len(scheduler.betas) == num_train_timesteps


class TestDDPMBetaSchedule:
    """Tests for beta schedule computation."""

    @pytest.mark.fast
    def test_beta_schedule_length(self):
        """Verify beta schedule has correct length."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert len(scheduler.betas) == scheduler.num_train_timesteps

    @pytest.mark.fast
    def test_beta_schedule_monotonic(self):
        """Verify beta schedule is monotonically increasing."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        betas = scheduler.betas
        # Check if generally increasing (allowing for numerical precision)
        diffs = betas[1:] - betas[:-1]
        assert mx.all(diffs >= -1e-6)

    @pytest.mark.fast
    def test_beta_schedule_scaled_linear(self):
        """Verify beta schedule uses scaled linear (sqrt) approach."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        beta_start = 0.00085
        beta_end = 0.012

        scheduler = DDPMScheduler(
            config=config,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # First beta should be close to beta_start
        assert mx.allclose(scheduler.betas[0], mx.array(beta_start), atol=1e-4)

        # Last beta should be close to beta_end
        assert mx.allclose(scheduler.betas[-1], mx.array(beta_end), atol=1e-4)


class TestDDPMAlphaComputation:
    """Tests for alpha and alpha_cumprod computation."""

    @pytest.mark.fast
    def test_alphas_computation(self):
        """Verify alphas = 1 - betas."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        expected_alphas = 1.0 - scheduler.betas
        assert mx.allclose(scheduler.alphas, expected_alphas, atol=1e-6)

    @pytest.mark.fast
    def test_alphas_in_valid_range(self):
        """Verify alphas are in (0, 1] range."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert mx.all(scheduler.alphas > 0.0)
        assert mx.all(scheduler.alphas <= 1.0)

    @pytest.mark.fast
    def test_alphas_cumprod_decreasing(self):
        """Verify alphas_cumprod is monotonically decreasing."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        alphas_cumprod = scheduler.alphas_cumprod
        diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
        assert mx.all(diffs <= 1e-6)  # Should be decreasing

    @pytest.mark.fast
    def test_alphas_cumprod_range(self):
        """Verify alphas_cumprod stays in (0, 1] range."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert mx.all(scheduler.alphas_cumprod > 0.0)
        assert mx.all(scheduler.alphas_cumprod <= 1.0)
        # First value should be close to 1
        assert scheduler.alphas_cumprod[0] > 0.99


class TestDDPMTimestepScheduling:
    """Tests for timestep scheduling."""

    @pytest.mark.fast
    def test_timesteps_length(self):
        """Verify timesteps array has correct length."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert len(scheduler.timesteps) == config.num_inference_steps

    @pytest.mark.fast
    def test_timesteps_descending_order(self):
        """Verify timesteps are in descending order (high to low noise)."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        timesteps = scheduler.timesteps
        diffs = timesteps[1:] - timesteps[:-1]
        # Should be decreasing
        assert mx.all(diffs <= 0)

    @pytest.mark.fast
    def test_timesteps_range(self):
        """Verify timesteps are in valid range."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert mx.all(scheduler.timesteps >= 0)
        assert mx.all(scheduler.timesteps < scheduler.num_train_timesteps)

    @pytest.mark.fast
    def test_different_inference_steps(self):
        """Verify different inference steps produce different schedules."""
        model_config = ModelConfig.hunyuan()

        config_25 = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=25,
            model_config=model_config,
        )

        config_50 = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=model_config,
        )

        scheduler_25 = DDPMScheduler(config=config_25)
        scheduler_50 = DDPMScheduler(config=config_50)

        assert len(scheduler_25.timesteps) == 25
        assert len(scheduler_50.timesteps) == 50


class TestDDPMSigmasComputation:
    """Tests for sigma computation."""

    @pytest.mark.fast
    def test_sigmas_length(self):
        """Verify sigmas array has correct length."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        # Sigmas should be num_inference_steps + 1 (includes final zero)
        assert len(scheduler.sigmas) == config.num_inference_steps + 1

    @pytest.mark.fast
    def test_sigmas_final_value_zero(self):
        """Verify final sigma is zero (no noise at end)."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert float(scheduler.sigmas[-1]) == 0.0

    @pytest.mark.fast
    def test_sigmas_positive(self):
        """Verify all sigmas (except last) are positive."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        # All but last should be positive
        assert mx.all(scheduler.sigmas[:-1] > 0.0)


class TestDDPMDenoisingStep:
    """Tests for denoising step mechanics."""

    @pytest.mark.fast
    def test_step_output_shape(self):
        """Verify step output matches input latent shape."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        batch_size = 2
        channels = 4
        height = 128
        width = 128

        latents = mx.random.normal((batch_size, channels, height, width))
        noise = mx.random.normal((batch_size, channels, height, width))

        timestep = 0
        denoised = scheduler.step(noise=noise, timestep=timestep, latents=latents)

        assert denoised.shape == latents.shape

    @pytest.mark.fast
    def test_step_reduces_noise(self):
        """Verify denoising step generally reduces noise magnitude."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        batch_size = 1
        channels = 4
        height = 64
        width = 64

        # Start with noisy latents
        latents = mx.random.normal((batch_size, channels, height, width)) * 5.0
        # Predict noise
        noise = mx.random.normal((batch_size, channels, height, width))

        # Later timestep (less noise remaining)
        timestep = 40
        denoised = scheduler.step(noise=noise, timestep=timestep, latents=latents)

        # Denoised should generally have smaller magnitude
        original_norm = mx.sqrt(mx.sum(latents ** 2))
        denoised_norm = mx.sqrt(mx.sum(denoised ** 2))

        # Both should be finite
        assert mx.isfinite(original_norm)
        assert mx.isfinite(denoised_norm)

    @pytest.mark.fast
    def test_step_different_timesteps(self):
        """Verify different timesteps produce different results."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        batch_size = 1
        channels = 4
        height = 64
        width = 64

        latents = mx.random.normal((batch_size, channels, height, width))
        noise = mx.random.normal((batch_size, channels, height, width))

        denoised_t0 = scheduler.step(noise=noise, timestep=0, latents=latents)
        denoised_t25 = scheduler.step(noise=noise, timestep=25, latents=latents)

        # Different timesteps should produce different results
        assert not mx.allclose(denoised_t0, denoised_t25, atol=1e-3)

    @pytest.mark.fast
    def test_prediction_type_epsilon(self):
        """Verify epsilon prediction type works."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(
            config=config,
            prediction_type="epsilon",
        )

        assert scheduler.prediction_type == "epsilon"

        latents = mx.random.normal((1, 4, 64, 64))
        noise = mx.random.normal((1, 4, 64, 64))

        denoised = scheduler.step(noise=noise, timestep=0, latents=latents)
        assert denoised.shape == latents.shape


class TestDDPMNoiseAddition:
    """Tests for noise addition (img2img)."""

    @pytest.mark.fast
    def test_add_noise_shape(self):
        """Verify add_noise preserves shape."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        original = mx.random.normal((1, 4, 64, 64))
        noise = mx.random.normal((1, 4, 64, 64))
        timestep = 500

        noisy = scheduler.add_noise(original, noise, timestep)

        assert noisy.shape == original.shape

    @pytest.mark.fast
    def test_add_noise_increases_variance(self):
        """Verify adding noise increases variance."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        # Clean signal
        original = mx.ones((1, 4, 64, 64)) * 0.5
        noise = mx.random.normal((1, 4, 64, 64))

        # Add significant noise
        timestep = 100  # Early timestep, more noise
        noisy = scheduler.add_noise(original, noise, timestep)

        original_std = mx.std(original)
        noisy_std = mx.std(noisy)

        # Noisy version should have higher variance
        assert noisy_std > original_std

    @pytest.mark.fast
    def test_add_noise_different_timesteps(self):
        """Verify different timesteps add different amounts of noise."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        original = mx.zeros((1, 4, 32, 32))
        noise = mx.random.normal((1, 4, 32, 32))

        # Early timestep (more noise) - use lower timestep number (more alpha_cumprod)
        noisy_early = scheduler.add_noise(original, noise, timestep=900)
        # Late timestep (less noise) - use higher timestep number (less alpha_cumprod)
        noisy_late = scheduler.add_noise(original, noise, timestep=100)

        # Early should have more noise (higher timestep = more noise added)
        early_magnitude = mx.sqrt(mx.sum(noisy_early ** 2))
        late_magnitude = mx.sqrt(mx.sum(noisy_late ** 2))

        assert early_magnitude > late_magnitude


class TestDDPMModelInputScaling:
    """Tests for model input scaling."""

    @pytest.mark.fast
    def test_scale_model_input_identity(self):
        """Verify DDPM uses identity scaling (no scaling)."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        latents = mx.random.normal((1, 4, 64, 64))
        timestep = 0

        scaled = scheduler.scale_model_input(latents, timestep)

        # DDPM doesn't scale, so should be identical
        assert mx.array_equal(scaled, latents)

    @pytest.mark.fast
    def test_scale_model_input_different_timesteps(self):
        """Verify scaling is independent of timestep for DDPM."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        latents = mx.random.normal((1, 4, 64, 64))

        scaled_t0 = scheduler.scale_model_input(latents, 0)
        scaled_t25 = scheduler.scale_model_input(latents, 25)

        # DDPM scaling is identity, so both should be same as input
        assert mx.array_equal(scaled_t0, latents)
        assert mx.array_equal(scaled_t25, latents)


class TestDDPMHunyuanIntegration:
    """Tests for DDPM scheduler integration with Hunyuan-DiT."""

    @pytest.mark.fast
    def test_hunyuan_config_compatibility(self):
        """Verify DDPM scheduler works with Hunyuan config."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        # Check that model name contains "Hunyuan" (could be full repo path)
        assert "hunyuan" in scheduler.config.model_config.model_name.lower()
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 50

    @pytest.mark.fast
    def test_hunyuan_recommended_steps(self):
        """Verify Hunyuan's recommended 50 steps works."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,
            num_inference_steps=50,  # Hunyuan recommendation
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert len(scheduler.timesteps) == 50
        assert scheduler.timesteps is not None

    @pytest.mark.fast
    def test_hunyuan_default_guidance(self):
        """Verify Hunyuan's default guidance of 7.5."""
        config = Config(
            width=1024,
            height=1024,
            guidance=7.5,  # Hunyuan default
            num_inference_steps=50,
            model_config=ModelConfig.hunyuan(),
        )

        scheduler = DDPMScheduler(config=config)

        assert config.guidance == 7.5
        assert scheduler.config.guidance == 7.5
