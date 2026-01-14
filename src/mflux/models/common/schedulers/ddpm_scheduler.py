"""DDPM (Denoising Diffusion Probabilistic Models) Scheduler.

Used by Hunyuan-DiT and other DDPM-based diffusion models.

Key characteristics:
- Linear beta schedule for noise addition
- Predicts noise (epsilon) rather than velocity
- Uses cumulative product of alphas for forward diffusion
"""

import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class DDPMScheduler(BaseScheduler):
    """DDPM scheduler for noise prediction-based diffusion models.

    Args:
        config: Model configuration with num_inference_steps
        beta_start: Starting value of beta schedule (default 0.00085)
        beta_end: Ending value of beta schedule (default 0.012)
        num_train_timesteps: Number of training timesteps (default 1000)
        prediction_type: Type of prediction ("epsilon" or "v_prediction")
    """

    def __init__(
        self,
        config: "Config",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        num_train_timesteps: int = 1000,
        prediction_type: str = "epsilon",
    ):
        self.config = config
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Compute betas using scaled linear schedule (same as HuggingFace diffusers)
        self.betas = self._compute_betas(beta_start, beta_end, num_train_timesteps)

        # Compute alphas and cumulative products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        # Compute sigmas for compatibility with base scheduler
        self._sigmas = self._compute_sigmas()
        self._timesteps = self._compute_timesteps()

    @property
    def sigmas(self) -> mx.array:
        """Return sigmas for compatibility with scheduler interface."""
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return timesteps for diffusion process."""
        return self._timesteps

    def _compute_betas(
        self,
        beta_start: float,
        beta_end: float,
        num_train_timesteps: int,
    ) -> mx.array:
        """Compute beta schedule using scaled linear schedule.

        Uses sqrt scaling for more stable training dynamics.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value
            num_train_timesteps: Total training timesteps

        Returns:
            Beta values as array [num_train_timesteps]
        """
        # Scaled linear schedule (square root scaling)
        betas = mx.linspace(
            math.sqrt(beta_start),
            math.sqrt(beta_end),
            num_train_timesteps,
        )
        betas = betas ** 2
        return betas.astype(mx.float32)

    def _compute_sigmas(self) -> mx.array:
        """Compute sigmas from alphas_cumprod for scheduler interface.

        Returns:
            Sigmas array [num_inference_steps + 1]
        """
        # sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        alphas_cumprod_selected = self._get_alphas_at_timesteps()

        sigmas = mx.sqrt((1 - alphas_cumprod_selected) / alphas_cumprod_selected)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        return sigmas.astype(mx.float32)

    def _compute_timesteps(self) -> mx.array:
        """Compute timesteps for inference.

        Returns:
            Timesteps array [num_inference_steps] in descending order
        """
        num_steps = self.config.num_inference_steps
        step_ratio = self.num_train_timesteps // num_steps

        # Create timesteps in descending order (from T-1 to 0)
        timesteps = mx.arange(0, num_steps) * step_ratio
        timesteps = timesteps[::-1]  # Reverse to go from high noise to low

        return timesteps.astype(mx.int32)

    def _get_alphas_at_timesteps(self) -> mx.array:
        """Get alphas_cumprod values at selected inference timesteps.

        Returns:
            Alphas cumprod at inference timesteps
        """
        num_steps = self.config.num_inference_steps
        step_ratio = self.num_train_timesteps // num_steps

        # Get indices for inference timesteps
        indices = mx.arange(0, num_steps) * step_ratio
        indices = indices[::-1]  # Reverse order

        return self.alphas_cumprod[indices.astype(mx.int32)]

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        """Perform one denoising step.

        Args:
            noise: Predicted noise (or velocity depending on prediction_type)
            timestep: Current timestep index (in inference timesteps, not training)
            latents: Current noisy latents

        Returns:
            Denoised latents after one step
        """
        # Get the actual training timestep
        t = self._timesteps[timestep].item()
        t = int(t)

        # Get alpha values for current and previous timestep
        alpha_prod_t = self.alphas_cumprod[t]

        # For the last step, use alpha=1 (no noise remaining)
        if timestep + 1 < len(self._timesteps):
            t_prev = self._timesteps[timestep + 1].item()
            t_prev = int(t_prev)
            alpha_prod_t_prev = self.alphas_cumprod[t_prev]
        else:
            alpha_prod_t_prev = mx.array(1.0)

        # Compute predicted original sample (x_0)
        if self.prediction_type == "epsilon":
            # x_0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
            pred_original_sample = (
                latents - mx.sqrt(1 - alpha_prod_t) * noise
            ) / mx.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            # x_0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
            pred_original_sample = (
                mx.sqrt(alpha_prod_t) * latents - mx.sqrt(1 - alpha_prod_t) * noise
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip predicted sample to valid range
        pred_original_sample = mx.clip(pred_original_sample, -1.0, 1.0)

        # Compute coefficients for DDPM sampling
        # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1 - alpha_{t-1}) * noise_direction
        pred_sample_direction = mx.sqrt(1 - alpha_prod_t_prev) * noise

        # Compute previous sample
        prev_sample = (
            mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        return prev_sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timestep: int,
    ) -> mx.array:
        """Add noise to samples for a given timestep.

        Used for img2img and inpainting.

        Args:
            original_samples: Clean samples to add noise to
            noise: Random noise
            timestep: Training timestep (not inference index)

        Returns:
            Noisy samples
        """
        alpha_prod_t = self.alphas_cumprod[timestep]

        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        noisy_samples = (
            mx.sqrt(alpha_prod_t) * original_samples
            + mx.sqrt(1 - alpha_prod_t) * noise
        )

        return noisy_samples

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """Scale model input (identity for DDPM).

        Args:
            latents: Input latents
            t: Timestep (unused for DDPM)

        Returns:
            Unscaled latents (DDPM doesn't require input scaling)
        """
        return latents
