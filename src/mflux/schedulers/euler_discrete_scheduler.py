from typing import Optional, Union

import mlx.core as mx
import numpy as np


class EulerDiscreteScheduler:
    """
    Euler scheduler for discrete diffusion models.
    Implements the Euler method for solving the reverse-time SDE.

    NOTE: This is a port of the diffusers.EulerDiscreteScheduler.
    Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: bool = False,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.interpolation_type = interpolation_type
        self.use_karras_sigmas = use_karras_sigmas
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        if beta_schedule == "linear":
            self.betas = mx.linspace(beta_start, beta_end, num_train_timesteps, dtype=mx.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=mx.float32) ** 2
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Create sigmas from alphas_cumprod in ascending order (low noise to high noise)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # Initialize timesteps
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[str] = None):
        """Set the discrete timesteps used for the diffusion chain."""
        self.num_inference_steps = num_inference_steps

        if self.use_karras_sigmas:
            # Karras sigmas are generated in descending order
            sigmas = self._get_karras_sigmas(num_inference_steps)
        else:
            # Sample from the train sigmas in descending order.
            timesteps_indices = mx.linspace(self.num_train_timesteps - 1, 0, num_inference_steps)
            sigmas = self.sigmas[timesteps_indices.astype(mx.int32)]

        # This is the final sigma schedule for inference (descending order)
        self.sigmas = mx.concatenate([sigmas, mx.array([0.0])])
        self.timesteps = mx.arange(num_inference_steps, dtype=mx.int32)

    def _get_karras_sigmas(self, num_inference_steps: int) -> mx.array:
        """Get Karras et al. (2022) timestep schedule."""
        rho = 7.0
        ramp = mx.linspace(0, 1, num_inference_steps)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def scale_model_input(self, sample: mx.array, timestep: Union[int, mx.array]) -> mx.array:
        """Scale the denoising model input."""
        if isinstance(timestep, mx.array):
            timestep = int(timestep.item())

        sigma = self.sigmas[timestep]
        return sample / ((sigma**2 + self.sigma_data**2) ** 0.5)

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        generator: Optional[int] = None,
        return_dict: bool = True,
    ) -> mx.array:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        """
        if timestep >= len(self.sigmas) - 1:
            return sample

        sigma = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1]

        if self.prediction_type == "epsilon":
            denoised = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            # The v_prediction formula was also slightly off, correcting it.
            denoised = self.alphas_cumprod[timestep]**0.5 * sample - (1 - self.alphas_cumprod[timestep])**0.5 * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        derivative = (sample - denoised) / sigma
        dt = sigma_next - sigma
        prev_sample = sample + derivative * dt

        return prev_sample
