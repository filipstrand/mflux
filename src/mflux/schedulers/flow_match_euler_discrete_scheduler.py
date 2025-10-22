import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    """
    Flow Matching Euler Discrete Scheduler for Qwen models.

    Implements the scheduler used by Qwen-Image-Edit with:
    - Dynamic resolution-dependent time shifting
    - Exponential time transformation
    - Terminal stretching to ensure proper end timestep

    Based on diffusers.schedulers.FlowMatchEulerDiscreteScheduler
    """

    def __init__(self, runtime_config: "RuntimeConfig"):
        self.runtime_config = runtime_config
        self.model_config = runtime_config.model_config

        # Scheduler configuration (matching Qwen's config)
        self.num_train_timesteps = 1000
        self.shift_terminal = 0.02

        # Dynamic shifting parameters (from Qwen scheduler config)
        self.base_shift = 0.5
        self.max_shift = 0.9  # From scheduler_config.json
        self.base_image_seq_len = 256
        self.max_image_seq_len = 8192  # From scheduler_config.json

        # Generate timesteps and sigmas
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _compute_mu(self) -> float:
        """
        Compute resolution-dependent mu for dynamic time shifting.

        For Qwen, mu is computed based on the image sequence length:
        - base_image_seq_len = 256 (corresponds to 256x256 image)
        - max_image_seq_len = 4096 (corresponds to 1024x1024 image)
        - Linear interpolation between base_shift and max_shift
        """
        # Calculate sequence length from image dimensions
        # Image patches: (height // 16) * (width // 16)
        h_patches = self.runtime_config.height // 16
        w_patches = self.runtime_config.width // 16
        seq_len = h_patches * w_patches

        # Linear interpolation: mu = m * seq_len + b
        m = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        b = self.base_shift - m * self.base_image_seq_len
        mu = m * seq_len + b

        return mu

    def _time_shift_exponential(self, mu: float, sigma_power: float, t: float) -> float:
        """
        Apply exponential time shift transformation.

        Args:
            mu: The shift amount (resolution-dependent)
            sigma_power: The exponent to apply (typically 1.0)
            t: The sigma value to transform

        Formula: exp(mu) / (exp(mu) + (1/t - 1) ** sigma_power)

        This matches PyTorch's implementation exactly.
        """
        return math.exp(mu) / (math.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    def _stretch_to_terminal(self, sigmas: list[float]) -> list[float]:
        """
        Stretch the sigma schedule to ensure it terminates at shift_terminal.

        This ensures the last timestep equals shift_terminal (e.g., 0.02 → timestep 20).
        """
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """
        Compute the full timestep and sigma schedule.

        Pipeline:
        1. Generate linear sigmas from 1.0 to 1/num_steps
        2. Apply exponential time shifting (resolution-dependent)
        3. Stretch to terminal value
        4. Convert to timesteps
        5. Append zero sigma at the end
        """
        num_steps = self.runtime_config.num_inference_steps

        # Step 1: Generate linear sigmas
        # Match PyTorch exactly: linspace(sigma_max, sigma_min) where sigma_min = 1/num_train_timesteps
        # PyTorch does: timesteps = linspace(1000, 1, 20); sigmas = timesteps / 1000
        # This creates [1.0, 0.9484, ..., 0.001] for 20 steps
        sigma_min = 1.0 / self.num_train_timesteps  # 0.001 for 1000 timesteps
        sigma_max = 1.0

        # Create linear timesteps from max to min, then convert to sigmas
        timesteps_linear = [
            sigma_max * self.num_train_timesteps
            - i * (sigma_max - sigma_min) * self.num_train_timesteps / (num_steps - 1)
            for i in range(num_steps)
        ]
        sigmas_linear = [t / self.num_train_timesteps for t in timesteps_linear]

        # Step 2: Apply exponential time shift with resolution-dependent mu
        # 🔧 CRITICAL: Use mu=1.0 to match PyTorch's pipeline behavior!
        # The dynamic mu computation (_compute_mu) produces mu=0.6935 for 1024x1024,
        # but PyTorch's pipeline actually passes mu=1.0 to set_timesteps()
        mu = 1.0  # Fixed value matching PyTorch pipeline
        # mu = self._compute_mu()  # Dynamic computation (not used in PyTorch)
        # Match PyTorch: time_shift(mu, sigma_power=1.0, t=each_sigma)
        sigmas_shifted = [self._time_shift_exponential(mu, 1.0, s) for s in sigmas_linear]

        # Step 3: Stretch to terminal value
        sigmas_final = self._stretch_to_terminal(sigmas_shifted)

        # Step 4: Convert to timesteps
        timesteps = [s * self.num_train_timesteps for s in sigmas_final]

        # Step 5: Append zero sigma for final step
        sigmas_with_zero = sigmas_final + [0.0]

        sigmas_arr = mx.array(sigmas_with_zero, dtype=mx.float32)
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)

        return (sigmas_arr, timesteps_arr)

    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array:
        """
        Perform one denoising step using the Euler method.

        Args:
            model_output: The model's predicted noise/velocity
            timestep: Index into the timestep schedule (0, 1, 2, ...)
            sample: Current latent sample

        Returns:
            Denoised sample for the next timestep
        """
        # Get current and next sigma values
        sigma = self._sigmas[timestep]
        sigma_next = self._sigmas[timestep + 1]

        # Compute dt (change in sigma)
        dt = sigma_next - sigma

        # Euler step: x_{t+1} = x_t + dt * v_t
        # This is the same formula as PyTorch's FlowMatchEulerDiscreteScheduler
        prev_sample = sample + dt * model_output

        return prev_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input (no scaling needed for flow matching).
        """
        return latents
