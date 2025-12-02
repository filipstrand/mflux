"""Z-Image scheduler with shift=3.0.

This scheduler uses the formula from diffusers FlowMatchEulerDiscreteScheduler
with use_dynamic_shifting=False and shift=3.0.
"""

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig


class ZImageScheduler:
    """Flow matching Euler scheduler for Z-Image with shift=3.0.

    Formula: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    """

    SHIFT = 3.0  # Z-Image uses shift=3.0
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(self, runtime_config: "RuntimeConfig"):
        self.runtime_config = runtime_config
        self.num_inference_steps = runtime_config.num_inference_steps
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """Compute sigmas and timesteps with shift=3.0."""
        num_steps = self.num_inference_steps

        # Linear spacing from 1.0 to 1/num_steps
        sigmas_linear = [1.0 - i / num_steps for i in range(num_steps + 1)]

        # Apply shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        sigmas_shifted = []
        for s in sigmas_linear:
            if s > 0:
                shifted = self.SHIFT * s / (1 + (self.SHIFT - 1) * s)
            else:
                shifted = 0.0
            sigmas_shifted.append(shifted)

        # Convert to timesteps (sigma * num_train_timesteps)
        timesteps = [s * self.NUM_TRAIN_TIMESTEPS for s in sigmas_shifted[:-1]]

        sigmas_arr = mx.array(sigmas_shifted, dtype=mx.float32)
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)

        return sigmas_arr, timesteps_arr

    def step(self, model_output: mx.array, timestep_idx: int, sample: mx.array) -> mx.array:
        """Euler step: x_next = x + dt * velocity.

        Args:
            model_output: Velocity prediction from transformer
            timestep_idx: Index into timesteps array
            sample: Current sample

        Returns:
            Updated sample
        """
        dt = self._sigmas[timestep_idx + 1] - self._sigmas[timestep_idx]
        prev_sample = sample + dt * model_output
        return prev_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """Scale model input (identity for flow matching)."""
        return latents
