import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    def __init__(self, config: "Config"):
        self.config = config
        self.model_config = config.model_config
        self.num_train_timesteps = 1000
        self.shift_terminal = 0.02
        self.base_shift = 0.5
        self.max_shift = 0.9
        self.base_image_seq_len = 256
        self.max_image_seq_len = 8192
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _compute_mu(self) -> float:
        h_patches = self.config.height // 16
        w_patches = self.config.width // 16
        seq_len = h_patches * w_patches
        m = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        b = self.base_shift - m * self.base_image_seq_len
        mu = m * seq_len + b
        return mu

    @staticmethod
    def _time_shift_exponential(mu: float, sigma_power: float, t: float) -> float:
        return math.exp(mu) / (math.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    def _stretch_to_terminal(self, sigmas: list[float]) -> list[float]:
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        num_steps = self.config.num_inference_steps
        sigma_min = 1.0 / self.num_train_timesteps
        sigma_max = 1.0
        timesteps_linear = [
            sigma_max * self.num_train_timesteps
            - i * (sigma_max - sigma_min) * self.num_train_timesteps / (num_steps - 1)
            for i in range(num_steps)
        ]
        sigmas_linear = [t / self.num_train_timesteps for t in timesteps_linear]
        sigmas_shifted = [FlowMatchEulerDiscreteScheduler._time_shift_exponential(1.0, 1.0, s) for s in sigmas_linear]
        sigmas_final = self._stretch_to_terminal(sigmas_shifted)
        timesteps = [s * self.num_train_timesteps for s in sigmas_final]
        sigmas_with_zero = sigmas_final + [0.0]
        sigmas_arr = mx.array(sigmas_with_zero, dtype=mx.float32)
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)
        return sigmas_arr, timesteps_arr

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return latents + dt * noise

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        return latents
