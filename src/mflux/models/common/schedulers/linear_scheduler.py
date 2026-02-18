import math
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = self._get_sigmas()
        self._timesteps = self._get_timesteps()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @staticmethod
    def _generate_base_sigmas(num_steps: int, schedule: str = "linear") -> mx.array:
        sigma_max = 1.0
        sigma_min = 1.0 / num_steps
        if schedule == "cosine":
            t = mx.linspace(0, 1, num_steps)
            sigmas = (1.0 + mx.cos(t * math.pi)) / 2.0
        elif schedule == "karras":
            rho = 7.0
            ramp = mx.linspace(0, 1, num_steps)
            min_inv_rho = sigma_min ** (1.0 / rho)
            max_inv_rho = sigma_max ** (1.0 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        elif schedule == "exponential":
            sigmas = mx.exp(mx.linspace(math.log(sigma_max), math.log(sigma_min), num_steps))
        else:
            sigmas = mx.linspace(sigma_max, sigma_min, num_steps)
        return sigmas.astype(mx.float32)

    def _get_sigmas(self) -> mx.array:
        model_config = self.config.model_config
        sigmas = self._generate_base_sigmas(self.config.num_inference_steps, self.config.sigma_schedule)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])
        if model_config.requires_sigma_shift:
            y1 = 0.5
            x1 = 256
            m = (1.15 - y1) / (4096 - x1)
            b = y1 - m * x1
            mu = m * self.config.width * self.config.height / 256 + b
            if self.config.shift is not None:
                mu = self.config.shift
            mu = mx.array(mu)
            shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
            shifted_sigmas[-1] = 0
            return shifted_sigmas
        else:
            return sigmas

    def _get_timesteps(self) -> mx.array:
        num_steps = self.config.num_inference_steps
        timesteps = mx.arange(num_steps, dtype=mx.float32)

        return timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt
