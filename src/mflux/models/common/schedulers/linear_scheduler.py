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

    def _get_sigmas(self) -> mx.array:
        model_config = self.config.model_config
        num_steps = self.config.num_inference_steps
        sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])
        if model_config.requires_sigma_shift:
            m = (model_config.sigma_max_shift - model_config.sigma_base_shift) / (
                model_config.sigma_max_seq_len - model_config.sigma_base_seq_len
            )
            b = model_config.sigma_base_shift - m * model_config.sigma_base_seq_len
            mu = m * self.config.width * self.config.height / 256 + b
            mu = mx.array(mu)

            shifted = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas[:-1] - 1))

            if model_config.sigma_shift_terminal is not None:
                one_minus = 1.0 - shifted
                scale = one_minus[-1] / (1.0 - model_config.sigma_shift_terminal)
                shifted = 1.0 - (one_minus / scale)

            return mx.concatenate([shifted, mx.zeros(1)])
        else:
            return sigmas

    def _get_timesteps(self) -> mx.array:
        num_steps = self.config.num_inference_steps
        timesteps = mx.arange(num_steps, dtype=mx.float32)

        return timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt
