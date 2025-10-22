from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig

from mflux.schedulers.base_scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    """
    Linear scheduler - the default/classic scheduler used in mflux.
    Creates a linear schedule from 1.0 to 1/num_steps.
    """

    def __init__(self, runtime_config: "RuntimeConfig"):
        self.runtime_config = runtime_config
        self._sigmas = self._get_sigmas()
        self._timesteps = self._get_timesteps()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_sigmas(self) -> mx.array:
        model_config = self.runtime_config.model_config
        sigmas = mx.linspace(
            1.0,
            1.0 / self.runtime_config.num_inference_steps,
            self.runtime_config.num_inference_steps,
        )
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])
        if model_config.requires_sigma_shift:
            y1 = 0.5
            x1 = 256
            m = (1.15 - y1) / (4096 - x1)
            b = y1 - m * x1
            mu = m * self.runtime_config.width * self.runtime_config.height / 256 + b
            mu = mx.array(mu)
            shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
            shifted_sigmas[-1] = 0
            return shifted_sigmas
        else:
            return sigmas

    def _get_timesteps(self) -> mx.array:
        """
        Generate timesteps as indices for the linear scheduler.
        Returns indices [0, 1, 2, ..., num_steps-1] to match the loop in txt2img.
        """
        num_steps = self.runtime_config.num_inference_steps
        timesteps = mx.arange(num_steps, dtype=mx.float32)

        return timesteps

    def step(self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return sample + model_output * dt
