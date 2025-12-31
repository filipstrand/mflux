from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class SeedVR2EulerScheduler(BaseScheduler):
    def __init__(self, config: "Config"):
        self.config = config
        self.num_inference_steps = config.num_inference_steps
        self.num_train_timesteps = config.num_train_steps if config.num_train_steps is not None else 1000
        self.cfg_scale = config.guidance
        self.T = float(self.num_train_timesteps)
        self._timesteps, self._sigmas = self._compute_timesteps_and_sigmas()

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        step_size = self.T / self.num_inference_steps
        timesteps = []
        for i in range(self.num_inference_steps + 1):
            t = self.T - i * step_size
            timesteps.append(max(t, 0.0))
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)
        sigmas_arr = timesteps_arr / self.T
        return timesteps_arr, sigmas_arr

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        model_output = noise
        sample = latents
        timestep_idx = timestep
        t = self._timesteps[timestep_idx]
        s = self._timesteps[timestep_idx + 1]
        t_norm = t / self.T
        s_norm = s / self.T
        pred_x_0 = sample - t_norm * model_output
        pred_noise = sample + (1 - t_norm) * model_output
        if s > 0:
            next_sample = (1 - s_norm) * pred_x_0 + s_norm * pred_noise
        else:
            next_sample = pred_x_0
        return next_sample
