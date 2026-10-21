from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mflux.models.common.schedulers.base_scheduler import BaseScheduler

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config


class MOSSImageScheduler(BaseScheduler):
    # Flow-matching Euler on the schedule MOSS-Image was trained with: shifted sigmas that
    # are snapped to the nearest point of the 1000-step training grid.
    NUM_TRAIN_TIMESTEPS = 1000
    SHIFT = 5.0
    TRAINING_SHIFT = 1.0

    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = MOSSImageScheduler.compute_sigmas(config.num_inference_steps)
        self._timesteps = self._sigmas[:-1] * MOSSImageScheduler.NUM_TRAIN_TIMESTEPS

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return latents.astype(mx.float32) + noise.astype(mx.float32) * dt

    @staticmethod
    def compute_sigmas(num_inference_steps: int) -> mx.array:
        if num_inference_steps < 1:
            raise ValueError(f"MOSS-Image needs at least 1 step, got {num_inference_steps}.")
        total = MOSSImageScheduler.NUM_TRAIN_TIMESTEPS
        sigmas = MOSSImageScheduler._grid(num_inference_steps, MOSSImageScheduler.SHIFT)
        training = MOSSImageScheduler._grid(total, MOSSImageScheduler.TRAINING_SHIFT)[::-1]
        upper = np.clip(np.searchsorted(training, sigmas), None, total - 1)
        lower = np.clip(upper - 1, 0, None)
        nearest = np.where(np.abs(sigmas - training[lower]) <= np.abs(training[upper] - sigmas), lower, upper)
        snapped = training[nearest]
        if len(np.unique(snapped)) != num_inference_steps:
            # Above about 218 steps two steps snap to the same grid point and the reference
            # silently runs fewer steps. Refuse instead, so the step count stays honest.
            raise ValueError(
                f"MOSS-Image cannot run {num_inference_steps} steps: the training grid gives only "
                f"{len(np.unique(snapped))} distinct steps. Use fewer steps."
            )
        return mx.array(np.concatenate([snapped, [0.0]]).astype(np.float32))

    @staticmethod
    def _grid(count: int, shift: float) -> np.ndarray:
        total = MOSSImageScheduler.NUM_TRAIN_TIMESTEPS
        values = np.linspace(1.0, shift / (total + shift - 1), count)
        return shift * values / (1 + (shift - 1) * values)
