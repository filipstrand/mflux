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

    @property
    def sigmas(self) -> mx.array:
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
