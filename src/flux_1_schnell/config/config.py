import logging

import mlx.core as mx
import numpy as np

from flux_1_schnell.config.model_config import ModelConfig

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_train_steps: int = 1000,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
            sigmas: mx.array | None = None,
    ):
        self.num_train_steps = num_train_steps
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (height // 16)
        self.height = 16 * (width // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.sigmas = sigmas

    def copy_with_sigmas(self, model: ModelConfig) -> "Config":
        sigmas = Config._get_sigmas(self.num_inference_steps)
        if model == ModelConfig.FLUX1_DEV:
            sigmas = Config._shift_sigmas(sigmas, self.width, self.height)

        return Config(
            num_train_steps=self.num_train_steps,
            num_inference_steps=self.num_inference_steps,
            width=self.width,
            height=self.height,
            guidance=self.guidance,
            sigmas=sigmas,
        )

    @staticmethod
    def _get_sigmas(num_inference_steps):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int):
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas
