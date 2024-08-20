import mlx.core as mx
import numpy as np

from flux_1.config.config import Config
from flux_1.config.model_config import ModelConfig


class RuntimeConfig:

    def __init__(self, config: Config, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)

    @property
    def height(self):
        return self.config.height

    @property
    def width(self):
        return self.config.width

    @property
    def guidance(self):
        return self.config.guidance

    @property
    def num_inference_steps(self):
        return self.config.num_inference_steps

    @property
    def precision(self):
        return self.config.precision

    @property
    def num_train_steps(self):
        return self.model_config.num_train_steps

    @staticmethod
    def _create_sigmas(config, model):
        sigmas = RuntimeConfig._create_sigmas_values(config.num_inference_steps)
        if model == ModelConfig.FLUX1_DEV:
            sigmas = RuntimeConfig._shift_sigmas(sigmas, config.width, config.height)
        return sigmas

    @staticmethod
    def _create_sigmas_values(num_inference_steps: int) -> mx.array:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int) -> mx.array:
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas
