import mlx.core as mx
import numpy as np

from flux_1.config.config import Config, ConfigImg2Img
from flux_1.config.model_config import ModelConfig


class RuntimeConfig:

    def __init__(self, config: Config | ConfigImg2Img, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)
        self._height = config.height
        self._width = config.width

    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        self._width = value

    @property
    def guidance(self):
        return self.config.guidance

    @property
    def num_total_denoising_steps(self):
        return self.config.num_total_denoising_steps

    @property
    def precision(self):
        return self.config.precision

    @property
    def num_train_steps(self):
        return self.model_config.num_train_steps
    
    @property
    def inference_steps(self):
        return self.config.inference_steps
    
    @property
    def strength(self):
        if hasattr(self.config, "strength"):
            return self.config.strength
        else:
            raise AttributeError("Strength is not defined in Config. Please use ConfigImg2Img instead.")

    @staticmethod
    def _create_sigmas(config, model):
        sigmas = RuntimeConfig._create_sigmas_values(config.num_total_denoising_steps)
        if model == ModelConfig.FLUX1_DEV:
            sigmas = RuntimeConfig._shift_sigmas(sigmas, config.width, config.height)
        return sigmas

    @staticmethod
    def _create_sigmas_values(num_total_denoising_steps: int) -> mx.array:
        sigmas = np.linspace(1.0, 1 / num_total_denoising_steps, num_total_denoising_steps)
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
