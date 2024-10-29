import logging

import mlx.core as mx
import numpy as np

from mflux.config.config import Config, ConfigControlnet
from mflux.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class RuntimeConfig:
    def __init__(self, config: Config | ConfigControlnet, model_config):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def guidance(self) -> float:
        return self.config.guidance

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def precision(self) -> mx.Dtype:
        return self.config.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps

    @property
    def init_image_path(self) -> str:
        return self.config.init_image_path

    @property
    def init_image_strength(self) -> float:
        return self.config.init_image_strength

    @property
    def init_time_step(self) -> int:
        if self.config.init_image_path is None:
            # text to image, always begin at time step 0
            return 0
        else:
            # we skip to the time step as informed by the init_image_strength
            # the higher the strength number, the more time steps we skip
            strength = max(0.0, min(1.0, self.config.init_image_strength))
            # if the strength is too small to even influence the image
            # help the user round up so the init_image has influence at step 1
            t = max(1, int(self.num_inference_steps * strength))
            return t

    @property
    def controlnet_strength(self) -> float:
        if isinstance(self.config, ConfigControlnet):
            return self.config.controlnet_strength
        else:
            raise NotImplementedError("Controlnet conditioning scale is only available for ConfigControlnet")

    @staticmethod
    def _create_sigmas(config, model) -> mx.array:
        sigmas = RuntimeConfig._create_sigmas_values(config.num_inference_steps)
        if model == ModelConfig.FLUX1_DEV:
            sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=config.width, height=config.height)
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
