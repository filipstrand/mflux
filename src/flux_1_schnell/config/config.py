from dataclasses import dataclass
import mlx.core as mx
import numpy as np
import logging

log = logging.getLogger(__name__)


def get_sigmas(num_inference_steps):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    sigmas = mx.array(sigmas).astype(mx.float32)
    return mx.concatenate([sigmas, mx.zeros(1)])

def shift_sigmas(sigmas, width, height):
    y1 = 0.5
    x1 = 256
    m = (1.15 - y1) / (4096 - x1)
    b = y1 - m * x1
    mu = m * width * height / 256   + b
    shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
    shifted_sigmas[-1] = 0
    return shifted_sigmas


@dataclass
class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_train_steps: int = 1000,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
    ):
        self.num_train_steps = num_train_steps
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (height // 16)
        self.height = 16 * (width // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance

    def __post_init__(self, **data):
        super().__init__(**data)
        self.__config__.frozen = True
    
    
