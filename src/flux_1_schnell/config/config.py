import mlx.core as mx
import numpy as np
import logging

log = logging.getLogger(__name__)

class Config:
    precision: mx.Dtype = mx.float16
    num_train_steps = 1000

    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (height // 16)
        self.height = 16 * (width // 16)
        self.sigmas = Config.base_sigmas(num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance

    @staticmethod
    def base_sigmas(num_inference_steps):
        sigmas = np.linspace(1.0, 0, num_inference_steps+1)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return sigmas
    
    def shift_sigmas(self):
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * self.width * self.height / 256   + b
        self.sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / self.sigmas - 1))
