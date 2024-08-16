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
    ):
        if width %16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (height // 16)
        self.height = 16 * (width // 16)
        base_sigmas = Config.base_sigmas(num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.time_steps = base_sigmas * self.num_train_steps
        self.sigmas = mx.concatenate([base_sigmas, mx.zeros(1)])

    @staticmethod
    def base_sigmas(num_inference_steps):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return sigmas
