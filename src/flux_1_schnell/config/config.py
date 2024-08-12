import mlx.core as mx
import numpy as np


class Config:
    precision: mx.Dtype = mx.float16
    num_train_steps = 1000

    def __init__(
            self,
            num_inference_steps: int = 4,
    ):
        base_sigmas = Config.base_sigmas(num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.time_steps = base_sigmas * self.num_train_steps
        self.sigmas = mx.concatenate([base_sigmas, mx.zeros(1)])

    @staticmethod
    def base_sigmas(num_inference_steps):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return sigmas
