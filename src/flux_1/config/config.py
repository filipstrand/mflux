import logging

import mlx.core as mx
from sympy import N

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
    ):
        if num_inference_steps < 1:
            log.warning("Number of inference steps should be an integer greater than 0. Setting to 1.")
            num_inference_steps = 1
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_total_denoising_steps = num_inference_steps
        self.inference_steps = list(range(num_inference_steps))
        self.guidance = guidance

class ConfigImg2Img:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
            strength: float = 0.5
    ):
        if num_inference_steps < 1:
            log.warning("Number of inference steps should be an integer greater than 0. Setting to 1.")
            num_inference_steps = 1
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.num_inference_steps = num_inference_steps
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.guidance = guidance

        self.strength = strength
        if strength <= 0.0 or strength >= 1.0:
            raise ValueError("Strength should be a float between 0 and 1.")

        self.num_total_denoising_steps = int(self.num_inference_steps / (1-strength))
        self.init_timestep = int(self.num_total_denoising_steps - self.num_inference_steps) 
        self.inference_steps = list(range(self.num_total_denoising_steps))[self.init_timestep:]




