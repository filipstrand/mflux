import logging
from pathlib import Path

import mlx.core as mx

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        num_inference_steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4.0,
        init_image_path: Path | None = None,
        init_image_strength: float | None = None,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.init_image_path = init_image_path
        self.init_image_strength = init_image_strength


class ConfigControlnet(Config):
    def __init__(
        self,
        num_inference_steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4.0,
        controlnet_strength: float = 1.0,
    ):
        super().__init__(num_inference_steps=num_inference_steps, width=width, height=height, guidance=guidance)
        self.controlnet_strength = controlnet_strength
