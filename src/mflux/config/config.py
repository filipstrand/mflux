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
        image_path: Path | None = None,
        image_strength: float | None = None,
        depth_image_path: Path | None = None,
        redux_image_paths: list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        masked_image_path: Path | None = None,
        controlnet_strength: float | None = None,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.image_path = image_path
        self.image_strength = image_strength
        self.depth_image_path = depth_image_path
        self.redux_image_paths = redux_image_paths
        self.redux_image_strengths = redux_image_strengths
        self.masked_image_path = masked_image_path
        self.controlnet_strength = controlnet_strength
