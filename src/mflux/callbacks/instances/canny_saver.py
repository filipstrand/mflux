import os
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux import ImageUtil
from mflux.callbacks.callback import BeforeLoopCallback
from mflux.config.runtime_config import RuntimeConfig


class CannyImageSaver(BeforeLoopCallback):
    def __init__(self, path: str):
        self.path = Path(path)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        base, ext = os.path.splitext(self.path)
        ImageUtil.save_image(
            image=canny_image,
            path=f"{base}_controlnet_canny{ext}",
        )
