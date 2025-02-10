import os
from pathlib import Path

import PIL.Image

from mflux import ImageUtil
from mflux.callbacks.callback import BeforeLoopCallback


class CannyImageSaver(BeforeLoopCallback):
    def __init__(self, path: str):
        self.path = Path(path)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        canny_image: PIL.Image.Image | None = None,
    ) -> None:  # fmt: off
        base, ext = os.path.splitext(self.path)
        ImageUtil.save_image(
            image=canny_image,
            path=f"{base}_controlnet_canny{ext}"
        )  # fmt: off
