import logging
import os

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil

log = logging.getLogger(__name__)


class ControlnetUtil:
    @staticmethod
    def encode_image(
        vae: VAE,
        config: RuntimeConfig,
        controlnet_image_path: str,
        controlnet_save_canny: bool,
        output: str,
    ) -> mx.array:
        from mflux import ImageUtil

        control_image = ImageUtil.load_image(controlnet_image_path)
        control_image = ControlnetUtil._scale_image(config.height, config.width, control_image)
        control_image = ControlnetUtil._preprocess_canny(control_image)

        if controlnet_save_canny:
            base, ext = os.path.splitext(output)
            ImageUtil.save_image(
                image=control_image,
                path=f"{base}_controlnet_canny{ext}"
            )  # fmt: off

        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = vae.encode(controlnet_cond)
        controlnet_cond = (controlnet_cond / vae.scaling_factor) + vae.shift_factor
        controlnet_cond = ArrayUtil.pack_latents(latents=controlnet_cond, height=config.height, width=config.width)
        return controlnet_cond

    @staticmethod
    def _preprocess_canny(img: PIL.Image) -> PIL.Image:
        image_to_canny = np.array(img)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = np.array(image_to_canny[:, :, None])
        image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
        return PIL.Image.fromarray(image_to_canny)

    @staticmethod
    def _scale_image(height: int, width: int, img: PIL.Image) -> PIL.Image:
        if height != img.height or width != img.width:
            log.warning(f"Control image has different dimensions than the model. Resizing to {width}x{height}")
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img
