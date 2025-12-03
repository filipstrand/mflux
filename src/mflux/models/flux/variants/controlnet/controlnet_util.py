import logging

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.utils.image_util import StrOrBytesPath

log = logging.getLogger(__name__)


class ControlnetUtil:
    @staticmethod
    def encode_image(
        vae: VAE,
        height: int,
        width: int,
        controlnet_image_path: StrOrBytesPath,
        is_canny: bool,
    ) -> tuple[mx.array, PIL.Image.Image]:
        from mflux.utils.image_util import ImageUtil

        control_image = ImageUtil.load_image(controlnet_image_path)
        control_image = ControlnetUtil._scale_image(height=height, width=width, img=control_image)
        if is_canny:
            control_image = ControlnetUtil._preprocess_canny(control_image)
        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = vae.encode(controlnet_cond)
        if is_canny:
            controlnet_cond = (controlnet_cond / vae.scaling_factor) + vae.shift_factor
        controlnet_cond = FluxLatentCreator.pack_latents(latents=controlnet_cond, height=height, width=width)
        return controlnet_cond, control_image

    @staticmethod
    def _preprocess_canny(img: PIL.Image.Image) -> PIL.Image.Image:
        image_to_canny = np.array(img)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = np.array(image_to_canny[:, :, None])
        image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
        return PIL.Image.fromarray(image_to_canny)

    @staticmethod
    def _scale_image(height: int, width: int, img: PIL.Image.Image) -> PIL.Image.Image:
        if height != img.height or width != img.width:
            log.warning(
                f"Control image {img.width}x{img.height} has different dimensions than the model requirements or requested width x height. Resizing to {width}x{height}"
            )
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img
