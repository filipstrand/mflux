from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.image_util import ImageUtil


class Krea2ControlnetUtil:
    _HIGH_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I", "F")
    # The depth ControlNet was trained on Depth-Anything-V2 maps: near is white, far is black. Depth Pro's
    # depth_image follows the same convention, so either source goes to the VAE as it is.

    @staticmethod
    def encode_depth_control(
        vae: QwenVAE,
        depth_image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> tuple[mx.array, PIL.Image.Image]:
        # Always one untiled pass, even with --low-ram: a tiled encode changes the control latent enough to move
        # the image (about 20 dB at 1024x1024), and the ControlNet was trained on whole-image encodes.
        scaled = ImageUtil.scale_to_dimensions(
            image=Krea2ControlnetUtil._to_rgb(depth_image),
            target_width=width,
            target_height=height,
        )
        encoded = VAEUtil.encode(vae=vae, image=ImageUtil.to_array(scaled))
        return Krea2LatentCreator.pack_latents(encoded, height, width), depth_image

    @staticmethod
    def _to_rgb(depth_image: PIL.Image.Image) -> PIL.Image.Image:
        # A plain convert("RGB") clips a 16-bit or float map to white. A 16-bit map keeps its values, scaled
        # from 0..65535 to 0..255; an "I" or "F" map has no fixed range, so its own minimum and maximum are used.
        if depth_image.mode in Krea2ControlnetUtil._HIGH_DEPTH_MODES:
            values = np.asarray(depth_image, dtype=np.float32)
            if depth_image.mode.startswith("I;16"):
                scaled = values / 65535.0
            else:
                low, high = float(values.min()), float(values.max())
                scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
            depth_image = PIL.Image.fromarray((scaled * 255).round().astype(np.uint8))
        return depth_image.convert("RGB")

    @staticmethod
    def get_depth_image(
        depth_pro: DepthPro | None,
        image_path: str | Path | None,
        depth_image_path: str | Path | None,
    ) -> PIL.Image.Image:
        if depth_image_path:
            if not Path(depth_image_path).exists():
                raise FileNotFoundError(f"Depth map file not found: {depth_image_path}")
            # A 16-bit or float map is kept as it is, for _to_rgb to stretch; load_image would clip it to RGB first.
            image = open_oriented(depth_image_path)
            if image.mode in Krea2ControlnetUtil._HIGH_DEPTH_MODES:
                return image
            return ImageUtil.load_image(depth_image_path)
        if not image_path:
            raise ValueError(
                "Pass a depth map (depth_image_path), or a photo (image_path) for Depth Pro to estimate one."
            )
        if depth_pro is None:
            raise ValueError("Depth Pro is needed to estimate the depth when no depth map is given.")
        return depth_pro.create_depth_map(image_path=image_path).depth_image
