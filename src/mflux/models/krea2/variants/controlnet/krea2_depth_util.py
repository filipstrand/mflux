from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.image_util import ImageUtil


class Krea2DepthUtil:
    # The depth ControlNet was trained on Depth-Anything-V2 maps: near is white, far is black. Depth Pro's
    # depth_image follows the same convention, so either source goes to the VAE as it is.

    @staticmethod
    def encode_depth_control(
        vae: QwenVAE,
        depth_pro: DepthPro | None,
        width: int,
        height: int,
        image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        tiling_config=None,
    ) -> tuple[mx.array, PIL.Image.Image]:
        depth_image = Krea2DepthUtil.get_depth_image(
            depth_pro=depth_pro,
            image_path=image_path,
            depth_image_path=depth_image_path,
        )
        scaled = ImageUtil.scale_to_dimensions(
            image=depth_image.convert("RGB"),
            target_width=width,
            target_height=height,
        )
        encoded = VAEUtil.encode(vae=vae, image=ImageUtil.to_array(scaled), tiling_config=tiling_config)
        return Krea2LatentCreator.pack_latents(encoded, height, width), depth_image

    @staticmethod
    def get_depth_image(
        depth_pro: DepthPro | None,
        image_path: str | Path | None,
        depth_image_path: str | Path | None,
    ) -> PIL.Image.Image:
        if depth_image_path:
            if not Path(depth_image_path).exists():
                raise FileNotFoundError(f"Depth map file not found: {depth_image_path}")
            return ImageUtil.load_image(depth_image_path)
        if not image_path:
            raise ValueError("Give --depth-image-path, or --image-path to estimate the depth with Depth Pro.")
        if depth_pro is None:
            raise ValueError("Depth Pro is needed to estimate the depth when no depth map is given.")
        return depth_pro.create_depth_map(image_path=image_path).depth_image
