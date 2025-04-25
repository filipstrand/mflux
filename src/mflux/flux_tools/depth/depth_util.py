import logging
import os
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil

logger = logging.getLogger(__name__)


class DepthUtil:
    @staticmethod
    def encode_depth_map(
        vae: VAE,
        depth_pro: DepthPro,
        config: RuntimeConfig,
        image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
    ) -> tuple[mx.array, PIL.Image.Image]:
        # 1. Create the depth map or use existing one
        depth_image_path, depth_image = DepthUtil.get_or_create_depth_map(
            depth_pro=depth_pro,
            image_path=image_path,
            depth_map_path=depth_image_path,
        )

        # 2. Encode the depth map
        scaled_depth_map = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(depth_image_path).convert("RGB"),
            target_width=config.width,
            target_height=config.height,
        )
        depth_map_array = ImageUtil.to_array(scaled_depth_map)
        encoded_depth = vae.encode(depth_map_array)
        depth_latents = ArrayUtil.pack_latents(latents=encoded_depth, height=config.height, width=config.width)

        return depth_latents, depth_image

    @staticmethod
    def get_or_create_depth_map(
        depth_pro: DepthPro,
        image_path: str | Path | None = None,
        depth_map_path: str | Path | None = None,
    ) -> tuple[str | Path, PIL.Image.Image | None]:
        # 1. If a depth map path is provided, use it directly
        if depth_map_path:
            if not os.path.exists(depth_map_path):
                raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")
            return depth_map_path, None
        if not image_path:
            raise ValueError("Either depth_map_path or image_path must be provided")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 2. Generate a depth map from the image using the DepthPro model
        depth_result = depth_pro(image_path)

        # 3. Save the depth map to a file with the same name + _depth suffix
        generated_depth_path = str(image_path).rsplit(".", 1)[0] + "_depth.png"
        depth_result.depth_image.save(generated_depth_path)
        return generated_depth_path, depth_result.depth_image
