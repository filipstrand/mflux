from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mflux.models.depth_pro.depth_pro_model import DepthProModel
from mflux.post_processing.image_util import ImageUtil


@dataclass
class DepthResult:
    depth_image: Image.Image
    depth_array: mx.array
    min_depth: float
    max_depth: float


class DepthPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_pro_model = DepthProModel()

    def __call__(self, image_path: str | Path) -> DepthResult:
        # 1. Process the image
        image = Image.open(image_path).convert("RGB")
        input_array = ImageUtil.preprocess_for_model(image)
        input_array = DepthPro._resize(input_array)

        # 2. Run inference
        depth, _ = self.depth_pro_model(input_array)

        # 3. Process the depth map and convert to PIL image
        depth_min = mx.min(depth)
        depth_max = mx.max(depth)
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        depth_np = normalized_depth.squeeze().cpu().numpy()
        depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))
        return DepthResult(
            depth_image=depth_image,
            depth_array=depth,
            min_depth=depth_min.item(),
            max_depth=depth_max.item(),
        )

    @staticmethod
    def _resize(x: mx.array) -> mx.array:
        # Convert MLX array to PIL Image for resizing
        x_np = np.array(x.squeeze())
        x_np = (x_np.transpose(1, 2, 0) + 1) * 127.5
        pil_image = Image.fromarray(x_np.astype(np.uint8))

        # Use ImageUtil to resize
        resized = ImageUtil.scale_to_dimensions(pil_image, target_width=1536, target_height=1536)

        # Convert back to MLX array
        resized_np = np.array(resized).astype(np.float32)
        resized_np = resized_np.transpose(2, 0, 1)  # HWC -> CHW
        resized_np = resized_np / 127.5 - 1  # Normalize back
        resized_array = mx.array(resized_np)
        resized_array = mx.expand_dims(resized_array, axis=0)  # Add batch dim back

        return resized_array
