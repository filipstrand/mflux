import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

from mflux.models.depth_pro.depth_pro_initializer import DepthProInitializer
from mflux.models.depth_pro.depth_pro_model import DepthProModel
from mflux.post_processing.image_util import ImageUtil


@dataclass
class DepthResult:
    depth_image: Image.Image
    depth_array: mx.array
    min_depth: float
    max_depth: float


class DepthPro:
    def __init__(self, quantize: int | None = None):
        self._depth_pro_model = DepthProModel()
        DepthProInitializer.init(self._depth_pro_model, quantize=quantize)

    def create_depth_map(self, image_path: str | Path) -> DepthResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        input_array, height, width = self._pre_process(image_path)
        depth = self._depth_pro_model(input_array)
        return self._post_process(depth, height=height, width=width)

    @staticmethod
    def _pre_process(image_path):
        image = Image.open(image_path).convert("RGB")
        input_array = ImageUtil.preprocess_for_depth_pro(image)
        input_array = DepthPro._resize(input_array)
        return input_array, image.height, image.width

    @staticmethod
    def _post_process(depth: mx.array, height: int, width: int):
        depth_min = mx.min(depth)
        depth_max = mx.max(depth)
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        depth_np = np.asarray(normalized_depth.squeeze())
        depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))
        depth_image = depth_image.resize((width, height))
        depth_np = np.array(depth_image) / 255.0
        return DepthResult(
            depth_image=depth_image,
            depth_array=depth,
            min_depth=depth_min.item(),
            max_depth=depth_max.item(),
        )

    @staticmethod
    def _resize(x: mx.array) -> mx.array:
        x_np = np.array(x)
        x_torch = torch.from_numpy(x_np)
        x_torch = x_torch.unsqueeze(0)
        x_torch = torch.nn.functional.interpolate(
            x_torch,
            size=(1536, 1536),
            mode="bilinear",
            align_corners=False,
        )
        return mx.array(x_torch)
