from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.utils.scale_factor import ScaleFactor


class SeedVR2Util:
    @staticmethod
    def preprocess_image(
        image_path: str | Path,
        resolution: int | ScaleFactor,
    ) -> tuple[mx.array, int, int]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # 1. Determine target dimensions based on resolution
        if isinstance(resolution, ScaleFactor):
            target_res = resolution.get_scaled_value(min(w, h))
        else:
            target_res = resolution

        scale = target_res / min(w, h)
        target_w = int(w * scale)
        target_h = int(h * scale)
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16

        # 2. Downsample to 1/4 of target resolution
        # I've found that the model is performing better on several test images when we downsample first before upscaling to the target resolution.
        down_w = max(1, target_w // 4)
        down_h = max(1, target_h // 4)
        image = image.resize((down_w, down_h), Image.Resampling.LANCZOS)

        # 3. Resize to final target dimensions
        resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        img_mx = mx.array(np.array(resized)).astype(mx.float32) / 255.0
        img_mx = mx.clip(img_mx, 0.0, 1.0)
        img_mx = img_mx * 2.0 - 1.0
        img_mx = mx.transpose(img_mx, (2, 0, 1))
        img_mx = img_mx[None, ...]
        return img_mx, img_mx.shape[2], img_mx.shape[3]
