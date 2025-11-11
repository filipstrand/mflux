import math
from typing import Optional, Union

import numpy as np
from PIL import Image

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenImageProcessor:
    def __init__(
        self,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        height, width = image.size[1], image.size[0]

        if resized_height is None or resized_width is None:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        if (height, width) != (resized_height, resized_width):
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        image_np = np.array(image).astype(np.float32)
        image_np = image_np / 255.0

        mean_np = np.array(self.image_mean, dtype=np.float32)
        std_np = np.array(self.image_std, dtype=np.float32)
        image_np = (image_np - mean_np) / std_np

        image_np = image_np.transpose(2, 0, 1)
        patches = image_np[np.newaxis]  # Shape: (1, channel, height, width)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: Union[Image.Image, list[Image.Image]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(images, list):
            images = [images]

        pixel_values_list = []
        vision_grid_thws = []

        for image in images:
            patches, image_grid_thw = self._preprocess(image)
            pixel_values_list.append(patches)
            vision_grid_thws.append([image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]])

        # Concatenate all patches from all images along the patch dimension
        pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else np.array([])

        vision_grid_thws = np.array(vision_grid_thws)

        return pixel_values, vision_grid_thws

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> int:
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        return grid_h * grid_w
