"""Image preprocessing for i2L vision encoders.

SigLIP2-G384: resize to 384x384, normalize with mean=0.5, std=0.5
DINOv3-7B: resize to 224x224, normalize with ImageNet mean/std

Both resize with center crop to maintain aspect ratio, matching
the DiffSynth-Studio ImageCropAndResize operator.
"""

import mlx.core as mx
import numpy as np
from PIL import Image


def _center_crop_and_resize(image: Image.Image, size: int) -> Image.Image:
    """Center crop to square then resize, matching DiffSynth-Studio ImageCropAndResize."""
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    image = image.resize((size, size), Image.LANCZOS)
    return image


def preprocess_for_siglip2(image: Image.Image, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Preprocess image for SigLIP2-G384 encoder.

    Args:
        image: PIL Image (any size, RGB).
        dtype: Output dtype.

    Returns:
        Tensor of shape (1, 3, 384, 384), normalized to [-1, 1].
    """
    image = image.convert("RGB")
    image = _center_crop_and_resize(image, 384)

    # To numpy [H, W, 3] float32 in [0, 1]
    pixels = np.array(image).astype(np.float32) / 255.0

    # Normalize: (pixel - 0.5) / 0.5 = pixel * 2 - 1
    pixels = pixels * 2.0 - 1.0

    # To (1, 3, H, W)
    pixels = np.transpose(pixels, (2, 0, 1))[np.newaxis, ...]

    return mx.array(pixels).astype(dtype)


def preprocess_for_dinov3(image: Image.Image, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Preprocess image for DINOv3-7B encoder.

    Args:
        image: PIL Image (any size, RGB).
        dtype: Output dtype.

    Returns:
        Tensor of shape (1, 3, 224, 224), normalized with ImageNet mean/std.
    """
    image = image.convert("RGB")
    image = _center_crop_and_resize(image, 224)

    # To numpy [H, W, 3] float32 in [0, 1]
    pixels = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std

    # To (1, 3, H, W)
    pixels = np.transpose(pixels, (2, 0, 1))[np.newaxis, ...]

    return mx.array(pixels).astype(dtype)
