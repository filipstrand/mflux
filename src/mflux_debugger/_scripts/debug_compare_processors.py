#!/usr/bin/env python3
"""Compare MLX and HF image processors side-by-side."""

import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# Add mflux to path
mflux_src = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mflux_src))

import numpy as np
from PIL import Image
from transformers import Qwen2VLImageProcessor

from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor
from mflux_debugger.semantic_checkpoint import debug_checkpoint

# Use the dog image from the test
image_path = "dog.jpg"
if not os.path.exists(image_path):
    image_path = "tests/resources/reference_upscaled.png"

image = Image.open(image_path).convert("RGB")

print("=" * 80)
print("Comparing MLX vs HF Image Processors")
print("=" * 80)
print(f"Input image: {image_path}")
print(f"Original size: {image.size}")

# Initialize processors
mlx_processor = QwenImageProcessor()
hf_processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen-Image-Edit-2509")

print("\n" + "=" * 80)
print("MLX Processor")
print("=" * 80)

# Process with MLX
mlx_pixel_values, mlx_grid_thw = mlx_processor.preprocess([image])

print(f"MLX pixel_values shape: {mlx_pixel_values.shape}")
print(f"MLX image_grid_thw: {mlx_grid_thw}")
print(f"MLX pixel_values dtype: {mlx_pixel_values.dtype}")
print(f"MLX pixel_values stats: mean={mlx_pixel_values.mean():.6f}, std={mlx_pixel_values.std():.6f}")
print(f"MLX pixel_values range: [{mlx_pixel_values.min():.6f}, {mlx_pixel_values.max():.6f}]")
print(f"MLX pixel_values first 10: {mlx_pixel_values[0, :10]}")

print("\n" + "=" * 80)
print("HF Processor")
print("=" * 80)

# Process with HF
hf_result = hf_processor.preprocess([image], return_tensors="np")
hf_pixel_values = hf_result["pixel_values"]
hf_grid_thw = hf_result["image_grid_thw"]

print(f"HF pixel_values shape: {hf_pixel_values.shape}")
print(f"HF image_grid_thw: {hf_grid_thw}")
print(f"HF pixel_values dtype: {hf_pixel_values.dtype}")
print(f"HF pixel_values stats: mean={hf_pixel_values.mean():.6f}, std={hf_pixel_values.std():.6f}")
print(f"HF pixel_values range: [{hf_pixel_values.min():.6f}, {hf_pixel_values.max():.6f}]")
print(f"HF pixel_values first 10: {hf_pixel_values[0, :10]}")

print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)

# Compare shapes
print(f"Shapes match: {mlx_pixel_values.shape == hf_pixel_values.shape}")
print(f"Grid THW match: {np.array_equal(mlx_grid_thw, hf_grid_thw)}")

# Compare values
if mlx_pixel_values.shape == hf_pixel_values.shape:
    diff = np.abs(mlx_pixel_values - hf_pixel_values)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Values close (atol=1e-5): {np.allclose(mlx_pixel_values, hf_pixel_values, atol=1e-5)}")
    print(f"Values close (atol=1e-3): {np.allclose(mlx_pixel_values, hf_pixel_values, atol=1e-3)}")
    
    # Find where differences are largest
    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"Largest difference at index {max_diff_idx}:")
    print(f"  MLX: {mlx_pixel_values[max_diff_idx]:.6f}")
    print(f"  HF:  {hf_pixel_values[max_diff_idx]:.6f}")
    print(f"  Diff: {diff[max_diff_idx]:.6f}")
else:
    print("⚠️  Shapes don't match - cannot compare values directly")
    print(f"  MLX shape: {mlx_pixel_values.shape}")
    print(f"  HF shape:  {hf_pixel_values.shape}")

print("\n" + "=" * 80)

