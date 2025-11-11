#!/usr/bin/env python3
"""Test HF processor directly to generate checkpoint JSON files."""

import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# Add mflux to path for debug_checkpoint
mflux_src = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mflux_src))

from PIL import Image
from transformers import Qwen2VLImageProcessor

from mflux_debugger._scripts.debug_edit_config import EDIT_DEBUG_CONFIG

# Load image
image = Image.open(EDIT_DEBUG_CONFIG.image_path).convert("RGB")

# Initialize HF processor (this will use our modified version with checkpoints)
processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen-Image-Edit-2509")

# Process image - this will trigger checkpoints in the modified transformers code
result = processor.preprocess([image], return_tensors="np")

print(f"Processed image with HF processor")
print(f"pixel_values shape: {result['pixel_values'].shape}")
print(f"image_grid_thw: {result['image_grid_thw']}")

