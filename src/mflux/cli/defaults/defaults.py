import os
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = ["dev", "schnell", "krea-dev", "dev-krea", "qwen", "fibo", "z-image-turbo"]
MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "fibo": 20,
    "z-image-turbo": 9,
}
QUANTIZE_CHOICES = [2, 3, 4, 5, 6, 8]  # INT2 added for extreme compression

# Quantization mode presets: map user-friendly names to bit depths
# These provide semantic options that users can choose based on their priorities
QUANTIZE_MODES = {
    "speed": 4,  # INT4: Fastest inference, smaller memory footprint
    "quality": 8,  # INT8: Best quality, larger memory footprint
    "balanced": 4,  # INT4: Good balance (same as speed)
    "mixed": 4,  # INT4 for most, could be expanded to INT8 attention + INT4 FFN
    "extreme": 2,  # INT2: Maximum compression, experimental
}

# All valid quantization values (int bits + string modes)
QUANTIZE_MODE_CHOICES = list(QUANTIZE_MODES.keys())

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
