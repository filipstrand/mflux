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
MAX_PIXELS_WARNING_THRESHOLD = 2048 * 2048
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = ["dev", "schnell", "krea-dev", "dev-krea", "qwen", "fibo", "z-image-turbo"]
MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "fibo": 20,
    "z-image-turbo": 9,
}
QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

# Determine cache directory
if os.environ.get("MFLUX_CACHE_DIR"):
    # User specified cache directory (e.g. external storage)
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    # macOS-idiomatic cache directory @ /Users/username/Library/Caches/mflux
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
