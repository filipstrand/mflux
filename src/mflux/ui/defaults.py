import logging
import os
import shutil
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

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
MODEL_CHOICES = ["dev", "schnell", "dev-kontext", "dev-fill"]
MODEL_INFERENCE_STEPS = {
    "dev": 14,
    "dev-fill": 14,
    "dev-depth": 14,
    "dev-redux": 14,
    "dev-kontext": 14,
    "schnell": 4,
}
QUANTIZE_CHOICES = [3, 4, 6, 8]


def _migrate_legacy_cache(new_cache_dir: Path) -> None:
    """Migrate legacy ~/.cache/mflux to new location if needed."""
    legacy_cache = Path.home() / ".cache" / "mflux"

    # Skip if legacy path doesn't exist or is already a symlink
    if not legacy_cache.exists() or legacy_cache.is_symlink():
        return

    # Skip if we're already using the legacy path
    if new_cache_dir == legacy_cache:
        return

    try:
        logger.warning(f"Migrating cache from {legacy_cache} to {new_cache_dir}")

        # Create new directory
        new_cache_dir.mkdir(parents=True, exist_ok=True)

        # Move all contents from old to new location
        for item in legacy_cache.iterdir():
            src = legacy_cache / item.name
            dst = new_cache_dir / item.name

            if dst.exists():
                logger.warning(f"  Skipping {item.name} (already exists in destination)")
                continue

            logger.warning(f"  Moving {item.name}")
            shutil.move(str(src), str(dst))

        # Remove the now-empty old directory
        legacy_cache.rmdir()

        # Create symlink from old location to new location for backward compatibility
        legacy_cache.parent.mkdir(parents=True, exist_ok=True)
        legacy_cache.symlink_to(new_cache_dir)
        logger.info(f"Created symlink: {legacy_cache} -> {new_cache_dir}")

    except (OSError, IOError, shutil.Error) as e:
        logger.warning(f"Cache migration failed: {e}")
        logger.info("Continuing with existing location")


# Determine cache directory
if os.environ.get("MFLUX_CACHE_DIR"):
    # User specified cache directory (e.g. external storage)
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    # macOS-idiomatic cache directory @ /Users/username/Library/Caches/mflux
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

# Perform one-time migration if needed
_migrate_legacy_cache(MFLUX_CACHE_DIR)

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
