"""
Image archiving utilities for the debugger.

Archives images when starting debug sessions to keep latest/ clean.
"""

import logging
import shutil
from datetime import datetime
from typing import Optional

from mflux_debugger.image_tensor_paths import (
    get_images_archive_dir,
    get_images_latest_dir,
)

logger = logging.getLogger(__name__)


def archive_images(framework: Optional[str] = None) -> int:
    """
    Archive existing images from latest/ to archive/.

    When framework is provided, only archives images from that framework's directory.
    When framework is None, archives all images from latest/.

    Args:
        framework: Optional framework name ("mlx" or "pytorch") to archive specific directory

    Returns:
        Number of files archived
    """
    if framework:
        # Archive specific framework directory
        latest_dir = get_images_latest_dir() / framework
        if not latest_dir.exists():
            logger.debug(f"No images to archive for {framework}")
            return 0

        image_files = list(latest_dir.glob("*.png")) + list(latest_dir.glob("*.jpg")) + list(latest_dir.glob("*.jpeg"))

        if not image_files:
            logger.debug(f"No images to archive for {framework}")
            return 0

        # Create timestamped archive directory for this framework
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = get_images_archive_dir() / f"{framework}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move files to archive
        archived = 0
        total_size = 0
        for image_file in image_files:
            try:
                total_size += image_file.stat().st_size
                dest = archive_dir / image_file.name
                shutil.move(str(image_file), str(dest))
                archived += 1
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning(f"Failed to archive {image_file.name}: {e}")

        if archived > 0:
            size_mb = total_size / (1024 * 1024)
            logger.info(f"📦 Archived {archived} image(s) ({size_mb:.2f} MB) from {framework}/ to archive")
            print(f"📦 Archived {archived} image(s) ({size_mb:.2f} MB) from {framework}/")

        return archived
    else:
        # Archive all images from latest/ (both mlx and pytorch)
        latest_dir = get_images_latest_dir()
        total_archived = 0

        for framework_subdir in ["mlx", "pytorch"]:
            framework_dir = latest_dir / framework_subdir
            if framework_dir.exists():
                archived = archive_images(framework_subdir)
                total_archived += archived

        return total_archived
