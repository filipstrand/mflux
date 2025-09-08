import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ReferenceVsOutputImageError(AssertionError): ...


# How we determined DEFAULT_MISMATCH_THRESHOLD value: by eye test
# successive mlx/mflux versions have generated images that are visually close enough
# to consider as a valid upgrade. Minor visual differences are attributable to mlx updates
DEFAULT_MISMATCH_THRESHOLD = 0.15
USER_MISMATCH_THRESHOLD = float(os.environ.get("MFLUX_IMAGE_MISMATCH_THRESHOLD", DEFAULT_MISMATCH_THRESHOLD))


def check_images_close_enough(image1_path: str | Path, image2_path: str | Path, error_message_prefix: str):
    image1_path = Path(image1_path)
    image2_path = Path(image2_path)
    image1_data = np.array(Image.open(image1_path))
    image2_data = np.array(Image.open(image2_path))
    closeness_array = np.isclose(
        image1_data,
        image2_data,
        rtol=float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_RTOL", 0.1)),
        atol=0,  # ignore absolute tolerance
    )
    num_mismatched = np.count_nonzero(~closeness_array)
    total_elements = image1_data.size
    mismatch_ratio = num_mismatched / total_elements
    if mismatch_ratio > USER_MISMATCH_THRESHOLD:
        diff = image1_data - image2_data
        # Scale the difference so it's visible. A difference of 50 will become bright.
        # We'll scale it so that a difference of 50 or more becomes pure white (255).
        diff_visual = (diff / 50 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(image2_path.with_stem(image1_path.stem + "_diff"), diff_visual)
        raise AssertionError(
            f"{error_message_prefix} Check {image1_path} vs {image2_path} :: their elements are {mismatch_ratio:.1%} different. Fails assertion for {USER_MISMATCH_THRESHOLD=}"
        )
