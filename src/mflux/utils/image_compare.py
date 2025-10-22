"""
Image Comparison Utility.

Provides utilities for comparing images between PyTorch and MLX implementations,
useful for verifying that implementations produce similar results.

The default mismatch threshold of 15% was determined by eye test - successive
mlx/mflux versions have generated images that are visually close enough to consider
as a valid upgrade. Minor visual differences are attributable to mlx updates.
"""

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
ENV_MISMATCH_THRESHOLD = float(os.environ.get("MFLUX_IMAGE_MISMATCH_THRESHOLD", DEFAULT_MISMATCH_THRESHOLD))


def check_images_close_enough(
    image1_path: str | Path,
    image2_path: str | Path,
    error_message_prefix: str,
    mismatch_threshold: float = ENV_MISMATCH_THRESHOLD,
) -> float:
    """
    Check if two images are close enough within a threshold.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        error_message_prefix: Prefix for error message if comparison fails
        mismatch_threshold: Maximum allowed mismatch ratio (default: 0.15 = 15%)

    Returns:
        The mismatch ratio (0.0 to 1.0)

    Raises:
        AssertionError: If mismatch exceeds threshold
    """
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
    if mismatch_ratio > mismatch_threshold:
        diff = image1_data - image2_data
        # Scale the difference so it's visible. A difference of 50 will become bright.
        # We'll scale it so that a difference of 50 or more becomes pure white (255).
        diff_visual = (np.abs(diff) / 50 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(image2_path.with_stem(image1_path.stem + "_diff"), diff_visual)
        raise AssertionError(
            f"{error_message_prefix} Check {image1_path} vs {image2_path} :: their elements are {mismatch_ratio:.1%} different. Fails assertion for {mismatch_threshold=}"
        )
    return mismatch_ratio


def compare_images(
    image1_path: str | Path,
    image2_path: str | Path,
    mismatch_threshold: float | None = None,
    create_diff: bool = True,
) -> dict:
    """
    Compare two images and return detailed comparison results.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        mismatch_threshold: Optional threshold to check against (if None, just returns stats)
        create_diff: Whether to create a diff visualization image

    Returns:
        Dictionary with comparison results:
        - mismatch_ratio: Ratio of mismatched pixels (0.0 to 1.0)
        - total_pixels: Total number of pixels
        - mismatched_pixels: Number of mismatched pixels
        - image1_shape: Shape of first image
        - image2_shape: Shape of second image
        - diff_path: Path to diff image if created, None otherwise
        - passes_threshold: True if mismatch_ratio <= threshold (if threshold provided)
    """
    image1_path = Path(image1_path)
    image2_path = Path(image2_path)
    image1_data = np.array(Image.open(image1_path))
    image2_data = np.array(Image.open(image2_path))

    rtol = float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_RTOL", 0.1))
    closeness_array = np.isclose(
        image1_data,
        image2_data,
        rtol=rtol,
        atol=0,
    )

    num_mismatched = np.count_nonzero(~closeness_array)
    total_elements = image1_data.size
    mismatch_ratio = num_mismatched / total_elements

    diff_path = None
    if create_diff:
        diff = image1_data - image2_data
        diff_visual = (np.abs(diff) / 50 * 255).clip(0, 255).astype(np.uint8)
        diff_path = image2_path.with_stem(image1_path.stem + "_diff")
        cv2.imwrite(str(diff_path), diff_visual)

    result = {
        "mismatch_ratio": mismatch_ratio,
        "total_pixels": total_elements,
        "mismatched_pixels": num_mismatched,
        "image1_shape": image1_data.shape,
        "image2_shape": image2_data.shape,
        "diff_path": str(diff_path) if diff_path else None,
    }

    if mismatch_threshold is not None:
        result["passes_threshold"] = mismatch_ratio <= mismatch_threshold
        result["threshold"] = mismatch_threshold

    return result
