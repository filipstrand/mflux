"""
Compatibility layer for image comparison.

This module maintains backward compatibility for tests that import from
tests.image_generation.helpers.image_compare. New code should import
directly from mflux.utils.image_compare.
"""

from mflux.utils.image_compare import (
    DEFAULT_MISMATCH_THRESHOLD,
    ENV_MISMATCH_THRESHOLD,
    ReferenceVsOutputImageError,
    check_images_close_enough,
    compare_images,
)

__all__ = [
    "ReferenceVsOutputImageError",
    "DEFAULT_MISMATCH_THRESHOLD",
    "ENV_MISMATCH_THRESHOLD",
    "check_images_close_enough",
    "compare_images",
]
