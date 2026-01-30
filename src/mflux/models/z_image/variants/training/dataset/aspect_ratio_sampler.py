"""Aspect ratio bucketing for Z-Image training.

Groups images by aspect ratio for more efficient training on
varied datasets with different image proportions.
"""

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.dataset.batch import Example


# Default aspect ratio (1.0 = square) used when image dimensions unavailable
DEFAULT_ASPECT_RATIO = 1.0

# Standard aspect ratio buckets for Z-Image training
# Key is aspect ratio string, value is (width, height) tuple
# Expanded from 7 to 11 buckets for finer granularity (2-3% efficiency gain)
Z_IMAGE_BUCKETS = {
    "0.50": (352, 704),  # Portrait 1:2 (extreme tall)
    "0.56": (368, 656),  # Portrait ~9:16 (social media portrait)
    "0.67": (384, 576),  # Portrait 2:3 (standard portrait)
    "0.75": (384, 512),  # Portrait 3:4 (traditional photo portrait)
    "0.89": (448, 504),  # Near-square portrait (soft portrait)
    "1.00": (512, 512),  # Square (Instagram classic)
    "1.12": (504, 448),  # Near-square landscape (soft landscape)
    "1.33": (512, 384),  # Landscape 4:3 (traditional photo landscape)
    "1.50": (576, 384),  # Landscape 3:2 (classic 35mm)
    "1.78": (656, 368),  # Landscape ~16:9 (widescreen/video)
    "2.00": (704, 352),  # Landscape 2:1 (cinematic/panoramic)
}

# Legacy 7-bucket configuration for backward compatibility
Z_IMAGE_BUCKETS_LEGACY = {
    "0.50": (352, 704),  # Portrait 1:2
    "0.67": (384, 576),  # Portrait 2:3
    "0.75": (384, 512),  # Portrait 3:4
    "1.00": (512, 512),  # Square
    "1.33": (512, 384),  # Landscape 4:3
    "1.50": (576, 384),  # Landscape 3:2
    "2.00": (704, 352),  # Landscape 2:1
}


@dataclass
class BucketAssignment:
    """An example's bucket assignment."""

    example_index: int
    bucket_key: str
    target_width: int
    target_height: int


class AspectRatioSampler:
    """Samples batches from aspect ratio buckets.

    Groups examples by their closest aspect ratio bucket,
    then samples batches from a single bucket at a time to
    ensure consistent batch dimensions.
    """

    def __init__(
        self,
        examples: list["Example"],
        buckets: dict[str, tuple[int, int]] | None = None,
        seed: int | None = None,
    ):
        """Initialize sampler.

        Args:
            examples: List of examples to bucket
            buckets: Optional custom bucket definitions.
                     Defaults to Z_IMAGE_BUCKETS.
            seed: Random seed for reproducibility. If None, uses OS entropy.
        """
        if not examples:
            raise ValueError("AspectRatioSampler requires at least one example")

        # Validate seed type
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")

        self.examples = examples
        self.buckets = buckets or Z_IMAGE_BUCKETS
        # Use OS entropy if no seed provided (not cryptographic, just standard randomness)
        self.rng = random.Random(seed)

        # Assign each example to a bucket
        self.bucket_assignments = self._assign_to_buckets()

        # Group assignments by bucket
        self.bucket_indices: dict[str, list[int]] = {}
        for i, assignment in enumerate(self.bucket_assignments):
            if assignment.bucket_key not in self.bucket_indices:
                self.bucket_indices[assignment.bucket_key] = []
            self.bucket_indices[assignment.bucket_key].append(i)

        # Track which buckets have unsampled examples
        self._reset_epoch()

    def _assign_to_buckets(self) -> list[BucketAssignment]:
        """Assign each example to its nearest aspect ratio bucket."""
        assignments = []

        # Parse bucket aspect ratios for comparison
        bucket_ratios = [(key, float(key)) for key in self.buckets.keys()]

        for i, example in enumerate(self.examples):
            # Get example aspect ratio from encoded image shape
            # Latent shape is typically (1, H/8, W/8, C) or similar
            # We need the original image dimensions from the path
            # For now, assume square if not available
            aspect_ratio = self._get_aspect_ratio(example)

            # Find nearest bucket
            nearest_key = min(bucket_ratios, key=lambda x: abs(x[1] - aspect_ratio))[0]
            target_width, target_height = self.buckets[nearest_key]

            assignments.append(
                BucketAssignment(
                    example_index=i,
                    bucket_key=nearest_key,
                    target_width=target_width,
                    target_height=target_height,
                )
            )

        return assignments

    def _get_aspect_ratio(self, example: "Example") -> float:
        """Get aspect ratio for an example.

        Attempts to get the original image dimensions from the latent shape.
        Supports multiple latent formats:
        - [C, H, W] - 3D packed latents after pack_latents
        - [C, F, H, W] - 4D format before packing

        Falls back to 1.0 (square) with warning if shape is unexpected.

        Returns:
            Aspect ratio (width/height), always > 0. Defaults to 1.0.
        """
        # Try to get from latent shape (if available)
        if hasattr(example, "encoded_image") and example.encoded_image is not None:
            shape = example.encoded_image.shape

            # Z-Image latents: [C, H/8, W/8] after pack_latents
            if len(shape) == 3:
                h, w = shape[1], shape[2]
                if h > 0 and w > 0:
                    return float(w) / float(h)
                # Log warning for malformed latent shapes with actionable guidance
                logger.warning(
                    f"Example {example.example_id} has invalid latent dimensions (h={h}, w={w}). "
                    f"Using square aspect ratio ({DEFAULT_ASPECT_RATIO}). This is usually caused by corrupted image encoding. "
                    f"Training will continue but batch bucketing efficiency may be affected."
                )
                return DEFAULT_ASPECT_RATIO

            # Also handle [C, F, H, W] format before packing
            if len(shape) == 4:
                h, w = shape[2], shape[3]
                if h > 0 and w > 0:
                    return float(w) / float(h)
                logger.warning(
                    f"Example {example.example_id} has invalid 4D latent dimensions (h={h}, w={w}). "
                    f"Using square aspect ratio ({DEFAULT_ASPECT_RATIO})."
                )
                return DEFAULT_ASPECT_RATIO

            # Unexpected shape - warn with specific shape info
            logger.warning(
                f"Example {example.example_id} has unexpected latent shape {shape} "
                f"(expected 3D [C,H,W] or 4D [C,F,H,W]). "
                f"Using square aspect ratio ({DEFAULT_ASPECT_RATIO}). Training will continue normally."
            )

        return DEFAULT_ASPECT_RATIO

    def _reset_epoch(self) -> None:
        """Reset for a new epoch."""
        self.remaining_in_bucket = {key: list(indices) for key, indices in self.bucket_indices.items()}
        # Shuffle each bucket
        for indices in self.remaining_in_bucket.values():
            self.rng.shuffle(indices)

    def get_batch(
        self,
        batch_size: int,
    ) -> tuple[list[int], tuple[int, int]] | None:
        """Sample a batch from a bucket.

        Returns:
            Tuple of (example_indices, (target_width, target_height)),
            or None if no examples remain in any bucket.
        """
        # Find non-empty buckets
        non_empty = [key for key, indices in self.remaining_in_bucket.items() if len(indices) > 0]

        if not non_empty:
            return None

        # Sample bucket proportionally to remaining examples
        weights = [len(self.remaining_in_bucket[key]) for key in non_empty]
        total = sum(weights)
        r = self.rng.random() * total

        cumsum = 0
        selected_bucket = non_empty[0]
        for key, weight in zip(non_empty, weights):
            cumsum += weight
            if r <= cumsum:
                selected_bucket = key
                break

        # Get batch from selected bucket
        indices = self.remaining_in_bucket[selected_bucket]
        actual_batch_size = min(batch_size, len(indices))

        batch_indices = indices[:actual_batch_size]
        self.remaining_in_bucket[selected_bucket] = indices[actual_batch_size:]

        # Get target resolution for this bucket
        target_width, target_height = self.buckets[selected_bucket]

        # Convert assignment indices to example indices
        example_indices = [self.bucket_assignments[i].example_index for i in batch_indices]

        return example_indices, (target_width, target_height)

    def has_examples(self) -> bool:
        """Check if any examples remain to be sampled."""
        return any(len(indices) > 0 for indices in self.remaining_in_bucket.values())

    def get_bucket_stats(self) -> dict[str, int]:
        """Get the number of examples in each bucket."""
        return {key: len(indices) for key, indices in self.bucket_indices.items()}

    def reset(self) -> None:
        """Reset for a new epoch."""
        self._reset_epoch()
