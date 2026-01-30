import random
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image

from mflux.models.z_image.variants.training.dataset.batch import Example


class ZImagePreProcessing:
    """Image augmentation for Z-Image training."""

    @staticmethod
    def augment(example: Example, num_augmentations: int = 4) -> list[Example]:
        """Create augmented versions of an example.

        Augmentations include:
        - Original (no augmentation)
        - Horizontal flip
        - Random crops (preserving aspect ratio)
        """
        augmented = [example]

        # Horizontal flip
        if num_augmentations >= 2:
            flipped = ZImagePreProcessing._flip_horizontal(example, augmentation_id=1)
            augmented.append(flipped)

        # Additional augmentations can be added here for larger datasets
        # For now, keeping it simple with just flip

        return augmented

    @staticmethod
    def _flip_horizontal(example: Example, augmentation_id: int) -> Example:
        """Create a horizontally flipped version.

        Note: The actual flip happens during encoding in the dataset preparation.
        This just marks the example as flipped for tracking.
        """
        return replace(
            example,
            augmentation_id=augmentation_id,
            is_flipped=True,
        )

    @staticmethod
    def repeat_examples(
        examples: list[Example],
        repeat_count: int,
    ) -> list[Example]:
        """Repeat each example N times for small datasets.

        This is useful when training with a small number of images
        to increase effective dataset size and epoch length.

        Args:
            examples: List of examples to repeat
            repeat_count: How many times to repeat each example

        Returns:
            List with each example repeated repeat_count times,
            each with a unique augmentation_id for tracking.
        """
        if repeat_count <= 1:
            return examples

        repeated = []
        for i, example in enumerate(examples):
            repeated.extend(replace(example, augmentation_id=i * repeat_count + r) for r in range(repeat_count))
        return repeated

    @staticmethod
    def random_crop_params(
        original_width: int,
        original_height: int,
        target_width: int,
        target_height: int,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Calculate random crop parameters.

        Returns:
            (crop_top, crop_left) offsets for cropping
        """
        if rng is None:
            rng = random.Random()

        max_top = max(0, original_height - target_height)
        max_left = max(0, original_width - target_width)

        crop_top = rng.randint(0, max_top) if max_top > 0 else 0
        crop_left = rng.randint(0, max_left) if max_left > 0 else 0

        return crop_top, crop_left

    @staticmethod
    def apply_random_crop(
        image: "PIL.Image.Image",
        crop_top: int,
        crop_left: int,
        target_width: int,
        target_height: int,
    ) -> "PIL.Image.Image":
        """Apply random crop to an image.

        Args:
            image: PIL Image to crop
            crop_top: Top offset for crop
            crop_left: Left offset for crop
            target_width: Width of cropped region
            target_height: Height of cropped region

        Returns:
            Cropped PIL Image
        """
        return image.crop(
            (
                crop_left,
                crop_top,
                crop_left + target_width,
                crop_top + target_height,
            )
        )
