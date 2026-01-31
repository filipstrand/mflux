import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from PIL import ImageEnhance

if TYPE_CHECKING:
    import PIL.Image

from mflux.models.z_image.variants.training.dataset.batch import Example


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation during training.

    All augmentations are applied with random variation within specified ranges.
    Color augmentations are applied in PIL/Pillow before conversion to tensors.

    Attributes:
        enable_flip: Enable horizontal flip augmentation (default: True)
        enable_brightness: Enable random brightness adjustment
        brightness_range: Min/max brightness multiplier (0.8=darker, 1.2=brighter)
        enable_contrast: Enable random contrast adjustment
        contrast_range: Min/max contrast multiplier
        enable_color_jitter: Enable random color/saturation jitter
        color_jitter_strength: Strength of color variation (0.0-1.0)
        enable_rotation: Enable small random rotation
        rotation_degrees: Maximum rotation in degrees (+/-)
    """

    enable_flip: bool = True
    enable_brightness: bool = False
    brightness_range: tuple[float, float] = (0.8, 1.2)
    enable_contrast: bool = False
    contrast_range: tuple[float, float] = (0.8, 1.2)
    enable_color_jitter: bool = False
    color_jitter_strength: float = 0.1
    enable_rotation: bool = False
    rotation_degrees: float = 5.0

    def __post_init__(self) -> None:
        """Validate augmentation configuration."""
        if self.brightness_range[0] <= 0:
            raise ValueError("brightness_range min must be positive")
        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("brightness_range min must be <= max")
        if self.contrast_range[0] <= 0:
            raise ValueError("contrast_range min must be positive")
        if self.contrast_range[0] > self.contrast_range[1]:
            raise ValueError("contrast_range min must be <= max")
        if not 0.0 <= self.color_jitter_strength <= 1.0:
            raise ValueError("color_jitter_strength must be in [0, 1]")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        if self.rotation_degrees > 45:
            raise ValueError("rotation_degrees must be <= 45 (extreme rotations degrade training)")
        # Upper bounds for brightness/contrast to prevent resource exhaustion
        if self.brightness_range[1] > 5.0:
            raise ValueError("brightness_range max must be <= 5.0")
        if self.contrast_range[1] > 5.0:
            raise ValueError("contrast_range max must be <= 5.0")


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

    @staticmethod
    def apply_augmentations(
        image: "PIL.Image.Image",
        config: AugmentationConfig,
        rng: random.Random | None = None,
    ) -> "PIL.Image.Image":
        """Apply configured augmentations to an image.

        Augmentations are applied in order:
        1. Horizontal flip (if enabled)
        2. Brightness adjustment
        3. Contrast adjustment
        4. Color jitter
        5. Rotation

        Args:
            image: PIL Image to augment
            config: Augmentation configuration
            rng: Random number generator for reproducibility

        Returns:
            Augmented PIL Image
        """
        if rng is None:
            rng = random.Random()

        result = image

        # 1. Horizontal flip
        if config.enable_flip and rng.random() < 0.5:
            result = result.transpose(method=0)  # FLIP_LEFT_RIGHT = 0

        # 2. Brightness adjustment
        if config.enable_brightness:
            factor = rng.uniform(config.brightness_range[0], config.brightness_range[1])
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(factor)

        # 3. Contrast adjustment
        if config.enable_contrast:
            factor = rng.uniform(config.contrast_range[0], config.contrast_range[1])
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(factor)

        # 4. Color/saturation jitter
        if config.enable_color_jitter:
            # Jitter saturation
            sat_factor = 1.0 + rng.uniform(-config.color_jitter_strength, config.color_jitter_strength)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(sat_factor)

        # 5. Small rotation (preserving aspect ratio)
        if config.enable_rotation and config.rotation_degrees > 0:
            angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
            # Use bilinear interpolation and expand=False to preserve size
            result = result.rotate(angle, resample=2, expand=False, fillcolor=(128, 128, 128))

        return result

    @staticmethod
    def apply_brightness(
        image: "PIL.Image.Image",
        factor: float,
    ) -> "PIL.Image.Image":
        """Apply brightness adjustment.

        Args:
            image: PIL Image
            factor: Brightness factor (1.0 = no change, <1 darker, >1 brighter)

        Returns:
            Adjusted PIL Image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_contrast(
        image: "PIL.Image.Image",
        factor: float,
    ) -> "PIL.Image.Image":
        """Apply contrast adjustment.

        Args:
            image: PIL Image
            factor: Contrast factor (1.0 = no change)

        Returns:
            Adjusted PIL Image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_color_jitter(
        image: "PIL.Image.Image",
        strength: float,
        rng: random.Random | None = None,
    ) -> "PIL.Image.Image":
        """Apply random color/saturation jitter.

        Args:
            image: PIL Image
            strength: Jitter strength (0.0-1.0)
            rng: Random number generator

        Returns:
            Jittered PIL Image
        """
        if rng is None:
            rng = random.Random()

        factor = 1.0 + rng.uniform(-strength, strength)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
