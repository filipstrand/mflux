"""Tests for Z-Image data augmentation functionality.

Data augmentation helps prevent overfitting and improves generalization,
especially important for small training datasets.
"""

import random

import pytest
from PIL import Image

from mflux.models.z_image.variants.training.dataset.preprocessing import (
    AugmentationConfig,
    ZImagePreProcessing,
)
from mflux.models.z_image.variants.training.state.training_spec import (
    AugmentationSpec,
    DatasetSpec,
)


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentationConfig()

        assert config.enable_flip is True
        assert config.enable_brightness is False
        assert config.enable_contrast is False
        assert config.enable_color_jitter is False
        assert config.enable_rotation is False
        assert config.brightness_range == (0.8, 1.2)
        assert config.contrast_range == (0.8, 1.2)
        assert config.color_jitter_strength == 0.1
        assert config.rotation_degrees == 5.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AugmentationConfig(
            enable_flip=False,
            enable_brightness=True,
            brightness_range=(0.7, 1.3),
            enable_contrast=True,
            contrast_range=(0.9, 1.1),
            enable_color_jitter=True,
            color_jitter_strength=0.2,
            enable_rotation=True,
            rotation_degrees=10.0,
        )

        assert config.enable_flip is False
        assert config.enable_brightness is True
        assert config.brightness_range == (0.7, 1.3)
        assert config.enable_contrast is True
        assert config.contrast_range == (0.9, 1.1)
        assert config.enable_color_jitter is True
        assert config.color_jitter_strength == 0.2
        assert config.enable_rotation is True
        assert config.rotation_degrees == 10.0

    def test_invalid_brightness_range_negative(self):
        """Test that negative brightness range raises error."""
        with pytest.raises(ValueError, match="brightness_range min must be positive"):
            AugmentationConfig(brightness_range=(0.0, 1.2))

    def test_invalid_brightness_range_inverted(self):
        """Test that inverted brightness range raises error."""
        with pytest.raises(ValueError, match="brightness_range min must be <= max"):
            AugmentationConfig(brightness_range=(1.5, 0.8))

    def test_invalid_contrast_range_negative(self):
        """Test that negative contrast range raises error."""
        with pytest.raises(ValueError, match="contrast_range min must be positive"):
            AugmentationConfig(contrast_range=(-0.5, 1.0))

    def test_invalid_color_jitter_strength(self):
        """Test that invalid color jitter strength raises error."""
        with pytest.raises(ValueError, match="color_jitter_strength must be in"):
            AugmentationConfig(color_jitter_strength=1.5)

    def test_invalid_rotation_degrees(self):
        """Test that negative rotation degrees raises error."""
        with pytest.raises(ValueError, match="rotation_degrees must be non-negative"):
            AugmentationConfig(rotation_degrees=-5.0)


class TestAugmentationSpec:
    """Tests for AugmentationSpec in training_spec."""

    def test_default_spec(self):
        """Test default AugmentationSpec values."""
        spec = AugmentationSpec()

        assert spec.enable_flip is True
        assert spec.enable_brightness is False
        assert spec.enable_contrast is False

    def test_dataset_spec_with_augmentation(self):
        """Test DatasetSpec with nested AugmentationSpec."""
        aug_spec = AugmentationSpec(
            enable_brightness=True,
            enable_contrast=True,
        )
        dataset_spec = DatasetSpec(
            repeat_count=2,
            enable_augmentation=True,
            augmentation=aug_spec,
        )

        assert dataset_spec.repeat_count == 2
        assert dataset_spec.augmentation is not None
        assert dataset_spec.augmentation.enable_brightness is True

    def test_dataset_spec_without_augmentation(self):
        """Test DatasetSpec without augmentation config."""
        dataset_spec = DatasetSpec(enable_augmentation=True)

        assert dataset_spec.augmentation is None


class TestAugmentationMethods:
    """Tests for augmentation application methods."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (256, 256), color=(128, 128, 128))

    @pytest.fixture
    def rng(self):
        """Create deterministic RNG for testing."""
        return random.Random(42)

    def test_apply_brightness(self, test_image):
        """Test brightness adjustment."""
        # Brighten
        bright = ZImagePreProcessing.apply_brightness(test_image, 1.5)
        assert bright.size == test_image.size

        # Darken
        dark = ZImagePreProcessing.apply_brightness(test_image, 0.5)
        assert dark.size == test_image.size

    def test_apply_contrast(self, test_image):
        """Test contrast adjustment."""
        # Increase contrast
        high = ZImagePreProcessing.apply_contrast(test_image, 1.5)
        assert high.size == test_image.size

        # Decrease contrast
        low = ZImagePreProcessing.apply_contrast(test_image, 0.5)
        assert low.size == test_image.size

    def test_apply_color_jitter(self, test_image, rng):
        """Test color jitter."""
        jittered = ZImagePreProcessing.apply_color_jitter(test_image, strength=0.2, rng=rng)
        assert jittered.size == test_image.size

    def test_apply_augmentations_flip_only(self, test_image, rng):
        """Test augmentation with flip only."""
        config = AugmentationConfig(enable_flip=True)
        result = ZImagePreProcessing.apply_augmentations(test_image, config, rng=rng)
        assert result.size == test_image.size

    def test_apply_augmentations_all_enabled(self, test_image, rng):
        """Test augmentation with all options enabled."""
        config = AugmentationConfig(
            enable_flip=True,
            enable_brightness=True,
            brightness_range=(0.9, 1.1),
            enable_contrast=True,
            contrast_range=(0.9, 1.1),
            enable_color_jitter=True,
            color_jitter_strength=0.1,
            enable_rotation=True,
            rotation_degrees=5.0,
        )
        result = ZImagePreProcessing.apply_augmentations(test_image, config, rng=rng)
        assert result.size == test_image.size

    def test_apply_augmentations_deterministic(self, test_image):
        """Test that augmentations are deterministic with same seed."""
        config = AugmentationConfig(
            enable_flip=True,
            enable_brightness=True,
            enable_contrast=True,
        )

        rng1 = random.Random(42)
        result1 = ZImagePreProcessing.apply_augmentations(test_image, config, rng=rng1)

        rng2 = random.Random(42)
        result2 = ZImagePreProcessing.apply_augmentations(test_image, config, rng=rng2)

        # Results should be identical with same seed
        assert list(result1.getdata()) == list(result2.getdata())

    def test_apply_augmentations_disabled(self, test_image, rng):
        """Test that disabled augmentations don't modify image."""
        config = AugmentationConfig(
            enable_flip=False,
            enable_brightness=False,
            enable_contrast=False,
            enable_color_jitter=False,
            enable_rotation=False,
        )
        result = ZImagePreProcessing.apply_augmentations(test_image, config, rng=rng)

        # With all augmentations disabled, image should be unchanged
        assert list(result.getdata()) == list(test_image.getdata())


class TestAugmentationIntegration:
    """Integration tests for augmentation in dataset pipeline."""

    def test_augmentation_config_from_spec(self):
        """Test creating AugmentationConfig from AugmentationSpec."""
        spec = AugmentationSpec(
            enable_brightness=True,
            brightness_range=(0.85, 1.15),
        )

        # Convert spec to config
        config = AugmentationConfig(
            enable_flip=spec.enable_flip,
            enable_brightness=spec.enable_brightness,
            brightness_range=spec.brightness_range,
            enable_contrast=spec.enable_contrast,
            contrast_range=spec.contrast_range,
            enable_color_jitter=spec.enable_color_jitter,
            color_jitter_strength=spec.color_jitter_strength,
            enable_rotation=spec.enable_rotation,
            rotation_degrees=spec.rotation_degrees,
        )

        assert config.enable_brightness is True
        assert config.brightness_range == (0.85, 1.15)
