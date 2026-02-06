from pathlib import Path
from unittest.mock import MagicMock, patch

from mflux.models.common.training.runner import TrainingRunner
from mflux.models.common.training.state.training_spec import TrainingSpec


class TestDimensionResolution:
    def test_no_specified_dimensions_small_image(self):
        # Given: An image smaller than 1024x1024
        mock_spec = MagicMock(spec=TrainingSpec)
        mock_spec.max_resolution = None

        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (800, 600)
            mock_open.return_value.__enter__.return_value = mock_img

            # When: Resolving dimensions
            width, height = TrainingRunner._resolve_data_dimensions(
                training_spec=mock_spec, image_path=Path("dummy.jpg")
            )

            # Then: Dimensions should be adjusted to multiples of 16 (800, 592)
            # 800 / 16 = 50.0
            # 600 / 16 = 37.5 -> 37 * 16 = 592
            assert width == 800
            assert height == 592

    def test_no_specified_dimensions_large_image(self):
        # Given: An image larger than 1024 in one dimension (e.g., 2048x1024)
        mock_spec = MagicMock(spec=TrainingSpec)
        mock_spec.max_resolution = None

        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (2000, 1000)
            mock_open.return_value.__enter__.return_value = mock_img

            # When: Resolving dimensions
            width, height = TrainingRunner._resolve_data_dimensions(
                training_spec=mock_spec, image_path=Path("dummy.jpg")
            )

            # Then: Dimensions should only be adjusted to multiples of 16.
            # 2000 -> 2000 (already divisible by 16)
            # 1000 -> 992 (floor to multiple of 16)
            assert width == 2000
            assert height == 992

    def test_no_specified_dimensions_tall_large_image(self):
        # Given: A tall image larger than 1024 (e.g., 1000x2000)
        mock_spec = MagicMock(spec=TrainingSpec)
        mock_spec.max_resolution = None

        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (1000, 2000)
            mock_open.return_value.__enter__.return_value = mock_img

            # When: Resolving dimensions
            width, height = TrainingRunner._resolve_data_dimensions(
                training_spec=mock_spec, image_path=Path("dummy.jpg")
            )

            # Then: Dimensions should only be adjusted to multiples of 16.
            # 1000 -> 992
            # 2000 -> 2000
            assert width == 992
            assert height == 2000
