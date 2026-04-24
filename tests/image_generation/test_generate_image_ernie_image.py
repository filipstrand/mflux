import pytest

from tests.image_generation.helpers.image_generation_ernie_image_test_helper import ImageGeneratorErnieImageTestHelper

BOAT_LAKE_PROMPT = "a small red boat on a calm blue lake surrounded by pine trees"


class TestImageGeneratorErnieImage:
    @pytest.mark.slow
    def test_image_generation_ernie_image_turbo(self):
        ImageGeneratorErnieImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_ernie_image_turbo.png",
            output_image_path="output_ernie_image_turbo.png",
            prompt=BOAT_LAKE_PROMPT,
            steps=8,
            seed=42,
            height=368,
            width=640,
            guidance=1.0,
            quantize=8,
        )
