import pytest

from tests.image_generation.helpers.image_generation_moss_image_test_helper import ImageGeneratorMOSSImageTestHelper

BICYCLE_PROMPT = (
    "A vintage red bicycle leaning against a weathered cobblestone wall covered in green ivy, "
    "narrow European alley, warm golden afternoon sunlight, shallow depth of field, photorealistic."
)


class TestImageGeneratorMOSSImage:
    @pytest.mark.slow
    def test_image_generation_moss_image_sft(self):
        ImageGeneratorMOSSImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_moss_image_sft.png",
            output_image_path="output_moss_image_sft.png",
            prompt=BICYCLE_PROMPT,
            steps=20,
            seed=7,
            height=368,
            width=640,
            quantize=8,
        )
