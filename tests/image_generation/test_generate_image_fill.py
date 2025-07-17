from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_fill_test_helper import ImageGeneratorFillTestHelper


class TestImageGeneratorFill:
    SOURCE_IMAGE_FILENAME = "reference_dev_image_to_image_result.png"
    MASK_IMAGE_FILENAME = "mask.png"

    def test_fill(self):
        ImageGeneratorFillTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_image_to_image_result_inpaint.png",
            output_image_path="output_dev_image_to_image_result_inpaint.png",
            model_config=ModelConfig.dev_fill(),
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="A burger and crispy golden french fries next to it",
            image_path=TestImageGeneratorFill.SOURCE_IMAGE_FILENAME,
            masked_image_path=TestImageGeneratorFill.MASK_IMAGE_FILENAME,
        )
