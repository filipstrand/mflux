from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_depth_test_helper import ImageGeneratorDepthTestHelper


class TestImageGeneratorDepth:
    SOURCE_IMAGE_FILENAME = "reference_controlnet_dev_lora.png"

    def test_image_generation_with_reference_image(self):
        ImageGeneratorDepthTestHelper.assert_matches_reference_image(
            reference_image_path="reference_depth_dev_from_image.png",
            output_image_path="output_depth_dev_from_image.png",
            image_path=TestImageGeneratorDepth.SOURCE_IMAGE_FILENAME,
            model_config=ModelConfig.dev_depth(),
            steps=15,
            seed=42,
            height=512,
            width=320,
            prompt="Cartoon picture of einstein with a cane",
        )
