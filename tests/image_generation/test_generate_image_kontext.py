from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_kontext_test_helper import ImageGeneratorKontextTestHelper


class TestImageGeneratorKontext:
    def test_image_generation_kontext(self):
        ImageGeneratorKontextTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_kontext.png",
            output_image_path="output_dev_kontext.png",
            model_config=ModelConfig.dev_kontext(),
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            kontext_image_path="reference_upscaled.png",
        )
