from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_redux_test_helper import ImageGeneratorReduxTestHelper


class TestImageGeneratorRedux:
    def test_image_generation_redux(self):
        ImageGeneratorReduxTestHelper.assert_matches_reference_image(
            reference_image_path="reference_redux_dev.png",
            output_image_path="output_redux_dev.png",
            model_config=ModelConfig.dev_redux(),
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="A delicious burger",
            redux_image_path="reference_dev_image_to_image.png",
        )
