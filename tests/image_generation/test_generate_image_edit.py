from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.qwen.variants.edit import QwenImageEditPlus
from tests.image_generation.helpers.image_generation_edit_test_helper import ImageGeneratorEditTestHelper


class TestImageGeneratorEdit:
    def test_image_generation_flux_kontext(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_kontext.png",
            output_image_path="output_dev_kontext.png",
            model_class=Flux1Kontext,
            model_config=ModelConfig.dev_kontext(),
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
        )

    def test_image_generation_qwen_edit_plus(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit_plus.png",
            output_image_path="output_qwen_edit_plus.png",
            model_class=QwenImageEditPlus,
            model_config=ModelConfig.qwen_image_edit_plus(),
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=6,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
        )

    def test_image_generation_qwen_edit_plus_multiple_images(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit_plus_multiple_images.png",
            output_image_path="output_qwen_edit_plus_multiple_images.png",
            model_class=QwenImageEditPlus,
            model_config=ModelConfig.qwen_image_edit_plus(),
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=8,
            prompt="Make the hand fistbump the camera instead of showing a flat palm, and the man should wear this shirt. Maintain the original pose, body position, and overall stance.",
            image_paths=["reference_upscaled.png", "shirt.jpg"],
        )
