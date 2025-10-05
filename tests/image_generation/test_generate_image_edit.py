from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
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

    def test_image_generation_qwen_edit(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit.png",
            output_image_path="output_qwen_edit.png",
            model_class=QwenImageEdit,
            model_config=ModelConfig.qwen_image_edit(),
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=6,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
        )
