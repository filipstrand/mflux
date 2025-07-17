from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_ic_edit_test_helper import ImageGeneratorICEditTestHelper
from tests.image_generation.helpers.image_generation_in_context_test_helper import ImageGeneratorInContextTestHelper


class TestImageGeneratorInContext:
    def test_in_context_lora_identity(self):
        ImageGeneratorInContextTestHelper.assert_matches_reference_image(
            reference_image_path="ic_lora_reference_in_context_identity.png",
            output_image_path="output_ic_lora_reference_in_context_identity.png",
            model_config=ModelConfig.dev(),
            steps=25,
            seed=42,
            height=320,
            width=320,
            prompt="In this set of two images, a bold modern typeface with the brand name 'DEMA' is introduced and is shown on a company merchandise product photo; [IMAGE1] a simplistic black logo featuring a modern typeface with the brand name 'DEMA' on a bright light green/yellowish background; [IMAGE2] the design is printed on a green/yellowish hoodie as a company merchandise product photo with a plain white background.",
            image_path="ic_lora_reference_logo.png",
            lora_style="identity",
        )

    def test_ic_edit_with_instruction(self):
        ImageGeneratorICEditTestHelper.assert_matches_reference_image(
            reference_image_path="ic_edit_reference.png",
            output_image_path="output_ic_edit.png",
            steps=20,
            seed=2799665,
            height=224,
            width=512,
            prompt="A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but remove the cheese",
            reference_image="reference_redux_dev.png",
        )
