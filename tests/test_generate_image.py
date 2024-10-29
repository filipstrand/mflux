from mflux import ModelConfig
from tests.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class TestImageGenerator:
    OUTPUT_IMAGE_FILENAME = "output.png"

    def test_image_generation_schnell(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_schnell.png",
            output_image_path=TestImageGenerator.OUTPUT_IMAGE_FILENAME,
            model_config=ModelConfig.FLUX1_SCHNELL,
            steps=2,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
        )

    def test_image_generation_dev(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev.png",
            output_image_path=TestImageGenerator.OUTPUT_IMAGE_FILENAME,
            model_config=ModelConfig.FLUX1_DEV,
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
        )

    def test_image_generation_dev_lora(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_lora.png",
            output_image_path=TestImageGenerator.OUTPUT_IMAGE_FILENAME,
            model_config=ModelConfig.FLUX1_DEV,
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="mkym this is made of wool, burger",
            lora_paths=["FLUX-dev-lora-MiaoKa-Yarn-World.safetensors"],
            lora_scales=[1.0],
        )

    def test_image_generation_dev_image_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_image_to_image_result.png",
            init_image_path="reference_dev_lora.png",
            init_image_strength=0.4,
            output_image_path=TestImageGenerator.OUTPUT_IMAGE_FILENAME,
            model_config=ModelConfig.FLUX1_DEV,
            steps=8,
            seed=44,
            height=341,
            width=768,
            prompt="Luxury food photograph of a burger",
        )
