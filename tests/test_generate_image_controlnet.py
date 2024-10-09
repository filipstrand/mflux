from mflux import ModelConfig
from tests.helpers.image_generation_controlnet_test_helper import ImageGeneratorControlnetTestHelper


class TestImageGeneratorControlnet:
    OUTPUT_IMAGE_FILENAME = "output.png"
    CONTROLNET_REFERENCE_FILENAME = "controlnet_reference.png"

    def test_image_generation_schnell_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_schnell.png",
            output_image_path=TestImageGeneratorControlnet.OUTPUT_IMAGE_FILENAME,
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.FLUX1_SCHNELL,
            steps=2,
            seed=43,
            prompt="The joker with a hat and a cane",
            controlnet_strength=0.4,
        )

    def test_image_generation_dev_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_dev.png",
            output_image_path=TestImageGeneratorControlnet.OUTPUT_IMAGE_FILENAME,
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.FLUX1_DEV,
            steps=15,
            seed=42,
            prompt="The joker with a hat and a cane",
            controlnet_strength=0.4,
        )

    def test_image_generation_dev_lora_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_dev_lora.png",
            output_image_path=TestImageGeneratorControlnet.OUTPUT_IMAGE_FILENAME,
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.FLUX1_DEV,
            steps=15,
            seed=43,
            prompt="mkym this is made of wool, The joker with a hat and a cane",
            lora_paths=["FLUX-dev-lora-MiaoKa-Yarn-World.safetensors"],
            lora_scales=[1.0],
            controlnet_strength=0.4,
        )
