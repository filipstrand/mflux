from mflux.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_controlnet_test_helper import ImageGeneratorControlnetTestHelper


class TestImageGeneratorControlnet:
    CONTROLNET_REFERENCE_FILENAME = "controlnet_reference.png"

    def test_image_generation_schnell_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_schnell.png",
            output_image_path="output_controlnet_schnell.png",
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.schnell_controlnet_canny(),
            steps=2,
            seed=43,
            height=768,
            width=493,
            prompt="The joker with a hat and a cane",
            controlnet_strength=0.4,
        )

    def test_image_generation_dev_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_dev.png",
            output_image_path="output_controlnet_dev.png",
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.dev_controlnet_canny(),
            steps=15,
            seed=42,
            height=768,
            width=493,
            prompt="The joker with a hat and a cane",
            controlnet_strength=0.4,
        )

    def test_image_generation_dev_lora_controlnet(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_controlnet_dev_lora.png",
            output_image_path="output_controlnet_dev_lora.png",
            controlnet_image_path=TestImageGeneratorControlnet.CONTROLNET_REFERENCE_FILENAME,
            model_config=ModelConfig.dev_controlnet_canny(),
            steps=15,
            seed=43,
            height=768,
            width=493,
            prompt="mkym this is made of wool, The joker with a hat and a cane",
            lora_paths=["FLUX-dev-lora-MiaoKa-Yarn-World.safetensors"],
            lora_scales=[1.0],
            controlnet_strength=0.4,
        )

    def test_image_upscaling(self):
        ImageGeneratorControlnetTestHelper.assert_matches_reference_image(
            reference_image_path="reference_upscaled.png",
            output_image_path="output_upscaler.png",
            controlnet_image_path="low_res.jpg",
            model_config=ModelConfig.dev_controlnet_upscaler(),
            steps=20,
            seed=42,
            height=int(192 * 2),
            width=int(320 * 2),
            prompt="A man holding up a hand",
            controlnet_strength=0.6,
        )
