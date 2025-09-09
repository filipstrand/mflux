from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class TestImageGenerator:
    def test_image_generation_schnell(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_schnell.png",
            output_image_path="output_schnell.png",
            model_class=Flux1,
            model_config=ModelConfig.schnell(),
            steps=2,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
        )

    def test_image_generation_dev(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev.png",
            output_image_path="output_dev.png",
            model_class=Flux1,
            model_config=ModelConfig.dev(),
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
        )

    def test_image_generation_dev_lora(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_lora.png",
            output_image_path="output_dev_lora.png",
            model_class=Flux1,
            model_config=ModelConfig.dev(),
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="mkym this is made of wool, burger",
            lora_paths=["FLUX-dev-lora-MiaoKa-Yarn-World.safetensors"],
            lora_scales=[1.0],
        )

    def test_image_generation_dev_multiple_loras(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_lora_multiple.png",
            output_image_path="output_dev_lora_multiple.png",
            model_class=Flux1,
            model_config=ModelConfig.dev(),
            steps=15,
            seed=42,
            height=341,
            width=768,
            prompt="Renaissance painting, mkym this is made of wool, burger",
            lora_paths=["FLUX-dev-lora-MiaoKa-Yarn-World.safetensors", "Flux_-_Renaissance_art_style.safetensors"],
            lora_scales=[0.4, 0.6],
        )

    def test_image_generation_dev_image_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_dev_image_to_image.png",
            output_image_path="output_dev_image_to_image.png",
            model_class=Flux1,
            image_strength=0.4,
            model_config=ModelConfig.dev(),
            steps=8,
            seed=44,
            height=341,
            width=768,
            image_path="reference_dev_lora.png",
            prompt="Luxury food photograph of a burger",
        )

    def test_qwen_image_generation_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_txt2img.png",
            output_image_path="output_qwen_txt2img.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            quantize=6,
            steps=20,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
            negative_prompt="ugly, blurry, low quality",
        )

    def test_qwen_image_generation_image_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_img2img.png",
            output_image_path="output_qwen_img2img.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            quantize=6,
            steps=20,
            seed=44,
            height=341,
            width=768,
            image_path="reference_dev_image_to_image.png",
            image_strength=0.4,
            prompt="Luxury food photograph of a burger",
            negative_prompt="ugly, blurry, low quality",
        )

    def test_qwen_image_generation_lora(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_lora_hf.png",
            output_image_path="output_qwen_lora_hf.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            quantize=6,
            steps=8,
            seed=42,
            height=512,
            width=512,
            prompt="Luxury food photograph",
            negative_prompt="ugly, blurry, low quality",
            lora_repo_id="lightx2v/Qwen-Image-Lightning",
            lora_names=["Qwen-Image-Lightning-8steps-V1.1.safetensors"],
        )
