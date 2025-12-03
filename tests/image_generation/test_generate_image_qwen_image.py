import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class TestImageGeneratorQwenImage:
    @pytest.mark.slow
    def test_qwen_image_generation_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_txt2img.png",
            output_image_path="output_qwen_txt2img.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            quantize=6,  # We should probably use at least 8-bit, but it doesn't run on 32GB machines
            steps=20,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
            negative_prompt="ugly, blurry, low quality",
            mismatch_threshold=0.35,  # Qwen models produce visually similar images with minor pixel differences
        )

    @pytest.mark.slow
    def test_qwen_image_generation_image_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_img2img.png",
            output_image_path="output_qwen_img2img.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            quantize=6,  # We should probably use at least 8-bit, but it doesn't run on 32GB machines
            steps=20,
            seed=44,
            height=341,
            width=768,
            image_path="reference_dev_image_to_image.png",
            image_strength=0.4,
            prompt="Luxury food photograph of a burger",
            negative_prompt="ugly, blurry, low quality",
            mismatch_threshold=0.35,  # Qwen models produce visually similar images with minor pixel differences
        )

    @pytest.mark.slow
    def test_qwen_image_generation_lora(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_lora.png",
            output_image_path="output_qwen_lora.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            guidance=1.0,
            quantize=6,  # We should probably use at least 8-bit, but it doesn't run on 32GB machines
            steps=4,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
            negative_prompt="ugly, blurry, low quality",
            lora_paths=["lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0.safetensors"],
            lora_scales=[1.0],
            mismatch_threshold=0.65,  # LoRA tests have higher variance due to model updates
        )
