import os

# Set HF_HOME to use existing cache (only if not already set)
# Check if model exists at default location first
default_hf_home = os.path.expanduser("~/.cache/huggingface")
shared_hf_home = "/Users/Shared/.cache/huggingface"

# Use shared location if it exists and has the model, otherwise use default
if os.path.exists(shared_hf_home) and os.path.exists(f"{shared_hf_home}/hub/models--Qwen--Qwen-Image"):
    os.environ["HF_HOME"] = shared_hf_home
elif not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = default_hf_home

# Preserve test output images so they can be visually inspected
os.environ["MFLUX_PRESERVE_TEST_OUTPUT"] = "1"

from mflux.config.model_config import ModelConfig  # noqa: E402
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage  # noqa: E402
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper  # noqa: E402


# Lazy import Flux1 to avoid import errors when only running Qwen tests
def _get_flux1():
    from mflux.models.flux.variants.txt2img.flux import Flux1

    return Flux1


class TestImageGenerator:
    def test_image_generation_schnell(self):
        Flux1 = _get_flux1()
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
        Flux1 = _get_flux1()
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
        Flux1 = _get_flux1()
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
        Flux1 = _get_flux1()
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
        Flux1 = _get_flux1()
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
            reference_image_path="reference_qwen_lora.png",
            output_image_path="output_qwen_lora.png",
            model_class=QwenImage,
            model_config=ModelConfig.qwen_image(),
            guidance=1.0,
            quantize=6,
            steps=4,
            seed=42,
            height=341,
            width=768,
            prompt="Luxury food photograph",
            negative_prompt="ugly, blurry, low quality",
            lora_repo_id="lightx2v/Qwen-Image-Lightning",
            lora_names=["Qwen-Image-Lightning-4steps-V2.0.safetensors"],
        )
