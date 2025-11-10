import os

# Set HF_HOME to use existing cache (only if not already set)
# Check if model exists at default location first
default_hf_home = os.path.expanduser("~/.cache/huggingface")
shared_hf_home = "/Users/Shared/.cache/huggingface"

# Check for edit models in default location first (they're there)
if os.path.exists(f"{default_hf_home}/hub/models--Qwen--Qwen-Image-Edit") or os.path.exists(
    f"{default_hf_home}/hub/models--Qwen--Qwen-Image-Edit-2509"
):
    os.environ["HF_HOME"] = default_hf_home
    os.environ["HF_HUB_CACHE"] = f"{default_hf_home}/hub"
# Use shared location if it exists and has the regular Qwen-Image model
elif os.path.exists(shared_hf_home) and os.path.exists(f"{shared_hf_home}/hub/models--Qwen--Qwen-Image"):
    os.environ["HF_HOME"] = shared_hf_home
    os.environ["HF_HUB_CACHE"] = f"{shared_hf_home}/hub"
elif not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = default_hf_home
    os.environ["HF_HUB_CACHE"] = f"{default_hf_home}/hub"

# Preserve test output images so they can be visually inspected
os.environ["MFLUX_PRESERVE_TEST_OUTPUT"] = "1"

from mflux.config.model_config import ModelConfig  # noqa: E402
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit  # noqa: E402
from tests.image_generation.helpers.image_generation_edit_test_helper import ImageGeneratorEditTestHelper  # noqa: E402


# Lazy import Flux1Kontext to avoid import errors when only running Qwen tests
def _get_flux1_kontext():
    from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

    return Flux1Kontext


# Lazy import QwenImageEditPlus to avoid import errors when only running regular edit tests
def _get_qwen_image_edit_plus():
    from mflux.models.qwen.variants.edit.qwen_image_edit_plus import QwenImageEditPlus

    return QwenImageEditPlus


class TestImageGeneratorEdit:
    def test_image_generation_flux_kontext(self):
        Flux1Kontext = _get_flux1_kontext()
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
            quantize=6,  # Test with quantization to reproduce garbage output issue
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
        )

    def test_image_generation_qwen_edit_plus(self):
        QwenImageEditPlus = _get_qwen_image_edit_plus()
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
            quantize=6,  # Test with quantization
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
        )
