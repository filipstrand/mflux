from types import SimpleNamespace

from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit


def test_qwen_edit_default_dimensions_preserve_reference_image(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, vl_width, vl_height, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.qwen_image_edit()),
        image_paths=[str(image_path)],
        num_inference_steps=4,
        height=None,
        width=None,
        guidance=2.5,
        image_path=image_path,
        scheduler="linear",
    )

    assert (config.width, config.height) == (512, 384)
    assert (vae_width, vae_height) == (512, 384)
    assert vl_width % 32 == 0
    assert vl_height % 32 == 0


def test_qwen_edit_explicit_dimensions_override_reference_image(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, _, _, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.qwen_image_edit()),
        image_paths=[str(image_path)],
        num_inference_steps=4,
        height=256,
        width=320,
        guidance=2.5,
        image_path=image_path,
        scheduler="linear",
    )

    assert (config.width, config.height) == (320, 256)
    assert (vae_width, vae_height) == (320, 256)
