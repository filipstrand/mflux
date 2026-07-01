import mlx.core as mx

from mflux.models.qwen.variants.edit import qwen_edit_util as qwen_edit_util_module
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil


def test_image_conditioning_latents_use_vae_target_dimensions(monkeypatch):
    captured = {}

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        captured["encode_height"] = height
        captured["encode_width"] = width
        return mx.zeros((1, height // 8, width // 8, 16))

    def fake_pack_latents(latents, height, width, num_channels_latents):
        captured["pack_height"] = height
        captured["pack_width"] = width
        captured["num_channels_latents"] = num_channels_latents
        return mx.zeros((1, (height // 16) * (width // 16), num_channels_latents * 4))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)
    monkeypatch.setattr(qwen_edit_util_module.QwenLatentCreator, "pack_latents", fake_pack_latents)

    _, image_ids, cond_h_patches, cond_w_patches, num_images = QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=1024,
        width=768,
        image_paths="input.png",
        tiling_config=None,
    )

    assert captured == {
        "encode_height": 1024,
        "encode_width": 768,
        "pack_height": 1024,
        "pack_width": 768,
        "num_channels_latents": 16,
    }
    assert image_ids.shape == (1, 64 * 48, 3)
    assert cond_h_patches == 64
    assert cond_w_patches == 48
    assert num_images == 1
