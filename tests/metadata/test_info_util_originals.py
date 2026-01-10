from mflux.utils.info_util import InfoUtil


def test_info_util_includes_original_generation_section():
    metadata = {
        "exif": {
            "prompt": "test prompt",
            "model": "numz/SeedVR2_comfyUI",
            "width": 2896,
            "height": 2160,
            "seed": 123,
            "steps": 1,
            "guidance": 1.0,
            "precision": "mlx.core.bfloat16",
            "original_model": "Tongyi-MAI/Z-Image-Turbo",
            "original_width": 1440,
            "original_height": 1072,
            "original_seed": 123,
            "original_steps": 9,
            "original_quantize": 8,
            "original_lora_paths": ["/tmp/a.safetensors"],
            "original_lora_scales": [0.9],
        }
    }

    output = InfoUtil.format_metadata(metadata)

    assert "Original Generation:" in output
    assert "Model: numz/SeedVR2_comfyUI (Original: Tongyi-MAI/Z-Image-Turbo)" in output
    assert "Width: 2896 (Original: 1440)" in output
    assert "Height: 2160 (Original: 1072)" in output
    assert "Steps: 1 (Original: 9)" in output
    assert "Quantization: 8-bit" in output
    assert "a.safetensors" in output
