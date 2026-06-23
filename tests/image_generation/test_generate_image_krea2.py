import sys

import pytest


@pytest.mark.fast
def test_krea2_cli_defaults_to_turbo_zero_guidance(monkeypatch):
    from mflux.models.krea2.cli import krea2_generate

    captured = {}

    class FakeImage:
        def save(self, path: str, export_json_metadata: bool) -> None:
            captured["save_path"] = path
            captured["export_json_metadata"] = export_json_metadata

    class FakeKrea2Image:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def generate_image(self, **kwargs) -> FakeImage:
            captured["generate"] = kwargs
            return FakeImage()

    monkeypatch.setattr(krea2_generate, "Krea2Image", FakeKrea2Image)
    monkeypatch.setattr(krea2_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-krea2",
            "--prompt",
            "a fox walking in the snow",
        ],
    )

    krea2_generate.main()

    assert captured["init"]["model_config"].model_name == "krea/Krea-2-Turbo"
    assert captured["generate"]["num_inference_steps"] == 8
    assert captured["generate"]["guidance"] == pytest.approx(0.0)
    assert captured["generate"]["scheduler"] == "flow_match_euler_discrete"
    assert captured["save_path"] == "image.png"


@pytest.mark.fast
def test_krea2_cli_rejects_guidance_for_turbo(monkeypatch):
    from mflux.models.krea2.cli import krea2_generate

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-krea2",
            "--prompt",
            "a fox walking in the snow",
            "--guidance",
            "3.5",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        krea2_generate.main()

    assert exc_info.value.code == 2


@pytest.mark.fast
def test_krea2_image_rejects_guidance_for_turbo_api():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2Image

    model = _Krea2ImageHarness()
    model.model_config = ModelConfig.krea2_turbo()

    with pytest.raises(ValueError, match="guidance"):
        Krea2Image.generate_image(model, seed=1, prompt="a fox", guidance=3.5)


class _Krea2ImageHarness:
    pass
