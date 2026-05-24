import sys
from pathlib import Path

import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.cli.flux2_model_validation import is_flux2_base_model, is_flux2_model


class _SavedImage:
    def save(self, path, *args, **kwargs):  # noqa: ARG002
        Path(path).write_bytes(b"fake image")


class _Callbacks:
    def register(self, callback):  # noqa: ARG002
        return None


def _fake_model_class():
    class FakeModel:
        instance = None

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.generate_kwargs = None
            self.callbacks = _Callbacks()
            FakeModel.instance = self

        def generate_image(self, **kwargs):
            self.generate_kwargs = kwargs
            return _SavedImage()

    return FakeModel


def test_flux2_validation_recognizes_inferred_local_base_path():
    model_config = ModelConfig.from_name("/models/local-flux2-klein-base-9b-q4")

    assert is_flux2_model(model_config)
    assert is_flux2_base_model(model_config)


def test_flux2_edit_accepts_distilled_guidance(monkeypatch, tmp_path, capsys):
    fake_model = _fake_model_class()
    output_path = tmp_path / "out.png"
    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinEdit", fake_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-4b",
            "--image-paths",
            "reference.png",
            "--prompt",
            "make the scene cinematic",
            "--steps",
            "1",
            "--seed",
            "7",
            "--width",
            "64",
            "--height",
            "64",
            "--guidance",
            "2.0",
            "--output",
            str(output_path),
        ],
    )

    flux2_edit_generate.main()

    captured = capsys.readouterr()
    assert "distilled FLUX.2 Klein" in captured.out
    assert output_path.exists()
    assert fake_model.instance.init_kwargs["model_config"].model_name == "black-forest-labs/FLUX.2-klein-4B"
    assert fake_model.instance.generate_kwargs["guidance"] == pytest.approx(2.0)


def test_flux2_edit_rejects_non_flux2_model(monkeypatch, tmp_path, capsys):
    fake_model = _fake_model_class()
    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinEdit", fake_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "schnell",
            "--image-paths",
            "reference.png",
            "--prompt",
            "edit",
            "--steps",
            "1",
            "--seed",
            "7",
            "--width",
            "64",
            "--height",
            "64",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        flux2_edit_generate.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "only supports FLUX.2 Klein models" in captured.err
    assert fake_model.instance is None


def test_flux2_txt2img_rejects_distilled_guidance_before_loading(monkeypatch, tmp_path, capsys):
    fake_model = _fake_model_class()
    monkeypatch.setattr(flux2_generate, "Flux2Klein", fake_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-flux2",
            "--model",
            "flux2-klein-4b",
            "--prompt",
            "a landscape",
            "--steps",
            "1",
            "--seed",
            "7",
            "--width",
            "64",
            "--height",
            "64",
            "--guidance",
            "2.0",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        flux2_generate.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "requires a FLUX.2 Klein base model" in captured.err
    assert fake_model.instance is None
