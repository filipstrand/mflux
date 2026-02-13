from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.training_adapter import (
    flux2_edit_training_adapter as flux2_edit_adapter_module,
    flux2_training_adapter as flux2_adapter_module,
)


class _FakeFlux2:
    def __init__(self, *, model_config, quantize):  # noqa: ARG002
        self.transformer = object()
        self.prompt_cache = {}
        self.last_generate_kwargs = None

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return SimpleNamespace(image="preview-image")


@pytest.mark.fast
def test_flux2_preview_image_uses_training_guidance(monkeypatch):
    monkeypatch.setattr(flux2_adapter_module, "Flux2Klein", _FakeFlux2)

    adapter = flux2_adapter_module.Flux2TrainingAdapter(model_config=ModelConfig.flux2_klein_4b(), quantize=None)
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())

    training_spec = SimpleNamespace(steps=12, guidance=3.5)
    adapter.create_config(training_spec=training_spec, width=1024, height=1024)
    adapter.generate_preview_image(seed=7, prompt="test", width=1024, height=1024, steps=6)

    assert adapter._flux2.last_generate_kwargs is not None
    assert adapter._flux2.last_generate_kwargs["guidance"] == 3.5


@pytest.mark.fast
def test_flux2_edit_preview_image_uses_training_guidance(monkeypatch):
    monkeypatch.setattr(flux2_edit_adapter_module, "Flux2KleinEdit", _FakeFlux2)

    adapter = flux2_edit_adapter_module.Flux2EditTrainingAdapter(
        model_config=ModelConfig.flux2_klein_4b(), quantize=None
    )
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())

    training_spec = SimpleNamespace(steps=12, guidance=4.0)
    adapter.create_config(training_spec=training_spec, width=1024, height=1024)
    adapter.generate_preview_image(
        seed=7,
        prompt="test",
        width=1024,
        height=1024,
        steps=6,
        image_paths=["/tmp/preview.png"],
    )

    assert adapter._flux2.last_generate_kwargs is not None
    assert adapter._flux2.last_generate_kwargs["guidance"] == 4.0
