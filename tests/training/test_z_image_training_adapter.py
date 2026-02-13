from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.training_adapter import z_image_training_adapter as z_adapter_module


class _FakeZImage:
    def __init__(self, *, quantize, model_config):  # noqa: ARG002
        self.transformer = object()
        self.last_generate_kwargs = None

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return "preview-image"


@pytest.mark.fast
def test_preview_image_uses_training_guidance(monkeypatch):
    monkeypatch.setattr(z_adapter_module, "ZImage", _FakeZImage)

    adapter = z_adapter_module.ZImageTrainingAdapter(model_config=ModelConfig.z_image(), quantize=None)
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())

    training_spec = SimpleNamespace(steps=12, guidance=5.0)
    adapter.create_config(training_spec=training_spec, width=1024, height=1024)
    adapter.generate_preview_image(seed=7, prompt="test", width=1024, height=1024, steps=6)

    assert adapter._z.last_generate_kwargs is not None
    assert adapter._z.last_generate_kwargs["guidance"] == 5.0
