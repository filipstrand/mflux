import sys

import pytest


@pytest.mark.fast
def test_krea2_save_cli_dispatches_to_krea2(monkeypatch, tmp_path):
    from mflux.models.common.cli import save

    captured = {}

    class FakeKrea2Image:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def save_model(self, path: str) -> None:
            captured["save_path"] = path

    monkeypatch.setattr(save, "Krea2Image", FakeKrea2Image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-save",
            "--model",
            "krea2-turbo",
            "--path",
            str(tmp_path / "saved"),
        ],
    )

    save.main()

    assert captured["model_config"].model_name == "krea/Krea-2-Turbo"
    assert captured["model_path"] is None
    assert captured["save_path"] == str(tmp_path / "saved")


@pytest.mark.fast
def test_krea2_saved_text_encoder_weights_reload_inner_encoder():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    weight_definition = Krea2WeightDefinition.resolve(ModelConfig.krea2_turbo())
    components = {component.name: component for component in weight_definition.get_components()}
    recorder = _RecorderModule()
    weights = LoadedWeights(
        components={
            "text_encoder": {
                "encoder": {
                    "norm": {
                        "weight": "sentinel",
                    },
                },
            },
        },
        meta_data=MetaData(),
    )

    WeightApplier._set_weights(
        weights=weights,
        models={"text_encoder": recorder},
        components=components,
    )

    assert recorder.weights == {"norm": {"weight": "sentinel"}}


class _RecorderModule:
    def update(self, weights, strict: bool) -> None:
        self.weights = weights
        self.strict = strict
