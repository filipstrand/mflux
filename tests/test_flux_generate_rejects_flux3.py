import sys

import pytest

from mflux.models.flux.cli import flux_generate


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "flux3"],
        ["--model", "flux3-image-klein"],
        ["--model", "flux3-base"],
        ["--model", "black-forest-labs/flux-3-image-klein"],
        ["--model", "black-forest-labs/flux-3-image-klein-base"],
        ["--model", "/models/flux-3-klein-q4"],
        ["--model", "/models/flux3-klein-q4", "--base-model", "flux3-klein-base"],
        ["--model", "flux3", "--base-model", "dev"],
    ],
)
def test_mflux_generate_rejects_flux3(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate", "--prompt", "a fox", *argv])
    with pytest.raises(SystemExit) as exit_info:
        flux_generate.main()
    assert exit_info.value.code == 2
    assert "mflux-generate-flux3" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "dev"],
        ["--model", "schnell"],
        ["--model", "/models/flux-1-dev-q4", "--base-model", "dev"],
        ["--model", "/models/flux3-experiments/flux-1-dev-q4", "--base-model", "dev"],
        ["--model", "/models/my-flux3-style-finetune", "--base-model", "dev"],
    ],
)
def test_mflux_generate_still_accepts_flux1_names(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate", "--prompt", "a fox", *argv])
    monkeypatch.setattr(
        flux_generate, "Flux1", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model load reached"))
    )
    with pytest.raises(RuntimeError, match="model load reached"):
        flux_generate.main()
