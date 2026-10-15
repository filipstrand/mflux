import sys

import pytest

from mflux.models.flux.cli import flux_generate


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "flux3"],
        ["--model", "flux3-image-klein"],
        ["--model", "flux3-base"],
        ["--model", "/models/flux3-klein-q4", "--base-model", "flux3-klein-base"],
    ],
)
def test_mflux_generate_rejects_flux3(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate", "--prompt", "a fox", *argv])
    with pytest.raises(SystemExit) as exit_info:
        flux_generate.main()
    assert exit_info.value.code == 2
    assert "mflux-generate-flux3" in capsys.readouterr().err
