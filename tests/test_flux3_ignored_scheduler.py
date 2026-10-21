import sys
import warnings

import pytest

from mflux.cli.capabilities import describe_command
from mflux.models.flux3.cli import flux3_edit_generate, flux3_generate

CLIS = [flux3_generate, flux3_edit_generate]


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
def test_scheduler_is_declared_ignored_and_reported_by_capabilities(cli):
    assert "--scheduler" in cli.IGNORED_OPTIONS
    entry = describe_command("cmd", cli.__name__)
    option = next(option for option in entry["options"] if option["flag"] == "--scheduler")
    assert option["status"] == "ignored"


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
def test_main_warns_when_scheduler_is_passed(monkeypatch, tmp_path, cli):
    reference = tmp_path / "ref.png"
    reference.touch()
    argv = ["prog", "--prompt", "x", "--scheduler", "linear"]
    if cli is flux3_edit_generate:
        argv += ["--image-paths", str(reference)]
    monkeypatch.setattr(sys, "argv", argv)
    stop = RuntimeError("model load reached")

    def explode(*args, **kwargs):
        raise stop

    monkeypatch.setattr(cli, "Flux3Klein" if cli is flux3_generate else "Flux3KleinEdit", explode)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(RuntimeError):
        warnings.simplefilter("always")
        cli.main()
    assert any("--scheduler is ignored" in str(w.message) for w in caught)
