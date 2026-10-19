# `mflux-save --hf-symlink` must link the saved folder into the Hugging Face cache, take the
# username from the documented sources in priority order, and refuse a bad name before any load.

import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.models.common.cli import save
from mflux.models.common.weights.saving import hf_cache_alias
from mflux.models.common.weights.saving.hf_cache_alias import HfCacheAlias

ENV_VARS = HfCacheAlias.USERNAME_ENV_VARS


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setattr(hf_cache_alias.hf_constants, "HF_HUB_CACHE", str(cache))
    for env_var in ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(hf_cache_alias.getpass, "getuser", lambda: "osuser")
    return cache


def make_saved_model(root: Path) -> Path:
    model_dir = root / "my-model_8bit"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "0.safetensors").write_bytes(b"weights")
    (model_dir / "model_index.json").write_text("{}")
    (model_dir / ".DS_Store").write_text("")
    return model_dir


def run_save(argv_extra: list[str], path: Path) -> list[str]:
    # Stubs the load and the save so the CLI runs without weights. Returns what the stubbed save saw.
    calls = []

    def fake_save(self, base_path):
        calls.append(base_path)
        make_saved_model(Path(base_path).parent)

    argv = ["mflux-save", "--model", "schnell", "--path", str(path), *argv_extra]
    model_class = save.MODEL_CLASSES["schnell"]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(model_class, "__init__", lambda self, **kwargs: None))
        stack.enter_context(patch.object(model_class, "save_model", fake_save))
        save.main()
    return calls


@pytest.mark.fast
def test_create_links_every_visible_file(hf_cache, tmp_path):
    model_dir = make_saved_model(tmp_path)

    result = HfCacheAlias.create(model_dir, "alice")

    assert result.repo_id == "alice/my-model_8bit"
    assert result.file_count == 2
    snapshot = hf_cache / "models--alice--my-model_8bit" / "snapshots" / result.revision
    assert (snapshot / "transformer" / "0.safetensors").resolve() == (
        model_dir / "transformer" / "0.safetensors"
    ).resolve()
    assert (snapshot / "transformer" / "0.safetensors").is_symlink()
    assert not (snapshot / ".DS_Store").exists()
    assert (hf_cache / "models--alice--my-model_8bit" / "refs" / "main").read_text() == result.revision


@pytest.mark.fast
def test_create_twice_replaces_the_snapshot(hf_cache, tmp_path):
    model_dir = make_saved_model(tmp_path)
    HfCacheAlias.create(model_dir, "alice")
    (model_dir / "model_index.json").unlink()

    result = HfCacheAlias.create(model_dir, "alice")

    assert result.file_count == 1


@pytest.mark.fast
def test_check_refuses_a_cache_entry_mflux_did_not_make(hf_cache, tmp_path):
    model_dir = make_saved_model(tmp_path)
    (hf_cache / "models--alice--my-model_8bit").mkdir(parents=True)

    with pytest.raises(ValueError, match="mflux did not make it"):
        HfCacheAlias.check(model_dir, "alice")


@pytest.mark.fast
@pytest.mark.parametrize("username", ["has space", "a/b", "", "-alice", "al..ice"])
def test_check_refuses_an_invalid_name(hf_cache, tmp_path, username):
    with pytest.raises(ValueError, match="not a valid Hugging Face repo id"):
        HfCacheAlias.check(make_saved_model(tmp_path), username)


@pytest.mark.fast
@pytest.mark.parametrize("folder", ["model..8bit", "model.git", "has space"])
def test_check_refuses_an_invalid_folder_name(hf_cache, tmp_path, folder):
    # The folder name becomes the repo name, so the Hub's rule applies to it as well.
    with pytest.raises(ValueError, match="not a valid Hugging Face repo id"):
        HfCacheAlias.check(tmp_path / folder, "alice")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("cli_name", "env", "expected"),
    [
        (None, {}, ("osuser", "OS user name")),
        ("", {}, ("osuser", "OS user name")),
        (None, {"HF_NAME": "name"}, ("name", "HF_NAME")),
        (None, {"HF_NAME": "name", "HF_NAMESPACE": "ns"}, ("ns", "HF_NAMESPACE")),
        (None, {"HF_NAME": "name", "HF_NAMESPACE": "ns", "MFLUX_HF_USERNAME": "mf"}, ("mf", "MFLUX_HF_USERNAME")),
        ("cli", {"HF_NAME": "name", "HF_NAMESPACE": "ns", "MFLUX_HF_USERNAME": "mf"}, ("cli", "--hf-symlink")),
        (None, {"MFLUX_HF_USERNAME": "", "HF_NAME": "name"}, ("name", "HF_NAME")),
    ],
)
def test_username_priority(hf_cache, monkeypatch, cli_name, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert HfCacheAlias.resolve_username(cli_name) == expected


@pytest.mark.fast
def test_cli_without_flag_makes_no_alias(hf_cache, tmp_path):
    run_save([], tmp_path / "my-model_8bit")

    assert not hf_cache.exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv_extra", "expected_user"), [(["--hf-symlink"], "osuser"), (["--hf-symlink", "bob"], "bob")]
)
def test_cli_flag_makes_the_alias_after_the_save(hf_cache, tmp_path, argv_extra, expected_user):
    run_save(argv_extra, tmp_path / "my-model_8bit")

    assert (hf_cache / f"models--{expected_user}--my-model_8bit" / HfCacheAlias.MARKER).exists()


@pytest.mark.fast
def test_cli_bad_name_exits_before_the_save(hf_cache, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_save(["--hf-symlink", "bad name"], tmp_path / "my-model_8bit")

    assert exit_info.value.code == 2
    # The stubbed save writes this folder, so its absence shows the save never ran.
    assert not (tmp_path / "my-model_8bit").exists()
