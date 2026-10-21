# `mflux-save --hf-symlink` must link the saved folder into the Hugging Face cache, take the
# username from the documented sources in priority order, and refuse a bad name before any load.

import contextlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.models.common.cli import save
from mflux.models.common.resolution import path_resolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving import hf_cache_alias
from mflux.models.common.weights.saving.hf_cache_alias import HfCacheAlias
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition

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


@pytest.mark.fast
def test_failed_rebuild_keeps_the_old_snapshot(hf_cache, tmp_path, monkeypatch):
    model_dir = make_saved_model(tmp_path)
    first = HfCacheAlias.create(model_dir, "alice")
    snapshot = hf_cache / "models--alice--my-model_8bit" / "snapshots" / first.revision

    def fail(self, source):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "symlink_to", fail)
    with pytest.raises(OSError):
        HfCacheAlias.create(model_dir, "alice")

    assert (snapshot / "model_index.json").is_symlink()
    assert sorted(p.name for p in snapshot.parent.iterdir()) == [first.revision]


@pytest.mark.fast
def test_interrupted_first_create_can_be_repaired(hf_cache, tmp_path, monkeypatch):
    model_dir = make_saved_model(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "symlink_to", lambda self, source: (_ for _ in ()).throw(OSError("stop")))
        with pytest.raises(OSError):
            HfCacheAlias.create(model_dir, "alice")

    result = HfCacheAlias.create(model_dir, "alice")

    assert result.file_count == 2
    assert not list((hf_cache / "models--alice--my-model_8bit").glob(".mflux_*-*"))


@pytest.mark.fast
def test_new_entry_is_never_without_its_marker(hf_cache, tmp_path, monkeypatch):
    # A stop before the marker is written must not leave an entry that check() refuses on the next run.
    model_dir = make_saved_model(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "write_text", lambda self, text: (_ for _ in ()).throw(OSError("stop")))
        with pytest.raises(OSError):
            HfCacheAlias.create(model_dir, "alice")

    assert not (hf_cache / "models--alice--my-model_8bit").exists()
    assert not list(hf_cache.glob(".mflux_alias-*"))
    assert HfCacheAlias.create(model_dir, "alice").file_count == 2


@pytest.mark.fast
def test_cli_expands_home_once_for_the_save_and_the_alias(hf_cache, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    calls = run_save(["--hf-symlink", "bob"], Path("~/my-model_8bit"))

    assert calls == [str((home / "my-model_8bit").resolve())]
    assert (hf_cache / "models--bob--my-model_8bit" / HfCacheAlias.MARKER).read_text().strip() == str(
        (home / "my-model_8bit").resolve()
    )


def make_component_save(root: Path, subdirs: list[str]) -> Path:
    # The layout ModelSaver writes: shards plus model.safetensors.index.json in each component folder.
    for subdir in subdirs:
        folder = root / subdir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "0.safetensors").write_bytes(b"w")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "0.safetensors"}}))
    return root


@pytest.mark.fast
@pytest.mark.parametrize(
    ("definition", "folder"),
    [
        (SeedVR2WeightDefinition, "SeedVR2_comfyUI-3b-mflux-q8"),
        (Krea2WeightDefinition, "Krea-2-Turbo-mflux-q8"),
    ],
    ids=["seedvr2", "krea2"],
)
def test_alias_of_a_saved_layout_loads_from_the_cache(hf_cache, tmp_path, monkeypatch, definition, folder):
    # These saves keep the saved layout, not the file names in the download patterns.
    # Without the saved-layout check, the alias missed the cache and went to the download path.
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(hf_cache))
    # Online, a download of this made-up repo id fails with a 404. The cache rule must answer first.
    monkeypatch.setattr(path_resolution, "snapshot_download", lambda **kwargs: pytest.fail("went to the download path"))
    saved_subdirs = list(ComponentDefinition.save_subdirs(definition.get_components()).values())
    model_dir = make_component_save(tmp_path / folder, saved_subdirs)
    alias = HfCacheAlias.create(model_dir, "alice")

    resolved = PathResolution.resolve(alias.repo_id, definition.get_download_patterns(), saved_subdirs)

    assert resolved == alias.repo_cache / "snapshots" / alias.revision


@pytest.mark.fast
def test_hf_layout_without_saved_index_is_not_taken_for_a_save(hf_cache, monkeypatch):
    # A download that misses the pattern files must not pass as complete only because its folders hold shards.
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(hf_cache))
    snapshot = hf_cache / "models--org--repo" / "snapshots" / "abc"
    for subdir in ("transformer", "vae"):
        (snapshot / subdir).mkdir(parents=True)
        (snapshot / subdir / "x.safetensors").write_bytes(b"w")

    assert (
        PathResolution._find_complete_cached_snapshot(
            "org/repo", ["seedvr2_ema_3b_fp16.safetensors"], ["transformer", "vae"]
        )
        is None
    )


@pytest.mark.fast
def test_failed_swap_restores_the_old_snapshot(hf_cache, tmp_path, monkeypatch):
    model_dir = make_saved_model(tmp_path)
    first = HfCacheAlias.create(model_dir, "alice")
    snapshot = hf_cache / "models--alice--my-model_8bit" / "snapshots" / first.revision
    real_rename = Path.rename

    def fail_on_install(self, target):
        if self.name.startswith(".mflux_building-"):
            raise OSError("rename failed")
        return real_rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", fail_on_install)
        with pytest.raises(OSError):
            HfCacheAlias.create(model_dir, "alice")

    assert (snapshot / "model_index.json").is_symlink()
    assert not list(snapshot.parent.parent.glob(".mflux_*-*"))


@pytest.mark.fast
def test_next_run_restores_a_snapshot_left_between_the_renames(hf_cache, tmp_path):
    model_dir = make_saved_model(tmp_path)
    first = HfCacheAlias.create(model_dir, "alice")
    repo_cache = hf_cache / "models--alice--my-model_8bit"
    # A crash after the first rename leaves the old snapshot under its temporary name.
    (repo_cache / "snapshots" / first.revision).rename(repo_cache / f".mflux_replaced-{first.revision}-x")

    result = HfCacheAlias.create(model_dir, "alice")

    assert result.file_count == 2
    assert not list(repo_cache.glob(".mflux_*-*"))


@pytest.mark.fast
def test_snapshot_folder_has_the_umask_mode(hf_cache, tmp_path):
    previous = os.umask(0o022)
    try:
        result = HfCacheAlias.create(make_saved_model(tmp_path), "alice")
    finally:
        os.umask(previous)

    snapshot = result.repo_cache / "snapshots" / result.revision
    assert snapshot.stat().st_mode & 0o777 == 0o755
