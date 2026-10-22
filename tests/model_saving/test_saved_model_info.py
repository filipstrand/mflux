# `mflux-save` without --path must pick a unique folder in $MFLUX_MODELS_DIR.
# Every save must write mflux_model.json. The file names the upstream source, the precision
# and the LoRAs, and holds no absolute paths. No test here loads weights or calls the Hub.

import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mflux.models.common.cli import save
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.weights.saving import saved_model_info
from mflux.models.common.weights.saving.hf_cache_alias import HfCacheAlias
from mflux.models.common.weights.saving.saved_model_info import SavedModelInfo, SaveTarget

ENV = SaveTarget.MODELS_DIR_ENV_VAR


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(saved_model_info.hf_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(saved_model_info.hf_constants, "HF_HUB_OFFLINE", False)
    # 1 "GB" = 1000 bytes, so the tiny test files give sizes that are not rounded to 0.
    monkeypatch.setattr(saved_model_info, "GB", 1000)
    for env_var in (ENV, *HfCacheAlias.USERNAME_ENV_VARS):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "osuser")
    # No test may reach the Hub. A test that wants a date patches this again.
    monkeypatch.setattr(saved_model_info.HfApi, "model_info", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    return cache


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    monkeypatch.setenv(ENV, str(folder))
    return folder


def make_snapshot(cache: Path, repo_id: str, revision: str, files: dict[str, bytes]) -> Path:
    # The real cache layout: snapshot entries are symlinks into blobs/.
    repo_cache = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / revision
    (repo_cache / "blobs").mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        blob = repo_cache / "blobs" / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        (snapshot / relative).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / relative).symlink_to(blob)
    return snapshot


def make_saved(target: Path, subdirs=("transformer", "text_encoder", "vae")) -> Path:
    for subdir in subdirs:
        (target / subdir).mkdir(parents=True, exist_ok=True)
        (target / subdir / "0.safetensors").write_bytes(b"x" * 1000)
    return target


def write_info(target, key="z-image-turbo", model_path=None, quantize=8, loras=(), scales=()):
    return SavedModelInfo.write(
        target=target,
        key=key,
        model_path=model_path,
        model_config_repo_id=None if model_path else AVAILABLE_MODELS[key].model_name,
        quantize=quantize,
        lora_sources=[str(s) for s in loras],
        lora_files=[str(s) for s in loras],
        lora_scales=list(scales),
    )


# --- SaveTarget -------------------------------------------------------------------------------


@pytest.mark.fast
def test_path_wins_over_models_dir(hf_cache, models_dir, tmp_path):
    assert SaveTarget.resolve(str(tmp_path / "x"), "dev", None, 8, False) == (tmp_path / "x").resolve()


@pytest.mark.fast
@pytest.mark.parametrize("path_arg", [None, ""])
def test_no_path_and_no_models_dir_names_both_options(hf_cache, path_arg):
    with pytest.raises(ValueError, match=r"Pass --path, or set \$MFLUX_MODELS_DIR"):
        SaveTarget.resolve(path_arg, "dev", None, 8, False)


@pytest.mark.fast
def test_empty_models_dir_counts_as_unset(hf_cache, monkeypatch):
    monkeypatch.setenv(ENV, "")
    with pytest.raises(ValueError, match="Pass --path"):
        SaveTarget.resolve(None, "dev", None, 8, False)


@pytest.mark.fast
@pytest.mark.parametrize("kind", ["missing", "file"])
def test_models_dir_must_be_a_folder(hf_cache, monkeypatch, tmp_path, kind):
    value = tmp_path / "not-there"
    if kind == "file":
        value.write_text("")
    monkeypatch.setenv(ENV, str(value))
    with pytest.raises(ValueError, match="is not a folder"):
        SaveTarget.resolve(None, "dev", None, 8, False)


@pytest.mark.fast
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any folder")
def test_models_dir_must_be_writable(hf_cache, models_dir):
    models_dir.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="is not writable"):
            SaveTarget.resolve(None, "dev", None, 8, False)
    finally:
        models_dir.chmod(0o700)


@pytest.mark.fast
def test_models_dir_expands_home(hf_cache, monkeypatch, tmp_path):
    (tmp_path / "models").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV, "~/models")
    assert SaveTarget.resolve(None, "dev", None, 8, False).parent == (tmp_path / "models").resolve()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("key", "quantize", "expected"),
    [
        ("flux3-image-klein-base", 8, "flux-3-image-klein-base-mflux-q8"),
        ("dev", None, "FLUX.1-dev-mflux-bf16"),
        ("dev-controlnet-canny", 4, "FLUX.1-dev-controlnet-canny-mflux-q4"),
        ("seedvr2-3b", 8, "SeedVR2_comfyUI-3b-mflux-q8"),
        ("seedvr2-7b", 8, "SeedVR2_comfyUI-7b-mflux-q8"),
        ("z-image-turbo", 8, "Z-Image-Turbo-mflux-q8"),
        ("ideogram-4-fp8", None, "ideogram-4-fp8-mflux-fp8"),
        ("ideogram-4-fp8", 8, "ideogram-4-fp8-mflux-q8"),
    ],
)
def test_auto_name_for_builtin_models(hf_cache, models_dir, key, quantize, expected):
    assert SaveTarget.resolve(None, key, None, quantize, False) == models_dir.resolve() / expected


@pytest.mark.fast
def test_every_saveable_model_gets_its_own_auto_name(hf_cache):
    # Drift guard: several keys share one repo (FLUX.1-dev, SeedVR2_comfyUI, Qwen-Image-2.1, ...).
    names = {key: SaveTarget.base_name(key, None) for key in save.MODEL_CLASSES}
    assert len(set(names.values())) == len(names), names


@pytest.mark.fast
def test_auto_name_clash_fails(hf_cache, models_dir):
    (models_dir / "Z-Image-Turbo-mflux-q8").mkdir()
    with pytest.raises(ValueError, match="exists"):
        SaveTarget.resolve(None, "z-image-turbo", None, 8, False)


@pytest.mark.fast
def test_lora_saves_count_up_from_the_highest_number(hf_cache, models_dir):
    assert SaveTarget.resolve(None, "z-image-turbo", None, 8, True).name == "Z-Image-Turbo-mflux-q8-lora01"
    (models_dir / "Z-Image-Turbo-mflux-q8-lora01").mkdir()
    (models_dir / "Z-Image-Turbo-mflux-q8-lora03").mkdir()
    assert SaveTarget.resolve(None, "z-image-turbo", None, 8, True).name == "Z-Image-Turbo-mflux-q8-lora04"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("models--krea--Krea-2-Turbo", "Krea-2-Turbo"),
        ("models--krea--Krea-2-Turbo/snapshots/abc123", "Krea-2-Turbo"),
        ("z-image-turbo-MFlux-q8", "z-image-turbo"),
        ("Krea-2-Turbo-MFlux-bf16", "Krea-2-Turbo"),
        ("black-forest-labs--flux-3-image-klein-mflux-q8", "flux-3-image-klein"),
        ("Qwen-Image-2.1-mflux-q8-lora02", "Qwen-Image-2.1"),
        ("my-finetune", "my-finetune"),
    ],
)
def test_auto_name_for_a_local_model(hf_cache, tmp_path, folder, expected):
    local = tmp_path / "src" / folder
    local.mkdir(parents=True)
    assert SaveTarget.base_name("krea-2", str(local)) == expected


@pytest.mark.fast
def test_auto_name_for_a_third_party_repo(hf_cache):
    assert SaveTarget.base_name("dev", "someone/My-Finetune") == "My-Finetune"


# --- SavedModelInfo ---------------------------------------------------------------------------


@pytest.mark.fast
def test_info_for_a_builtin_model(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: "rev1")
    make_snapshot(
        hf_cache,
        "Tongyi-MAI/Z-Image-Turbo",
        "rev1",
        {
            "transformer/a.safetensors": b"t" * 3000,
            "text_encoder/config.json": json.dumps({"model_type": "qwen3"}).encode(),
            "vae/a.safetensors": b"v" * 1000,
        },
    )
    target = make_saved(tmp_path / "out")

    info = write_info(target)

    assert info["model_name"] == "z-image-turbo"
    assert info["base_model"] is None
    assert info["quantize"] == 8
    assert info["size_gb"] == 3.0
    assert info["upstream"]["repo_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert info["upstream"]["revision"] == "rev1"
    assert info["upstream"]["revision_date"] is None
    assert info["text_encoders"] == ["qwen3"]
    assert info["loras"] == []
    assert json.loads((target / SavedModelInfo.FILE_NAME).read_text()) == info
    assert not list(target.glob(".*.tmp"))


@pytest.mark.fast
def test_upstream_size_counts_each_blob_once(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: "rev1")
    # Two snapshot entries with the same content point at one blob.
    make_snapshot(hf_cache, "Tongyi-MAI/Z-Image-Turbo", "rev1", {"a/x": b"1" * 4000, "b/x": b"1" * 4000})

    info = write_info(make_saved(tmp_path / "out"))

    assert info["upstream"]["size_gb"] == 4.0


@pytest.mark.fast
def test_newest_snapshot_without_a_pin(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: None)
    old = make_snapshot(hf_cache, "Tongyi-MAI/Z-Image-Turbo", "old", {"x": b"1"})
    new = make_snapshot(hf_cache, "Tongyi-MAI/Z-Image-Turbo", "new", {"x": b"2"})
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert write_info(make_saved(tmp_path / "out"))["upstream"]["revision"] == "new"


@pytest.mark.fast
def test_flux_lists_both_text_encoders(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: "rev1")
    make_snapshot(
        hf_cache,
        "black-forest-labs/FLUX.1-dev",
        "rev1",
        {
            "text_encoder/config.json": json.dumps({"model_type": "clip_text_model"}).encode(),
            "text_encoder_2/config.json": json.dumps({"model_type": "t5"}).encode(),
        },
    )
    target = make_saved(tmp_path / "out", ("transformer", "text_encoder", "text_encoder_2", "vae"))

    assert write_info(target, key="dev")["text_encoders"] == ["clip_text_model", "t5"]


@pytest.mark.fast
def test_text_encoder_without_config_uses_the_folder_name(hf_cache, tmp_path):
    target = make_saved(tmp_path / "out", ("transformer", "mllm"))
    assert write_info(target, key="boogu-image-turbo")["text_encoders"] == ["mllm"]


@pytest.mark.fast
def test_model_without_a_text_encoder(hf_cache, tmp_path):
    assert write_info(make_saved(tmp_path / "out", ("transformer", "vae")), key="seedvr2-3b")["text_encoders"] == []


@pytest.mark.fast
def test_info_has_no_absolute_paths(hf_cache, tmp_path):
    local_model = tmp_path / "private" / "my-finetune"
    make_saved(local_model)
    lora = tmp_path / "private" / "style.safetensors"
    lora.write_bytes(b"lora")
    target = make_saved(tmp_path / "out")

    info = write_info(target, key="z-image-turbo", model_path=str(local_model), loras=[lora], scales=[0.7])

    assert info["model_name"] == "my-finetune"
    assert info["base_model"] == "z-image-turbo"
    assert info["upstream"]["repo_id"] is None
    assert info["loras"] == [
        {"source": "style.safetensors", "scale": 0.7, "sha256": hashlib.sha256(b"lora").hexdigest()}
    ]
    assert str(tmp_path) not in (target / SavedModelInfo.FILE_NAME).read_text()


@pytest.mark.fast
def test_local_model_inside_an_hf_cache_finds_its_repo(hf_cache, tmp_path):
    snapshot = make_snapshot(tmp_path / "other-cache", "krea/Krea-2-Turbo", "abc", {"transformer/x": b"1"})

    info = write_info(make_saved(tmp_path / "out"), key="krea-2", model_path=str(snapshot))

    assert info["upstream"]["repo_id"] == "krea/Krea-2-Turbo"
    assert info["upstream"]["revision"] == "abc"
    assert info["model_name"] == "krea/Krea-2-Turbo"


@pytest.mark.fast
def test_resave_keeps_created_at_for_the_same_model_only(hf_cache, tmp_path):
    target = make_saved(tmp_path / "out")
    (target / SavedModelInfo.FILE_NAME).write_text(
        json.dumps({"model_name": "z-image-turbo", "created_at": "2020-01-01T00:00:00Z"})
    )
    assert write_info(target)["created_at"] == "2020-01-01T00:00:00Z"

    (target / SavedModelInfo.FILE_NAME).write_text(
        json.dumps({"model_name": "dev", "created_at": "2020-01-01T00:00:00Z"})
    )
    assert write_info(target)["created_at"] != "2020-01-01T00:00:00Z"


@pytest.mark.fast
def test_revision_date_from_the_hub(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: "rev1")
    make_snapshot(hf_cache, "Tongyi-MAI/Z-Image-Turbo", "rev1", {"x": b"1"})
    calls = []

    def fake_model_info(self, repo_id, revision=None, timeout=None):
        calls.append((repo_id, revision, timeout))
        return SimpleNamespace(last_modified=datetime(2026, 9, 2, 14, 10, tzinfo=timezone.utc))

    monkeypatch.setattr(saved_model_info.HfApi, "model_info", fake_model_info)

    assert write_info(make_saved(tmp_path / "out"))["upstream"]["revision_date"] == "2026-09-02T14:10:00Z"
    assert calls == [("Tongyi-MAI/Z-Image-Turbo", "rev1", 5)]


@pytest.mark.fast
def test_offline_mode_makes_no_hub_call(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(saved_model_info.ModelRevisions, "get", lambda repo_id: "rev1")
    monkeypatch.setattr(saved_model_info.hf_constants, "HF_HUB_OFFLINE", True)
    make_snapshot(hf_cache, "Tongyi-MAI/Z-Image-Turbo", "rev1", {"x": b"1"})
    monkeypatch.setattr(saved_model_info.HfApi, "model_info", lambda *a, **k: pytest.fail("Hub call in offline mode"))

    assert write_info(make_saved(tmp_path / "out"))["upstream"]["revision_date"] is None


# --- CLI --------------------------------------------------------------------------------------


def run_save(argv_extra: list[str], model: str = "z-image-turbo", base_model: str | None = None) -> list[str]:
    # Stubs the load and the save. Returns the folders that the stubbed save wrote.
    saved = []
    model_args = ["--model", model, *(["--base-model", base_model] if base_model else [])]

    def fake_save(self, base_path):
        saved.append(base_path)
        make_saved(Path(base_path))

    model_class = save.MODEL_CLASSES["z-image-turbo"]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", ["mflux-save", *model_args, *argv_extra]))
        # The CLI checks the constructor signature for LoRA support, so the stub keeps those names.
        stack.enter_context(
            patch.object(model_class, "__init__", lambda self, lora_paths=None, lora_scales=None, **kwargs: None)
        )
        stack.enter_context(patch.object(model_class, "save_model", fake_save))
        save.main()
    return saved


@pytest.mark.fast
def test_cli_saves_to_the_auto_folder(hf_cache, models_dir, capsys):
    saved = run_save(["-q", "8"])

    target = models_dir.resolve() / "Z-Image-Turbo-mflux-q8"
    assert saved == [str(target)]
    assert (target / SavedModelInfo.FILE_NAME).is_file()
    assert f"Saved model: {target}" in capsys.readouterr().out


@pytest.mark.fast
def test_cli_without_path_or_models_dir_exits_before_the_save(hf_cache):
    with pytest.raises(SystemExit) as exit_info:
        run_save([])
    assert exit_info.value.code == 2


@pytest.mark.fast
def test_cli_auto_name_clash_exits_before_the_save(hf_cache, models_dir):
    (models_dir / "Z-Image-Turbo-mflux-q8").mkdir()
    with pytest.raises(SystemExit) as exit_info:
        run_save(["-q", "8"])
    assert exit_info.value.code == 2


@pytest.mark.fast
def test_cli_alias_uses_the_auto_name_and_links_the_info_file(hf_cache, models_dir, monkeypatch):
    monkeypatch.setattr(
        HfCacheAlias,
        "_repo_cache",
        staticmethod(lambda model_dir, username: hf_cache / f"models--{username}--{model_dir.name}"),
    )

    run_save(["-q", "8", "--hf-symlink", "bob"])

    snapshots = list((hf_cache / "models--bob--Z-Image-Turbo-mflux-q8" / "snapshots").iterdir())
    assert (snapshots[0] / SavedModelInfo.FILE_NAME).is_symlink()


@pytest.mark.fast
def test_cli_lora_save_gets_a_numbered_folder_and_records_the_lora(hf_cache, models_dir, tmp_path):
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"lora")

    saved = run_save(["-q", "8", "--lora", str(lora), "0.7"])

    target = Path(saved[0])
    assert target.name == "Z-Image-Turbo-mflux-q8-lora01"
    info = json.loads((target / SavedModelInfo.FILE_NAME).read_text())
    assert info["loras"] == [
        {"source": "style.safetensors", "scale": 0.7, "sha256": hashlib.sha256(b"lora").hexdigest()}
    ]


@pytest.mark.fast
@pytest.mark.parametrize(("requested", "expected"), [(None, "q8"), (4, "q8")])
def test_resave_of_a_quantized_save_keeps_its_stored_width(hf_cache, models_dir, tmp_path, requested, expected):
    # The loader keeps the stored width and ignores -q, so the name must not say bf16 or q4.
    earlier = make_saved(tmp_path / "Z-Image-Turbo-mflux-q8")
    (earlier / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"quantization_level": "8"}})
    )
    quantize = save.QuantizationResolution.resolve(
        stored=SaveTarget.stored_quantization(str(earlier)), requested=requested
    )[0]

    target = SaveTarget.resolve(None, "z-image-turbo", str(earlier), quantize, True)

    assert target.name == f"Z-Image-Turbo-mflux-{expected}-lora01"


@pytest.mark.fast
def test_unquantized_save_has_no_stored_width(hf_cache, tmp_path):
    earlier = make_saved(tmp_path / "Z-Image-Turbo-mflux-bf16")
    (earlier / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"quantization_level": "None"}})
    )
    assert SaveTarget.stored_quantization(str(earlier)) is None
    assert SaveTarget.stored_quantization("Tongyi-MAI/Z-Image-Turbo") is None


def make_cached_q5_alias(hf_cache: Path) -> str:
    # A q5 save that --hf-symlink linked into the cache, loaded by its repo id.
    repo_id = "bob/Z-Image-Turbo-mflux-q5"
    files = {f"{d}/0.safetensors": b"x" * 1000 for d in ("transformer", "text_encoder", "vae")}
    files["transformer/model.safetensors.index.json"] = json.dumps({"metadata": {"quantization_level": "5"}}).encode()
    make_snapshot(hf_cache, repo_id, "abc123", files)
    return repo_id


@pytest.mark.fast
@pytest.mark.parametrize("level", ["q8", [8], {"bits": 8}])
def test_malformed_stored_width_is_skipped(hf_cache, tmp_path, level):
    earlier = make_saved(tmp_path / "Z-Image-Turbo-mflux-q8")
    (earlier / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"quantization_level": level}})
    )
    assert SaveTarget.stored_quantization(str(earlier)) is None


@pytest.mark.fast
def test_cached_repo_alias_has_its_stored_width(hf_cache):
    assert SaveTarget.stored_quantization(make_cached_q5_alias(hf_cache)) == 5


@pytest.mark.fast
def test_cli_resave_of_a_cached_repo_alias_keeps_its_stored_width(hf_cache, models_dir):
    # #912: the loader keeps q5, so the auto name and the metadata must not say q8.
    repo_id = make_cached_q5_alias(hf_cache)

    saved = run_save(["-q", "8"], model=repo_id, base_model="z-image-turbo")

    target = Path(saved[0])
    assert target.name == "Z-Image-Turbo-mflux-q5"
    assert json.loads((target / SavedModelInfo.FILE_NAME).read_text())["quantize"] == 5


@pytest.mark.fast
def test_cli_explicit_path_records_the_stored_width_of_a_cached_repo_alias(hf_cache, tmp_path):
    repo_id = make_cached_q5_alias(hf_cache)

    saved = run_save(["-q", "8", "--path", str(tmp_path / "out")], model=repo_id, base_model="z-image-turbo")

    assert json.loads((Path(saved[0]) / SavedModelInfo.FILE_NAME).read_text())["quantize"] == 5
