import json
from unittest.mock import patch

import pytest
from huggingface_hub.utils import filter_repo_objects

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.lens.weights.lens_weight_definition import LensWeightDefinition
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition3B

# An mflux-save of Krea 2 on HuggingFace, as mflux-community/krea-2-turbo-mflux-q4 and
# MLXBits/krea-2-mlx-q4 lay it out: the transformer's shards and index at the root (#869).
SAVED_KREA2_FILES = [
    "0.safetensors",
    "1.safetensors",
    "2.safetensors",
    "3.safetensors",
    "model.safetensors.index.json",
    "text_encoder/0.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "vae/0.safetensors",
    "vae/model.safetensors.index.json",
    "README.md",
]


def _saved_subdirs(definition) -> list[str]:
    return list(ComponentDefinition.save_subdirs(definition.get_components()).values())


def _download_patterns(definition, patterns, tmp_path) -> list[str]:
    with (
        patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)),
        patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download,
    ):
        download.return_value = str(tmp_path / "snapshot")
        PathResolution.resolve(path="someone/saved-model", patterns=patterns, saved_subdirs=_saved_subdirs(definition))
    return download.call_args.kwargs["allow_patterns"]


@pytest.mark.fast
@pytest.mark.parametrize("model_name", ["someone/krea-2-mflux-q4", "mflux-community/krea-2-raw-mflux-q8"])
def test_a_saved_krea2_downloads_its_transformer(tmp_path, model_name):
    patterns = _download_patterns(
        Krea2WeightDefinition, Krea2WeightDefinition.get_download_patterns(model_name), tmp_path
    )

    fetched = set(filter_repo_objects(SAVED_KREA2_FILES, allow_patterns=patterns))

    assert {f"{i}.safetensors" for i in range(4)} | {"model.safetensors.index.json"} <= fetched


@pytest.mark.fast
def test_the_official_krea2_turbo_download_is_unchanged(tmp_path):
    # The Turbo repo also carries the diffusers transformer/ shards (~26 GB) that the
    # native turbo.safetensors makes redundant. The saved-shard patterns must not reach them.
    official = [
        "turbo.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model-00001-of-00004.safetensors",
        "model_index.json",
    ]
    original = Krea2WeightDefinition.get_download_patterns("krea/Krea-2-Turbo")
    patterns = _download_patterns(Krea2WeightDefinition, original, tmp_path)

    assert set(filter_repo_objects(official, allow_patterns=patterns)) == set(
        filter_repo_objects(official, allow_patterns=original)
    )


@pytest.mark.fast
@pytest.mark.parametrize("definition", [FluxWeightDefinition, QwenWeightDefinition])
def test_patterns_that_already_cover_the_saved_layout_stay_as_they_are(tmp_path, definition):
    patterns = definition.get_download_patterns()

    assert _download_patterns(definition, patterns, tmp_path) == patterns


@pytest.mark.fast
def test_a_saved_seedvr2_downloads_its_component_subdirs(tmp_path):
    # mflux-save writes SeedVR2's two parts to transformer/ and vae/ (#823), while the
    # patterns name the original repo's flat files.
    patterns = _download_patterns(
        SeedVR2WeightDefinition3B, SeedVR2WeightDefinition3B.get_download_patterns(), tmp_path
    )
    saved = ["transformer/0.safetensors", "transformer/model.safetensors.index.json", "vae/0.safetensors"]

    assert set(filter_repo_objects(saved, allow_patterns=patterns)) == set(saved)


def _cached_saved_krea2(tmp_path, shards_on_disk: int, root_index: bool = True) -> object:
    snapshot = tmp_path / "models--someone--krea-2-mflux-q4" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for i in range(shards_on_disk):
        (snapshot / f"{i}.safetensors").write_bytes(b"")
    if root_index:
        index = {"weight_map": {f"blocks.{i}.weight": f"{i}.safetensors" for i in range(4)}}
        (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    for subdir in ("vae", "text_encoder"):
        (snapshot / subdir).mkdir()
        (snapshot / subdir / "0.safetensors").write_bytes(b"")
        index = {"weight_map": {"weight": "0.safetensors"}}
        (snapshot / subdir / "model.safetensors.index.json").write_text(json.dumps(index))
    (snapshot / "tokenizer").mkdir()
    (snapshot / "tokenizer" / "tokenizer.json").write_text("{}")
    return snapshot


@pytest.mark.fast
def test_a_cached_saved_krea2_loads_without_the_network(tmp_path):
    snapshot = _cached_saved_krea2(tmp_path, shards_on_disk=4)

    with (
        patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)),
        patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download,
    ):
        result = PathResolution.resolve(
            path="someone/krea-2-mflux-q4",
            patterns=Krea2WeightDefinition.get_download_patterns("someone/krea-2-mflux-q4"),
            saved_subdirs=_saved_subdirs(Krea2WeightDefinition),
        )

    assert result == snapshot
    download.assert_not_called()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("shards_on_disk", "root_index"),
    [(3, True), (1, False)],
    ids=["missing_a_shard", "cut_off_before_the_index"],
)
def test_a_partly_cached_saved_krea2_downloads_again(tmp_path, shards_on_disk, root_index):
    _cached_saved_krea2(tmp_path, shards_on_disk=shards_on_disk, root_index=root_index)

    with (
        patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)),
        patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download,
    ):
        download.return_value = str(tmp_path / "fresh")
        PathResolution.resolve(
            path="someone/krea-2-mflux-q4",
            patterns=Krea2WeightDefinition.get_download_patterns("someone/krea-2-mflux-q4"),
            saved_subdirs=_saved_subdirs(Krea2WeightDefinition),
        )

    download.assert_called_once()


@pytest.mark.fast
def test_a_cached_original_with_another_file_in_a_save_subdir_still_downloads(tmp_path):
    # Lens Turbo saves to diffusion_models/, where the original repo also keeps the base
    # model's file. A snapshot holding only that one isn't a save, and the turbo file still
    # has to come down.
    snapshot = tmp_path / "models--someone--lens" / "snapshots" / "abc123"
    (snapshot / "diffusion_models").mkdir(parents=True)
    (snapshot / "diffusion_models" / "lens_bf16.safetensors").write_bytes(b"")

    with (
        patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)),
        patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download,
    ):
        download.return_value = str(snapshot)
        PathResolution.resolve(
            path="someone/lens",
            patterns=LensWeightDefinition.get_download_patterns(),
            saved_subdirs=_saved_subdirs(LensWeightDefinition),
        )

    download.assert_called_once()


@pytest.mark.fast
def test_a_file_given_as_the_model_path_is_rejected(tmp_path):
    checkpoint = tmp_path / "krea2-fp8.safetensors"
    checkpoint.write_bytes(b"")

    with pytest.raises(ValueError, match="is a file"):
        WeightLoader.load(weight_definition=Krea2WeightDefinition, model_path=str(checkpoint))


@pytest.mark.fast
def test_a_root_with_no_krea2_transformer_names_the_layouts(tmp_path):
    with pytest.raises(FileNotFoundError, match="turbo.safetensors or raw.safetensors"):
        Krea2WeightDefinition._select_transformer_variant(tmp_path)


@pytest.mark.fast
def test_saved_shards_at_the_root_select_the_native_transformer(tmp_path):
    (tmp_path / "0.safetensors").write_bytes(b"")

    component = Krea2WeightDefinition._select_transformer_variant(tmp_path)

    assert component.hf_subdir == ""
