import json
import struct

import pytest

from mflux.models.common.checking.model_checker import ModelChecker
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition


def write_safetensors(path, metadata=None, names=None):
    header = {"__metadata__": metadata or {}}
    for name in names or ["weight"]:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0\0\0")


def write_index(path, shard_names):
    path.write_text(json.dumps({"weight_map": {f"weight_{index}": name for index, name in enumerate(shard_names)}}))


@pytest.fixture
def qwen_definition(monkeypatch):
    class Definition:
        @staticmethod
        def get_components():
            return [ComponentDefinition(name="transformer", hf_subdir="transformer")]

    monkeypatch.setattr(ModelChecker, "_definition_for", staticmethod(lambda key: Definition))
    monkeypatch.setattr(ModelChecker, "_resolve_model_key", staticmethod(lambda reference: "qwen-image"))


@pytest.mark.fast
def test_mflux_package_passes_when_all_parts_are_present(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", {"mflux_version": "0.21.0", "quantization_level": "6"})
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "mflux"
    assert result.size_bytes > 0


@pytest.mark.fast
def test_supported_hugging_face_safetensors_package_passes(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image"
    write_safetensors(root / "transformer" / "model.safetensors")

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "safetensors"


@pytest.mark.fast
def test_incomplete_mflux_package_warns_and_prints_the_fix_command(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"})
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors", "1.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "warn"
    assert "1.safetensors" in " ".join(result.problems)
    assert f"mflux-check model --fix {root}" in ModelChecker.format_result(result, str(root))


@pytest.mark.fast
def test_cached_hugging_face_reference_never_downloads(tmp_path, qwen_definition, monkeypatch):
    snapshot = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots" / "abc123"
    write_safetensors(snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"})
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))

    result = ModelChecker.check("mflux-community/qwen-image-mflux-q6")

    assert result.check == "pass"
    assert result.format == "mflux"


@pytest.mark.fast
def test_repair_restores_a_missing_cached_shard_from_another_snapshot(tmp_path, qwen_definition, monkeypatch):
    cache = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots"
    old_snapshot = cache / "abc123"
    current_snapshot = cache / "def456"
    write_safetensors(old_snapshot / "transformer" / "1.safetensors", {"mflux_version": "0.21.0"})
    write_safetensors(current_snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"})
    write_index(current_snapshot / "transformer" / "model.safetensors.index.json", ["0.safetensors", "1.safetensors"])
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))

    assert ModelChecker.check("mflux-community/qwen-image-mflux-q6").check == "warn"
    assert "Restored 1 shard" in ModelChecker.repair("mflux-community/qwen-image-mflux-q6")
    assert ModelChecker.check("mflux-community/qwen-image-mflux-q6").check == "pass"


@pytest.mark.fast
def test_unrecognized_hugging_face_safetensors_package_passes_structural_check(tmp_path, monkeypatch):
    root = tmp_path / "flux-3-image-klein"
    write_safetensors(root / "transformer" / "model.safetensors")
    monkeypatch.setattr(ModelChecker, "_resolve_model_key", staticmethod(lambda reference: None))

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "safetensors"
    assert result.problems == ()


@pytest.mark.known_model
@pytest.mark.parametrize(
    ("reference", "format_name"),
    [
        ("black-forest-labs/flux-3-image-klein", "safetensors"),
        ("/Volumes/Extreme2Tb/MFlux-Models/Qwen--Qwen-Image-2.1-Turbo-mflux-q8", "mflux"),
    ],
)
def test_known_good_model_passes(reference, format_name):
    result = ModelChecker.check(reference)

    assert result.check == "pass"
    assert result.format == format_name
