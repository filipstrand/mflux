import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.utils.exceptions import ModelConfigError


@dataclass(frozen=True)
class ModelCheckResult:
    check: str
    format: str | None
    size_bytes: int
    model: str | None
    problems: tuple[str, ...]


class ModelChecker:
    @staticmethod
    def check(reference: str) -> ModelCheckResult:
        root, resolution_problem = ModelChecker._resolve(reference)
        if root is None:
            return ModelCheckResult("fail", None, 0, None, (resolution_problem,))

        files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
        if not files:
            return ModelCheckResult("fail", None, 0, None, (f"No safetensors files exist under {root}.",))

        headers: dict[Path, dict[str, Any]] = {}
        problems: list[str] = []
        for path in files:
            header, problem = ModelChecker._read_header_or_problem(path, root)
            if header is not None:
                headers[path] = header
            if problem is not None:
                problems.append(problem)

        format_name = (
            "mflux" if any(ModelChecker._is_mflux_header(header) for header in headers.values()) else "safetensors"
        )
        key = ModelChecker._resolve_model_key(reference)
        if key is None:
            if format_name == "safetensors":
                ModelChecker._validate_indexes(root, problems)
                return ModelCheckResult(
                    "fail" if problems else "pass", format_name, ModelChecker._size(files), None, tuple(problems)
                )
            problems.append("Cannot identify a model that an MFlux CLI supports.")
            return ModelCheckResult("fail", format_name, ModelChecker._size(files), None, tuple(problems))

        definition = ModelChecker._definition_for(key)
        if definition is None:
            problems.append(f"MFlux has no checker definition for {key}.")
            return ModelCheckResult("fail", format_name, ModelChecker._size(files), key, tuple(problems))

        missing_parts = ModelChecker._validate_components(root, definition, format_name, headers, problems)
        missing_parts = ModelChecker._validate_indexes(root, problems) or missing_parts

        if format_name == "mflux" and missing_parts:
            return ModelCheckResult("warn", format_name, ModelChecker._size(files), key, tuple(problems))
        return ModelCheckResult(
            "fail" if problems else "pass", format_name, ModelChecker._size(files), key, tuple(problems)
        )

    @staticmethod
    def format_result(result: ModelCheckResult, reference: str, fix: bool = False, repair: str | None = None) -> str:
        lines = ["#################", "MFlux Check Model:", f" check: {result.check}"]
        if result.format is not None:
            lines.append(f" format: {result.format}")
        lines.append(f" size: {result.size_bytes / 1_000_000_000:.2f} GB")
        if result.model is not None and result.check != "pass":
            lines.append(f" model: {result.model}")
        lines.extend(f" problem: {problem}" for problem in result.problems)
        if result.check == "warn":
            lines.append(f" fix: mflux-check model --fix {reference}")
            if fix:
                lines.append(
                    f" repair: {repair or 'No local replacement is available. The check did not download files.'}"
                )
        return "\n".join(lines)

    @staticmethod
    def repair(reference: str) -> str:
        root, problem = ModelChecker._resolve(reference)
        if root is None:
            return problem
        snapshots = root.parent
        if snapshots.name != "snapshots":
            return "No local replacement is available. The check did not download files."

        repaired = 0
        for index_path in root.rglob("*.safetensors.index.json"):
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                names = set(weight_map.values()) if isinstance(weight_map, dict) else set()
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            relative_dir = index_path.parent.relative_to(root)
            for name in names:
                target = index_path.parent / name
                if not isinstance(name, str) or target.is_file():
                    continue
                source = next(
                    (
                        snapshot / relative_dir / name
                        for snapshot in snapshots.iterdir()
                        if snapshot != root and (snapshot / relative_dir / name).is_file()
                    ),
                    None,
                )
                if source is not None:
                    shutil.copy2(source, target)
                    repaired += 1
        if repaired:
            return f"Restored {repaired} shard(s) from another local cache snapshot."
        return "No local replacement is available. The check did not download files."

    @staticmethod
    def _resolve(reference: str) -> tuple[Path | None, str]:
        local = Path(reference).expanduser()
        if local.exists():
            return local, ""
        if not PathResolution._is_hf_format(reference):
            return None, f"Model path does not exist: {reference}."

        snapshots = Path(HF_HUB_CACHE) / f"models--{reference.replace('/', '--')}" / "snapshots"
        if not snapshots.is_dir():
            return None, f"Model is not in the Hugging Face cache: {reference}."
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime
        )
        if not candidates:
            return None, f"Model has no cached snapshots: {reference}."
        return candidates[-1], ""

    @staticmethod
    def _resolve_model_key(reference: str) -> str | None:
        try:
            return ConfigResolution.resolve_key(reference)
        except ModelConfigError:
            return None

    @staticmethod
    def _definition_for(key: str) -> type | None:
        if key in {"dev-redux"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxReduxSaveDefinition

            return FluxReduxSaveDefinition
        if key in {"dev-controlnet-canny", "dev-controlnet-upscaler", "schnell-controlnet-canny"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition

            return FluxControlnetWeightDefinition
        if key in {"dev", "schnell", "dev-depth", "dev-fill", "dev-fill-catvton", "dev-kontext", "krea-dev"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

            return FluxWeightDefinition
        if key.startswith("flux2-"):
            from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

            return Flux2KleinWeightDefinition
        if key in {"qwen-image", "qwen-image-edit", "qwen-image-edit-2511"}:
            from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

            return QwenWeightDefinition
        if key == "qwen-image-2.1-controlnet-union":
            from mflux.models.qwen21.weights.qwen_image21_controlnet_weight_definition import (
                QwenImage21ControlnetWeightDefinition,
            )

            return QwenImage21ControlnetWeightDefinition
        if key == "qwen-image-2.1":
            from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

            return Qwen21WeightDefinition
        if key in {"z-image", "z-image-turbo"}:
            from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

            return ZImageWeightDefinition
        if key == "z-image-turbo-controlnet-union-2.1":
            from mflux.models.z_image.weights.z_image_controlnet_weight_definition import (
                ZImageControlnetWeightDefinition,
            )

            return ZImageControlnetWeightDefinition
        if key in {"seedvr2-3b", "seedvr2-7b"}:
            from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
                SeedVR2WeightDefinition3B,
                SeedVR2WeightDefinition7B,
            )

            return SeedVR2WeightDefinition3B if key == "seedvr2-3b" else SeedVR2WeightDefinition7B
        if key in {"fibo", "fibo-lite", "fibo-edit", "fibo-edit-rmbg"}:
            from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition

            return FIBOWeightDefinition
        if key in {"ernie-image", "ernie-image-turbo"}:
            from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition

            return ErnieWeightDefinition
        if key == "boogu-image-turbo":
            from mflux.models.boogu.weights.boogu_weight_definition import BooguWeightDefinition

            return BooguWeightDefinition
        if key == "ideogram-4-fp8":
            from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition

            return Ideogram4WeightDefinition
        if key in {"krea-2", "krea-2-raw"}:
            from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

            return Krea2WeightDefinition
        if key == "ming-image-design":
            from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition

            return MingImageWeightDefinition
        return None

    @staticmethod
    def _validate_components(
        root: Path,
        definition: type,
        format_name: str,
        headers: dict[Path, dict[str, Any]],
        problems: list[str],
    ) -> bool:
        components = definition.get_components()
        save_subdirs = ComponentDefinition.save_subdirs(components)
        missing_parts = False
        for component in components:
            relative = save_subdirs[component.name] if format_name == "mflux" else component.hf_subdir
            directory = root / relative
            component_files = [path for path in headers if path.parent == directory]
            if not component_files:
                problems.append(f"Missing component: {relative or component.name}.")
                missing_parts = True
                continue
            if format_name == "safetensors" and component.mapping_getter is not None:
                keys = ModelChecker._component_keys(component, component_files, headers)
                missing = WeightMapper.missing_required_names(
                    keys, component.mapping_getter(), component.num_blocks, component.num_layers
                )
                if missing:
                    problems.append(f"Component {component.name} misses required weights: {', '.join(missing[:3])}.")
        return missing_parts

    @staticmethod
    def _component_keys(
        component: ComponentDefinition,
        files: list[Path],
        headers: dict[Path, dict[str, Any]],
    ) -> dict[str, None]:
        names = {}
        for path in files:
            for name in headers[path]:
                if name == "__metadata__":
                    continue
                if component.weight_prefix_filters is not None and not any(
                    name.startswith(prefix) for prefix in component.weight_prefix_filters
                ):
                    continue
                if component.key_transform is not None:
                    name = component.key_transform(name)
                    if name is None:
                        continue
                names[name] = None
        return names

    @staticmethod
    def _validate_indexes(root: Path, problems: list[str]) -> bool:
        missing_parts = False
        for index_path in root.rglob("*.safetensors.index.json"):
            missing, problem = ModelChecker._index_missing_parts(index_path, root)
            if problem is not None:
                problems.append(problem)
            if missing:
                missing_parts = True
        return missing_parts

    @staticmethod
    def _read_header(path: Path) -> dict[str, Any]:
        with path.open("rb") as file:
            raw_length = file.read(8)
            if len(raw_length) != 8:
                raise ValueError("file has no safetensors header")
            length = struct.unpack("<Q", raw_length)[0]
            if length > path.stat().st_size - 8:
                raise ValueError("header length exceeds file size")
            header = json.loads(file.read(length))
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        return header

    @staticmethod
    def _read_header_or_problem(path: Path, root: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return ModelChecker._read_header(path), None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return None, f"Cannot read safetensors header: {path.relative_to(root)} ({exc})."

    @staticmethod
    def _index_missing_parts(index_path: Path, root: Path) -> tuple[bool, str | None]:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is missing")
            names = set(weight_map.values())
            if not all(isinstance(name, str) and name and Path(name).name == name for name in names):
                raise ValueError("weight_map names a non-local shard")
            missing = [name for name in sorted(names) if not (index_path.parent / name).is_file()]
            if missing:
                return True, f"Index {index_path.relative_to(root)} misses shards: {', '.join(missing)}."
            return False, None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"Cannot read index {index_path.relative_to(root)} ({exc})."

    @staticmethod
    def _is_mflux_header(header: dict[str, Any]) -> bool:
        metadata = header.get("__metadata__")
        return isinstance(metadata, dict) and ("mflux_version" in metadata or "quantization_level" in metadata)

    @staticmethod
    def _size(files: list[Path]) -> int:
        seen = set()
        total = 0
        for path in files:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
        return total
