import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from safetensors.mlx import load_file as mlx_load_file
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.weight_definition import ComponentDefinition

if TYPE_CHECKING:
    from mflux.models.common.weights.weight_definition import WeightDefinitionType


class WeightLoader:
    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> LoadedWeights:
        # 1. Get root path
        root_path = WeightLoader._get_root_path(
            repo_id=repo_id,
            local_path=local_path,
            patterns=weight_definition.get_download_patterns(),
        )

        # 2. Load each component
        components = {}
        quantization_level = None
        mflux_version = None

        for component in weight_definition.get_components():
            weights, q_level, version = WeightLoader._load_component(root_path, component)
            components[component.name] = weights

            # Track metadata from first component that has it
            if quantization_level is None and q_level is not None:
                quantization_level = q_level
                mflux_version = version

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _get_root_path(
        repo_id: str | None,
        local_path: str | None,
        patterns: list[str],
    ) -> Path:
        if local_path:
            return Path(local_path)
        if repo_id:
            return Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
        raise ValueError("Either repo_id or local_path must be provided")

    @staticmethod
    def _load_component(
        root_path: Path,
        component: ComponentDefinition,
    ) -> tuple[dict, int | None, str | None]:
        component_path = root_path / component.hf_subdir

        # Try mflux saved format first
        weights, q_level, version = WeightLoader._try_load_mflux_format(component_path)
        if weights is not None:
            return weights, q_level, version

        # Fall back to HuggingFace format with mapping
        raw_weights = WeightLoader._load_safetensors(component_path, component.loading_mode)

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        if not path.exists():
            return None, None, None

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        # Check metadata on first file
        data = mx.load(str(shard_files[0]), return_metadata=True)
        if len(data) <= 1:
            return None, None, None

        quantization_level = data[1].get("quantization_level")
        mflux_version = data[1].get("mflux_version")

        # If no mflux metadata, this isn't our format
        if quantization_level is None and mflux_version is None:
            return None, None, None

        # Load all shards
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            shard_data = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(shard_data[0].items()))

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name

            # Try MLX native first, fall back to torch conversion
            try:
                file_weights = mlx_load_file(str(file_path))
            except Exception:  # noqa: BLE001
                torch_weights = torch_load_file(str(file_path))
                file_weights = {}
                for name, tensor in torch_weights.items():
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    file_weights[name] = mx.array(tensor.numpy())

            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v.astype(precision) for k, v in weights.items()}
