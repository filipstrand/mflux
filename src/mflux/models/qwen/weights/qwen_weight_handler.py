import json
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten
from safetensors.mlx import load_file as mlx_load_file
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.flux.weights.weight_handler import MetaData, WeightHandler
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        qwen_text_encoder: dict | None = None,
        transformer: dict | None = None,
        vae: dict | None = None,
    ):
        self.qwen_text_encoder = qwen_text_encoder
        self.transformer = transformer
        self.vae = vae
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "QwenWeightHandler":
        # Load the weights from disk, huggingface cache, or download from huggingface
        root_path = Path(local_path) if local_path else WeightHandler.download_or_get_cached_weights(repo_id)

        # Determine if we should load visual weights (for Edit model)
        load_visual_weights = repo_id and "edit" in repo_id.lower()

        # Load the weights
        transformer, quantization_level, mflux_version = QwenWeightHandler._load_transformer(root_path=root_path)
        qwen_text_encoder, _, _ = QwenWeightHandler._load_qwen_text_encoder(root_path=root_path, load_visual_weights=load_visual_weights)  # fmt: off
        vae, _, _ = QwenWeightHandler._load_vae(root_path=root_path)

        return QwenWeightHandler(
            qwen_text_encoder=qwen_text_encoder,
            transformer=transformer,
            vae=vae,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_transformer(root_path: Path) -> tuple[dict, int | None, str | None]:
        return QwenWeightHandler._load_component(
            root_path=root_path,
            component_name="transformer",
            loading_mode="multi_glob",
            mapping_getter=QwenWeightMapping.get_transformer_mapping,
        )

    @staticmethod
    def _load_qwen_text_encoder(
        root_path: Path, load_visual_weights: bool = False
    ) -> tuple[dict, int | None, str | None]:
        return QwenWeightHandler._load_component(
            root_path=root_path,
            component_name="text_encoder",
            loading_mode="multi_json",
            mapping_getter=QwenWeightMapping.get_text_encoder_mapping,
        )

    @staticmethod
    def _load_vae(root_path: Path) -> tuple[dict, int | None, str | None]:
        return QwenWeightHandler._load_component(
            root_path=root_path,
            component_name="vae",
            loading_mode="single",
            mapping_getter=QwenWeightMapping.get_vae_mapping,
        )

    @staticmethod
    def _load_component(
        root_path: Path,
        component_name: str,
        loading_mode: str,
        mapping_getter: Callable[[], list[WeightTarget]],
    ) -> tuple[dict, int | None, str | None]:
        component_path = root_path / component_name
        weights = QwenWeightHandler._load_safetensors_shards(component_path, loading_mode=loading_mode)

        # Check if this is a saved model (has metadata)
        quantization_level, mflux_version = QwenWeightHandler._detect_metadata(component_path)
        if quantization_level is not None or mflux_version is not None:
            return QwenWeightHandler._load_saved_model_weights(weights, None, quantization_level, mflux_version)

        # Otherwise, it's HuggingFace weights that need mapping
        mapping = mapping_getter()
        mapped_weights = WeightMapper.apply_mapping(weights, mapping)
        return mapped_weights, None, None

    @staticmethod
    def _load_safetensors_shards(path: Path, loading_mode: str = "multi_glob") -> dict[str, mx.array]:
        all_weights = {}

        if loading_mode == "single":
            # VAE style: Single file loading
            safetensors_files = list(path.glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

            weights_file = safetensors_files[0]
            data = mx.load(str(weights_file), return_metadata=True)
            all_weights = dict(data[0].items())

        elif loading_mode == "multi_json":
            # Text encoder style: Use JSON index to map params to files
            index_path = path / "model.safetensors.index.json"
            with open(index_path) as f:
                index = json.load(f)

            # Group weights by file
            files_to_load = {}
            for param_name, file_name in index["weight_map"].items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(param_name)

            # Load weights from each file
            for file_name, param_names in files_to_load.items():
                file_path = path / file_name

                # Load the safetensor file with fallback to torch conversion
                try:
                    file_weights = mlx_load_file(str(file_path))
                except Exception:  # noqa: BLE001
                    # If MLX can't load directly, try with torch and convert
                    torch_weights = torch_load_file(str(file_path))
                    file_weights = {}
                    for name, tensor in torch_weights.items():
                        # Convert to float32 if bfloat16, then to MLX
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        file_weights[name] = mx.array(tensor.numpy())

                # Add requested parameters to combined weights
                for param_name in param_names:
                    if param_name in file_weights:
                        all_weights[param_name] = file_weights[param_name]
        else:  # "multi_glob"
            # Transformer style: Directly glob all safetensors files
            shard_files = sorted([f for f in path.glob("*.safetensors") if not f.name.startswith("._")])
            if not shard_files:
                raise FileNotFoundError(f"No safetensors found in {path}")

            for shard in shard_files:
                data, metadata = mx.load(str(shard), return_metadata=True)
                all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _load_saved_model_weights(
        weights: dict[str, mx.array] | None,
        path: Path | None,
        quantization_level: int | None,
        mflux_version: str | None,
    ) -> tuple[dict, int | None, str | None]:
        # If weights already loaded, use them; otherwise load from path
        if weights is None:
            # For saved models, always use multi_glob (no index.json needed)
            weights = QwenWeightHandler._load_safetensors_shards(path, loading_mode="multi_glob")

        return tree_unflatten(list(weights.items())), quantization_level, mflux_version

    @staticmethod
    def _detect_metadata(path: Path) -> tuple[int | None, str | None]:
        file_glob = sorted(path.glob("*.safetensors"))
        if file_glob:
            data = mx.load(str(file_glob[0]), return_metadata=True)
            if len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")
                return quantization_level, mflux_version
        return None, None
