from pathlib import Path

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping
from mflux.models.flux.weights.weight_handler import (
    MetaData,
    WeightHandler as FluxWeightHandler,
)


class FIBOWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        vae: dict | None = None,
        transformer: dict | None = None,
        text_encoder: dict | None = None,
        decoder: dict | None = None,
        visual: dict | None = None,
        config: dict | None = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.visual = visual
        self.config = config
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        root_path: Path | None = None
        if local_path:
            root_path = Path(local_path)
        elif repo_id:
            root_path = FluxWeightHandler.download_or_get_cached_weights(repo_id)

        vae_weights = None
        transformer_weights = None
        text_encoder_weights = None
        quantization_level: int | None = None
        mflux_version: str | None = None

        if root_path is not None:
            vae_weights, _, _ = FIBOWeightHandler._try_load_saved_component(root_path, "vae")
            text_encoder_weights, _, _ = FIBOWeightHandler._try_load_saved_component(root_path, "text_encoder")  # fmt: off
            transformer_weights, quantization_level, mflux_version = FIBOWeightHandler._try_load_saved_component(root_path, "transformer")  # fmt: off

        if vae_weights is None or transformer_weights is None or text_encoder_weights is None:
            vae_weights = FIBOWeightHandler._load_vae_weights(repo_id, local_path)
            transformer_weights = FIBOWeightHandler._load_transformer_weights(repo_id, local_path)
            text_encoder_weights = FIBOWeightHandler._load_text_encoder_weights(repo_id, local_path)
            quantization_level = None
            mflux_version = None

        return FIBOWeightHandler(
            vae=vae_weights,
            transformer=transformer_weights,
            text_encoder=text_encoder_weights,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_vae_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        vae_path = root_path / "vae"
        raw_weights = FIBOWeightHandler._load_safetensors_shards(vae_path)
        mapping = FIBOWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)
        return mapped_weights

    @staticmethod
    def _load_transformer_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        transformer_path = root_path / "transformer"
        if transformer_path.exists() and list(transformer_path.glob("*.safetensors")):
            raw_weights = FIBOWeightHandler._load_safetensors_shards(transformer_path)
        else:
            raw_weights = FIBOWeightHandler._load_safetensors_shards(root_path)
        mapping = FIBOWeightMapping.get_transformer_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=38,
            num_layers=46,
        )
        return mapped_weights

    @staticmethod
    def _load_text_encoder_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        text_encoder_path = root_path / "text_encoder"
        raw_weights = FIBOWeightHandler._load_safetensors_shards(text_encoder_path)
        mapping = FIBOWeightMapping.get_text_encoder_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=36,
        )
        return mapped_weights

    @staticmethod
    def _get_root_path(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> Path:
        if local_path:
            return Path(local_path)
        return Path(FluxWeightHandler.download_or_get_cached_weights(repo_id or "briaai/FIBO"))

    @staticmethod
    def _load_safetensors_shards(path: Path) -> dict[str, mx.array]:
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
    def _try_load_saved_component(
        root_path: Path,
        component_name: str,
    ) -> tuple[dict | None, int | None, str | None]:
        component_path = root_path / component_name
        if not component_path.exists():
            return None, None, None

        shard_files = sorted(f for f in component_path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        all_weights: dict[str, mx.array] = {}
        quantization_level: int | None = None
        mflux_version: str | None = None

        for idx, shard in enumerate(shard_files):
            data = mx.load(str(shard), return_metadata=True)
            weights_dict = data[0]
            all_weights.update(dict(weights_dict.items()))

            if idx == 0 and len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")

        if quantization_level is None and mflux_version is None:
            return None, None, None

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version
