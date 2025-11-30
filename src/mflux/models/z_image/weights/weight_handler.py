from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping


@dataclass
class MetaData:
    quantization_level: int | None = None
    mflux_version: str | None = None


class WeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        vae: dict | None = None,
        transformer: dict | None = None,
        text_encoder: dict | None = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.meta_data = meta_data

    @staticmethod
    def load_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
        load_text_encoder: bool = True,
    ) -> "WeightHandler":
        root_path = WeightHandler._get_root_path(repo_id, local_path)

        # Try to load mflux-saved weights first
        vae_weights, quantization_level, mflux_version = WeightHandler._try_load_saved_component(root_path, "vae")
        transformer_weights, _, _ = WeightHandler._try_load_saved_component(root_path, "transformer")
        text_encoder_weights, _, _ = WeightHandler._try_load_saved_component(root_path, "text_encoder")

        # If not mflux format, load and map HuggingFace weights
        if vae_weights is None:
            vae_weights = WeightHandler._load_vae_weights(repo_id, local_path)
            quantization_level = None
            mflux_version = None

        if transformer_weights is None:
            transformer_weights = WeightHandler._load_transformer_weights(repo_id, local_path)

        if text_encoder_weights is None and load_text_encoder:
            text_encoder_weights = WeightHandler._load_text_encoder_weights(repo_id, local_path)

        return WeightHandler(
            vae=vae_weights,
            transformer=transformer_weights,
            text_encoder=text_encoder_weights,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_vae_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = WeightHandler._get_root_path(repo_id, local_path)
        vae_path = root_path / "vae"

        # Load raw weights
        raw_weights = WeightHandler._load_safetensors_shards(vae_path)

        # Apply declarative mapping
        mapping = ZImageWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)

        return mapped_weights

    @staticmethod
    def _load_transformer_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = WeightHandler._get_root_path(repo_id, local_path)
        transformer_path = root_path / "transformer"

        # Load raw weights
        raw_weights = WeightHandler._load_safetensors_shards(transformer_path)

        # Apply declarative mapping
        mapping = ZImageWeightMapping.get_transformer_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_layers=30)

        return mapped_weights

    @staticmethod
    def _load_text_encoder_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = WeightHandler._get_root_path(repo_id, local_path, include_text_encoder=True)
        text_encoder_path = root_path / "text_encoder"

        # Load raw weights
        raw_weights = WeightHandler._load_safetensors_shards(text_encoder_path)

        # Apply declarative mapping (36 layers for Qwen3)
        mapping = ZImageWeightMapping.get_text_encoder_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_layers=36)

        return mapped_weights

    @staticmethod
    def _get_root_path(
        repo_id: str | None = None,
        local_path: str | None = None,
        include_text_encoder: bool = False,
    ) -> Path:
        if local_path:
            return Path(local_path)
        return WeightHandler.download_or_get_cached_weights(
            repo_id or "Tongyi-MAI/Z-Image-Turbo",
            include_text_encoder=include_text_encoder,
        )

    @staticmethod
    def download_or_get_cached_weights(repo_id: str, include_text_encoder: bool = False) -> Path:
        patterns = [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
        ]

        if include_text_encoder:
            patterns.extend(
                [
                    "text_encoder/*.safetensors",
                    "text_encoder/*.json",
                    "tokenizer/*",
                ]
            )

        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,
            )
        )

    @staticmethod
    def _load_safetensors_shards(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            # Use MLX native loading - much faster than torch->numpy->mlx
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _try_load_saved_component(
        root_path: Path,
        component_name: str,
    ) -> tuple[dict | None, int | None, str | None]:
        from mlx.utils import tree_unflatten

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

        # If no mflux metadata, this isn't our format
        if quantization_level is None and mflux_version is None:
            return None, None, None

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version
