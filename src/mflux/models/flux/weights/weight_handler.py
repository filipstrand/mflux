from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from mflux.config.model_config import ModelConfig
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


@dataclass
class MetaData:
    quantization_level: int | None = None
    scale: float | None = None
    is_lora: bool = False
    mflux_version: str | None = None


class WeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        clip_encoder: dict | None = None,
        t5_encoder: dict | None = None,
        vae: dict | None = None,
        transformer: dict | None = None,
    ):
        self.clip_encoder = clip_encoder
        self.t5_encoder = t5_encoder
        self.vae = vae
        self.transformer = transformer
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
        transformer_repo_id: str | None = None,
    ) -> "WeightHandler":
        # Load the weights from disk, huggingface cache, or download from huggingface
        root_path = Path(local_path) if local_path else WeightHandler.download_or_get_cached_weights(repo_id)

        # Some custom models might have a different specific transformer setup
        if transformer_repo_id:
            transformer_path = WeightHandler._download_transformer_weights(transformer_repo_id)
        else:
            transformer_path = root_path

        # Load the weights
        transformer, quantization_level, mflux_version = WeightHandler.load_transformer(root_path=transformer_path)
        clip_encoder, _, _ = WeightHandler._load_clip_encoder(root_path=root_path)
        t5_encoder, _, _ = WeightHandler._load_t5_encoder(root_path=root_path)
        vae, _, _ = WeightHandler._load_vae(root_path=root_path)

        return WeightHandler(
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            vae=vae,
            transformer=transformer,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    def num_transformer_blocks(self) -> int:
        return len(self.transformer["transformer_blocks"])

    def num_single_transformer_blocks(self) -> int:
        return len(self.transformer["single_transformer_blocks"])

    @staticmethod
    def _load_clip_encoder(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("text_encoder", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level, mflux_version

        # Apply declarative mapping to HuggingFace weights
        weights = WeightMapper.apply_mapping(
            hf_weights=weights,
            mapping=FluxWeightMapping.get_clip_encoder_mapping(),
        )

        return weights, quantization_level, mflux_version

    @staticmethod
    def _load_t5_encoder(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("text_encoder_2", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level, mflux_version

        # Apply declarative mapping to HuggingFace weights (T5 has 24 blocks)
        weights = WeightMapper.apply_mapping(
            hf_weights=weights,
            mapping=FluxWeightMapping.get_t5_encoder_mapping(),
            num_blocks=24,
        )

        return weights, quantization_level, mflux_version

    @staticmethod
    def load_transformer(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("transformer", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level or mflux_version:
            return weights, quantization_level, mflux_version

        # Apply declarative mapping to HuggingFace weights
        weights = WeightMapper.apply_mapping(
            hf_weights=weights,
            mapping=FluxWeightMapping.get_transformer_mapping(),
        )

        return weights, quantization_level, mflux_version

    @staticmethod
    def _load_vae(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("vae", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level, mflux_version

        # Apply declarative mapping to HuggingFace weights
        weights = WeightMapper.apply_mapping(
            hf_weights=weights,
            mapping=FluxWeightMapping.get_vae_mapping(),
        )

        return weights, quantization_level, mflux_version

    @staticmethod
    def _get_model_file_pattern(model_name: str, root_path: Path):
        if model_name == "transformer":
            nested_files = list(root_path.glob("transformer/*.safetensors"))
            if nested_files:
                return root_path.glob("transformer/*.safetensors")
            else:
                return root_path.glob("*.safetensors")
        else:
            return root_path.glob(model_name + "/*.safetensors")

    @staticmethod
    def get_weights(model_name: str, root_path: Path) -> tuple[dict, int, str | None]:
        weights = []
        quantization_level = None
        mflux_version = None

        file_glob = WeightHandler._get_model_file_pattern(model_name, root_path)
        for file in sorted(file_glob):
            data = mx.load(str(file), return_metadata=True)
            weight = list(data[0].items())
            if len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")
            weights.extend(weight)

        # Non huggingface weights (i.e. ones exported from this project) don't need any reshaping.
        if quantization_level is not None or mflux_version is not None:
            return tree_unflatten(weights), quantization_level, mflux_version

        # HuggingFace weights: cast to precision (transforms like transpose are handled by declarative mapping)
        weights = [(k, v.astype(ModelConfig.precision)) for k, v in weights]
        return dict(weights), quantization_level, mflux_version

    @staticmethod
    def download_or_get_cached_weights(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "text_encoder/*.safetensors",
                    "text_encoder/*.json",
                    "text_encoder_2/*.safetensors",
                    "text_encoder_2/*.json",
                    "transformer/*.safetensors",
                    "transformer/*.json",
                    "vae/*.safetensors",
                    "vae/*.json",
                ],
            )
        )

    @staticmethod
    def _download_transformer_weights(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "transformer/*.safetensors",
                    "*.safetensors",
                ],
            )
        )
