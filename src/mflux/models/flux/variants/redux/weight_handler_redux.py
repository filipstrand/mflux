from pathlib import Path

from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from mflux.config.model_config import ModelConfig
from mflux.models.common.weights.mapping.weight_transforms import transpose_conv2d_weight
from mflux.models.flux.weights.weight_handler import MetaData, WeightHandler


class WeightHandlerRedux:
    def __init__(self, siglip: dict, redux_encoder: dict, meta_data: MetaData):
        self.siglip = siglip
        self.redux_encoder = redux_encoder
        self.meta_data = meta_data

    @staticmethod
    def load_weights() -> "WeightHandlerRedux":
        root_path = Path(snapshot_download(repo_id=ModelConfig.dev_redux().model_name, allow_patterns=["*.safetensors", "config.json"]))  # fmt:off

        siglip, _, _ = WeightHandlerRedux._load_siglip_weights(root_path=root_path)
        redux_encoder, _, _ = WeightHandlerRedux._load_redux_encoder_weights(root_path=root_path)

        return WeightHandlerRedux(
            siglip=siglip,
            redux_encoder=redux_encoder,
            meta_data=MetaData(quantization_level=None)
        )  # fmt:off

    @staticmethod
    def _load_siglip_weights(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("image_encoder", root_path)
        if quantization_level is None and mflux_version is None:
            weights = {k: transpose_conv2d_weight(v) for k, v in weights.items()}
            weights = tree_unflatten(list(weights.items()))
        return weights, quantization_level, mflux_version

    @staticmethod
    def _load_redux_encoder_weights(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("image_embedder", root_path)
        if quantization_level is None and mflux_version is None:
            weights = tree_unflatten(list(weights.items()))
        return weights, quantization_level, mflux_version
