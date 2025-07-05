from pathlib import Path

from mflux import ModelConfig
from mflux.weights.download import snapshot_download
from mflux.weights.weight_handler import MetaData, WeightHandler


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
        weights, _, _ = WeightHandler.get_weights("image_encoder", root_path)
        return weights, _, _

    @staticmethod
    def _load_redux_encoder_weights(root_path: Path) -> tuple[dict, int, str | None]:
        weights, _, _ = WeightHandler.get_weights("image_embedder", root_path)
        return weights, _, _
