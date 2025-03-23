from pathlib import Path

from huggingface_hub import snapshot_download

from mflux import ModelConfig
from mflux.weights.weight_handler import WeightHandler


class WeightHandlerRedux:
    def __init__(self, siglip: dict, redux_encoder: dict):
        self.siglip = siglip
        self.redux_encoder = redux_encoder

    @staticmethod
    def load_weights() -> "WeightHandlerRedux":
        root_path = Path(snapshot_download(repo_id=ModelConfig.dev_redux().model_name, allow_patterns=["*.safetensors", "config.json"]))  # fmt:off

        siglip, _, _ = WeightHandlerRedux._load_siglip_weights(root_path=root_path)
        redux_encoder, _, _ = WeightHandlerRedux._load_redux_encoder_weights(root_path=root_path)

        return WeightHandlerRedux(
            siglip=siglip,
            redux_encoder=redux_encoder,
        )  # fmt:off

    @staticmethod
    def _load_siglip_weights(root_path: Path) -> (dict, int, str | None):
        weights, _, _ = WeightHandler.get_weights("image_encoder", root_path)
        return weights, _, _

    @staticmethod
    def _load_redux_encoder_weights(root_path: Path) -> (dict, int, str | None):
        weights, _, _ = WeightHandler.get_weights("image_embedder", root_path)
        return weights, _, _
