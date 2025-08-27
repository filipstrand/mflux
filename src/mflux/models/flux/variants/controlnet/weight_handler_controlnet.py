import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.weights.download import snapshot_download
from mflux.weights.weight_handler import MetaData
from mflux.weights.weight_util import WeightUtil


class WeightHandlerControlnet:
    def __init__(self, meta_data: MetaData, config: dict, controlnet_transformer: dict | None = None):
        self.meta_data = meta_data
        self.controlnet_transformer = controlnet_transformer
        self.config = config

    @staticmethod
    def load_controlnet_transformer(controlnet_model: str) -> "WeightHandlerControlnet":
        controlnet_path = Path(snapshot_download(repo_id=controlnet_model, allow_patterns=["*.safetensors", "config.json"]))  # fmt:off
        file = next(controlnet_path.glob("diffusion_pytorch_model.safetensors"))
        quantization_level = mx.load(str(file), return_metadata=True)[1].get("quantization_level")
        weights = list(mx.load(str(file)).items())
        config = json.load(open(controlnet_path / "config.json"))

        if quantization_level is not None:
            return WeightHandlerControlnet(
                config=config,
                controlnet_transformer=tree_unflatten(weights),
                meta_data=MetaData(quantization_level=quantization_level),
            )

        weights = [WeightUtil.reshape_weights(k, v) for k, v in weights]
        weights = WeightUtil.flatten(weights)
        weights = tree_unflatten(weights)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return WeightHandlerControlnet(
                config=config,
                controlnet_transformer=weights,
                meta_data=MetaData(quantization_level=quantization_level),
            )

        # Reshape and process the huggingface weights
        if "transformer_blocks" in weights:
            for block in weights["transformer_blocks"]:
                block["ff"] = {
                    "linear1": block["ff"]["net"][0]["proj"],
                    "linear2": block["ff"]["net"][2],
                }
                if block.get("ff_context") is not None:
                    block["ff_context"] = {
                        "linear1": block["ff_context"]["net"][0]["proj"],
                        "linear2": block["ff_context"]["net"][2],
                    }

        return WeightHandlerControlnet(
            config=config,
            controlnet_transformer=weights,
            meta_data=MetaData(quantization_level=quantization_level),
        )

    def num_transformer_blocks(self) -> int:
        return self.config["num_layers"]

    def num_single_transformer_blocks(self) -> int:
        return self.config["num_single_layers"]
