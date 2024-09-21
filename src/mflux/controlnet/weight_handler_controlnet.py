import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from mflux.weights.weight_util import WeightUtil


class WeightHandlerControlnet:
    @staticmethod
    def load_controlnet_transformer(controlnet_id: str) -> (dict, int):
        controlnet_path = Path(
            snapshot_download(repo_id=controlnet_id, allow_patterns=["*.safetensors", "config.json"])
        )
        file = next(controlnet_path.glob("diffusion_pytorch_model.safetensors"))
        quantization_level = mx.load(str(file), return_metadata=True)[1].get("quantization_level")
        weights = list(mx.load(str(file)).items())

        if quantization_level is not None:
            return tree_unflatten(weights), quantization_level

        weights = [WeightUtil.reshape_weights(k, v) for k, v in weights]
        weights = WeightUtil.flatten(weights)
        weights = tree_unflatten(weights)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level

        # Reshape and process the huggingface weights
        if "transformer_blocks" in weights:
            for block in weights["transformer_blocks"]:
                block["ff"] = {
                    "linear1": block["ff"]["net"][0]["proj"],
                    "linear2": block["ff"]["net"][2]
                }  # fmt: off
                if block.get("ff_context") is not None:
                    block["ff_context"] = {
                        "linear1": block["ff_context"]["net"][0]["proj"],
                        "linear2": block["ff_context"]["net"][2],
                    }
        config = json.load(open(controlnet_path / "config.json"))
        return weights, quantization_level, config
