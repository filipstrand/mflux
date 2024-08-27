from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from flux_1.config.config import Config


class WeightHandler:

    def __init__(
            self,
            repo_id: str | None = None,
            local_path: str | None = None,
            should_reshape: bool = True,
    ):
        root_path = local_path if local_path else WeightHandler._download_or_get_cached_weights(repo_id)

        self.clip_encoder = WeightHandler._clip_encoder(
            should_reshape=should_reshape,
            weights=WeightHandler._load(root_path / "text_encoder/model.safetensors")
        )
        self.t5_encoder = WeightHandler._t5_encoder(
            should_reshape=should_reshape,
            weights_1=WeightHandler._load(root_path / "text_encoder_2/model-00001-of-00002.safetensors"),
            weights_2=WeightHandler._load(root_path / "text_encoder_2/model-00002-of-00002.safetensors"),
        )
        self.vae = WeightHandler._vae(
            should_reshape=should_reshape,
            weights=WeightHandler._load(root_path / "vae/diffusion_pytorch_model.safetensors")
        )
        self.transformer = WeightHandler._transformer(
            should_reshape=should_reshape,
            weights_1=WeightHandler._load(root_path / "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"),
            weights_2=WeightHandler._load(root_path / "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"),
            weights_3=WeightHandler._load(root_path / "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"),
        )

    @staticmethod
    def load_from_cache_or_huggingface(repo_id: str) -> "WeightHandler":
        return WeightHandler(repo_id=repo_id, local_path=None, should_reshape=True)

    @staticmethod
    def load_quantized_model_from_disk(path: str) -> "WeightHandler":
        return WeightHandler(repo_id=None, local_path=path, should_reshape=False)

    @staticmethod
    def load_huggingface_model_from_disk(path: str) -> "WeightHandler":
        return WeightHandler(repo_id=None, local_path=path, should_reshape=True)

    @staticmethod
    def _load(path: Path) -> list[dict]:
        return list(mx.load(str(path)).items())

    @staticmethod
    def _clip_encoder(should_reshape: bool, weights: list[dict]) -> dict:
        if should_reshape:
            weights = [WeightHandler._reshape_weights(k, v) for k, v in weights]
        weights = WeightHandler._flatten(weights)
        unflatten = tree_unflatten(weights)
        return unflatten

    @staticmethod
    def _t5_encoder(
            should_reshape: bool,
            weights_1: list[dict],
            weights_2: list[dict],
    ) -> dict:
        if should_reshape:
            weights_1 = [WeightHandler._reshape_weights(k, v) for k, v in weights_1]
            weights_2 = [WeightHandler._reshape_weights(k, v) for k, v in weights_2]
        weights_1 = WeightHandler._flatten(weights_1)
        weights_2 = WeightHandler._flatten(weights_2)
        unflatten = tree_unflatten(weights_1 + weights_2)
        unflatten["final_layer_norm"] = unflatten["encoder"]["final_layer_norm"]
        for block in unflatten["encoder"]["block"]:
            attention = block["layer"][0]
            ff = block["layer"][1]
            block.pop("layer")
            block["attention"] = attention
            block["ff"] = ff

        unflatten["t5_blocks"] = unflatten["encoder"]["block"]

        # Only the first layer has the weights for "relative_attention_bias", we duplicate them here to keep code simple
        relative_attention_bias = unflatten["t5_blocks"][0]["attention"]["SelfAttention"]["relative_attention_bias"]
        for block in unflatten["t5_blocks"][1:]:
            block["attention"]["SelfAttention"]["relative_attention_bias"] = relative_attention_bias

        unflatten.pop("encoder")
        return unflatten

    @staticmethod
    def _transformer(
            should_reshape: bool,
            weights_1: list[dict],
            weights_2: list[dict],
            weights_3: list[dict],
    ) -> dict:
        if should_reshape:
            weights_1 = [WeightHandler._reshape_weights(k, v) for k, v in weights_1]
            weights_2 = [WeightHandler._reshape_weights(k, v) for k, v in weights_2]
            weights_3 = [WeightHandler._reshape_weights(k, v) for k, v in weights_3]
        weights_1 = WeightHandler._flatten(weights_1)
        weights_2 = WeightHandler._flatten(weights_2)
        weights_3 = WeightHandler._flatten(weights_3)
        unflatten = tree_unflatten(weights_1 + weights_2 + weights_3)

        for block in unflatten["transformer_blocks"]:
            block["ff"] = {
                "linear1": block["ff"]["net"][0]["proj"],
                "linear2": block["ff"]["net"][2]
            }
            if block.get("ff_context") is not None:
                block["ff_context"] = {
                    "linear1": block["ff_context"]["net"][0]["proj"],
                    "linear2": block["ff_context"]["net"][2]
                }
        return unflatten

    @staticmethod
    def _vae(should_reshape: bool, weights: list[dict]) -> dict:
        if should_reshape:
            weights = [WeightHandler._reshape_weights(k, v) for k, v in weights]
        weights = WeightHandler._flatten(weights)
        unflatten = tree_unflatten(weights)
        unflatten['decoder']['conv_in'] = {'conv2d': unflatten['decoder']['conv_in']}
        unflatten['decoder']['conv_out'] = {'conv2d': unflatten['decoder']['conv_out']}
        unflatten['decoder']['conv_norm_out'] = {'norm': unflatten['decoder']['conv_norm_out']}
        unflatten['encoder']['conv_in'] = {'conv2d': unflatten['encoder']['conv_in']}
        unflatten['encoder']['conv_out'] = {'conv2d': unflatten['encoder']['conv_out']}
        unflatten['encoder']['conv_norm_out'] = {'norm': unflatten['encoder']['conv_norm_out']}
        return unflatten

    @staticmethod
    def _flatten(params):
        return [(k, v) for p in params for (k, v) in p]

    @staticmethod
    def _reshape_weights(key, value):
        if len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape).astype(Config.precision)
        return [(key, value)]

    @staticmethod
    def _download_or_get_cached_weights(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "text_encoder/*.safetensors",
                    "text_encoder_2/*.safetensors",
                    "transformer/*.safetensors",
                    "vae/*.safetensors",
                ]
            )
        )
