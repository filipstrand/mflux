import logging
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx.utils import tree_unflatten
from safetensors import safe_open

from flux_1.config.config import Config

log = logging.getLogger(__name__)


class WeightHandler:

    def __init__(
            self,
            repo_id: str | None = None,
            local_path: str | None = None,
            lora_files=None,
            lora_scales=None
    ):
        if lora_files is None:
            lora_files = []
        if lora_scales is None:
            lora_scales = [1.0]
        root_path = Path(local_path) if local_path else WeightHandler._download_or_get_cached_weights(repo_id)

        self.clip_encoder, _ = WeightHandler._clip_encoder(root_path=root_path)
        self.t5_encoder, _ = WeightHandler._t5_encoder(root_path=root_path)
        self.vae, _ = WeightHandler._vae(root_path=root_path)
        self.transformer, self.quantization_level = WeightHandler._transformer(root_path=root_path)
        if lora_files:
            if len(lora_files) < len(lora_scales):
                lora_scales = lora_scales[0:len(lora_files)]
            if len(lora_scales) < len(lora_files):
                lora_scales = lora_scales + (len(lora_files) - len(lora_scales)) * [1.0]
            for lora_file, lora_scale in zip(lora_files, lora_scales):
                if lora_scale < 0.0 or lora_scale > 1.0:
                    raise Exception(f"Invalid scale {lora_scale} provided for {lora_file}. Valid Range [0.0-1.0] ")

                try:
                    lora_transformer, _ = WeightHandler._lora_transformer(lora_file=lora_file)
                    if 'transformer' not in lora_transformer:
                        raise Exception(
                            "The key `transformer` is missing in the LoRA safetensors file. Please ensure that the file is correctly formatted and contains the expected keys.")
                    WeightHandler._apply_transformer(self.transformer, lora_transformer['transformer'], lora_scale)
                except Exception as e:
                    log.error(f"Error loading the LoRA safetensors file: {e}")

    @staticmethod
    def _apply_transformer(transformer, lora_transformer, lora_scale):
        lora_weights = tree_flatten(lora_transformer)
        visited = {}

        for key, weight in lora_weights:
            splits = key.split(".")
            target = transformer
            visiting = []
            for splitKey in splits:
                if isinstance(target, dict) and splitKey in target:
                    target = target[splitKey]
                    visiting.append(splitKey)
                elif isinstance(target, list) and len(target) > 0:
                    if len(target) < int(splitKey):
                        for _ in range(int(splitKey) - len(target) + 1):
                            target.append({})

                    target = target[int(splitKey)]
                    visiting.append(splitKey)
                else:
                    parentKey = ".".join(visiting)
                    if parentKey in visited and 'lora_A' in visited[parentKey] and 'lora_B' in visited[parentKey]:
                        continue
                    if not splitKey.startswith("lora_"):
                        visiting.append(splitKey)
                        parentKey = ".".join(visiting)
                        if splitKey == "net":
                            target['net'] = list({})
                            target = target['net']
                        elif splitKey == "0":
                            target.append({})
                            target = target[0]
                            continue
                        elif splitKey == "proj":
                            target[splitKey] = weight
                            if parentKey not in visited:
                                visited[parentKey] = {}
                        continue
                    if parentKey not in visited:
                        visited[parentKey] = {}
                    visited[parentKey][splitKey] = weight
                    if not 'weight' in target:
                        raise ValueError(f"LoRA weights for layer {parentKey} cannot be loaded into the model.")
                    if 'lora_A' in visited[parentKey] and 'lora_B' in visited[parentKey]:
                        lora_a = visited[parentKey]['lora_A']
                        lora_b = visited[parentKey]['lora_B']
                        transWeight = target['weight']
                        weight = transWeight + lora_scale * (lora_b @ lora_a)
                        target['weight'] = weight

    @staticmethod
    def _lora_transformer(lora_file: Path) -> (dict, int):
        quantization_level = safe_open(lora_file, framework="pt").metadata().get("quantization_level")
        weights = list(mx.load(str(lora_file)).items())
        weights = [WeightHandler._reshape_weights(k, v) for k, v in weights]
        weights = WeightHandler._flatten(weights)
        unflatten = tree_unflatten(weights)
        for block in unflatten["transformer"]["transformer_blocks"]:
            block["ff"] = {
                "linear1": block["ff"]["net"][0]["proj"],
                "linear2": block["ff"]["net"][2]
            }
            if block.get("ff_context") is not None:
                block["ff_context"] = {
                    "linear1": block["ff_context"]["net"][0]["proj"],
                    "linear2": block["ff_context"]["net"][2]
                }
        return unflatten, quantization_level

    @staticmethod
    def _clip_encoder(root_path: Path) -> (dict, int):
        weights, quantization_level = WeightHandler._get_weights("text_encoder", root_path)
        return weights, quantization_level

    @staticmethod
    def _t5_encoder(root_path: Path) -> (dict, int):
        weights, quantization_level = WeightHandler._get_weights("text_encoder_2", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level

        # Reshape and process the huggingface weights
        weights["final_layer_norm"] = weights["encoder"]["final_layer_norm"]
        for block in weights["encoder"]["block"]:
            attention = block["layer"][0]
            ff = block["layer"][1]
            block.pop("layer")
            block["attention"] = attention
            block["ff"] = ff

        weights["t5_blocks"] = weights["encoder"]["block"]

        # Only the first layer has the weights for "relative_attention_bias", we duplicate them here to keep code simple
        relative_attention_bias = weights["t5_blocks"][0]["attention"]["SelfAttention"]["relative_attention_bias"]
        for block in weights["t5_blocks"][1:]:
            block["attention"]["SelfAttention"]["relative_attention_bias"] = relative_attention_bias

        weights.pop("encoder")
        return weights, quantization_level

    @staticmethod
    def _transformer(root_path: Path) -> (dict, int):
        weights, quantization_level = WeightHandler._get_weights("transformer", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level

        # Reshape and process the huggingface weights
        for block in weights["transformer_blocks"]:
            block["ff"] = {
                "linear1": block["ff"]["net"][0]["proj"],
                "linear2": block["ff"]["net"][2]
            }
            if block.get("ff_context") is not None:
                block["ff_context"] = {
                    "linear1": block["ff_context"]["net"][0]["proj"],
                    "linear2": block["ff_context"]["net"][2]
                }
        return weights, quantization_level

    @staticmethod
    def _vae(root_path: Path) -> (dict, int):
        weights, quantization_level = WeightHandler._get_weights("vae", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level

        # Reshape and process the huggingface weights
        weights['decoder']['conv_in'] = {'conv2d': weights['decoder']['conv_in']}
        weights['decoder']['conv_out'] = {'conv2d': weights['decoder']['conv_out']}
        weights['decoder']['conv_norm_out'] = {'norm': weights['decoder']['conv_norm_out']}
        weights['encoder']['conv_in'] = {'conv2d': weights['encoder']['conv_in']}
        weights['encoder']['conv_out'] = {'conv2d': weights['encoder']['conv_out']}
        weights['encoder']['conv_norm_out'] = {'norm': weights['encoder']['conv_norm_out']}
        return weights, quantization_level

    @staticmethod
    def _get_weights(model_name: str, root_path: Path) -> (dict, int):
        weights = []
        quantization_level = None
        for file in sorted(root_path.glob(model_name + "/*.safetensors")):
            quantization_level = safe_open(file, framework="pt").metadata().get("quantization_level")
            weight = list(mx.load(str(file)).items())
            weights.extend(weight)

        # Non huggingface weights (i.e. ones exported from this project) don't need any reshaping.
        if quantization_level is not None:
            return tree_unflatten(weights), quantization_level

        # Huggingface weights needs to be reshaped
        weights = [WeightHandler._reshape_weights(k, v) for k, v in weights]
        weights = WeightHandler._flatten(weights)
        unflatten = tree_unflatten(weights)
        return unflatten, quantization_level

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
