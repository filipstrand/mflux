import logging
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx.utils import tree_unflatten
from safetensors import safe_open

from flux_1.weights.weight_handler import WeightHandler

log = logging.getLogger(__name__)


class LoraUtil:

    @staticmethod
    def apply_lora(transformer, lora_files, lora_scales):
        if lora_files:
            if len(lora_files) < len(lora_scales):
                lora_scales = lora_scales[0:len(lora_files)]
            if len(lora_scales) < len(lora_files):
                lora_scales = lora_scales + (len(lora_files) - len(lora_scales)) * [1.0]
            for lora_file, lora_scale in zip(lora_files, lora_scales):
                if lora_scale < 0.0 or lora_scale > 1.0:
                    raise Exception(f"Invalid scale {lora_scale} provided for {lora_file}. Valid Range [0.0-1.0] ")
                try:
                    lora_transformer, _ = LoraUtil._lora_transformer(lora_file=lora_file)
                    if 'transformer' not in lora_transformer:
                        raise Exception("The key `transformer` is missing in the LoRA safetensors file. Please ensure that the file is correctly formatted and contains the expected keys.")
                    LoraUtil._apply_transformer(transformer, lora_transformer['transformer'], lora_scale)
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
        weights = [WeightHandler.reshape_weights(k, v) for k, v in weights]
        weights = WeightHandler.flatten(weights)
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
