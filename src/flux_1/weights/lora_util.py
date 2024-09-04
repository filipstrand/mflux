import logging
from pathlib import Path

from mlx.utils import tree_flatten


log = logging.getLogger(__name__)


class LoraUtil:

    @staticmethod
    def apply_lora(transformer: dict, lora_file: str, lora_scale: float) -> None:
        if lora_scale < 0.0 or lora_scale > 1.0:
            raise Exception(f"Invalid scale {lora_scale} provided for {lora_file}. Valid Range [0.0 - 1.0] ")
        try:
            from flux_1.weights.weight_handler import WeightHandler
            lora_transformer, _ = WeightHandler.load_transformer(lora_path=lora_file)
            LoraUtil._apply_transformer(transformer, lora_transformer, lora_scale)
        except Exception as e:
            log.error(f"Error loading the LoRA safetensors file: {e}")

    @staticmethod
    def _apply_transformer(transformer: dict, lora_transformer: dict, lora_scale: float) -> None:
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
