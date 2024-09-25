import logging

from mlx.utils import tree_flatten

log = logging.getLogger(__name__)


class LoraUtil:
    @staticmethod
    def apply_loras(
        transformer: dict,
        lora_files: list[str],
        lora_scales: list[float] | None = None,
    ) -> None:
        lora_scales = LoraUtil._validate_lora_scales(lora_files, lora_scales)

        for lora_file, lora_scale in zip(lora_files, lora_scales):
            LoraUtil._apply_lora(transformer, lora_file, lora_scale)

    @staticmethod
    def _validate_lora_scales(lora_files: list[str], lora_scales: list[float]) -> list[float]:
        if len(lora_files) == 1:
            if not lora_scales:
                lora_scales = [1.0]
            if len(lora_scales) > 1:
                raise ValueError("Please provide a single scale for the LoRA, or skip it to default to 1")
        elif len(lora_files) > 1:
            if len(lora_files) != len(lora_scales):
                raise ValueError("When providing multiple LoRAs, be sure to specify a scale for each one respectively")
        return lora_scales

    @staticmethod
    def _apply_lora(transformer: dict, lora_file: str, lora_scale: float) -> None:
        from mflux.weights.weight_handler import WeightHandler

        lora_transformer, _ = WeightHandler.load_transformer(lora_path=lora_file)
        LoraUtil._apply_transformer(transformer, lora_transformer, lora_scale)

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
                    if parentKey in visited and "lora_A" in visited[parentKey] and "lora_B" in visited[parentKey]:
                        continue
                    if not splitKey.startswith("lora_"):
                        visiting.append(splitKey)
                        parentKey = ".".join(visiting)
                        if splitKey == "net":
                            target["net"] = list({})
                            target = target["net"]
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
                    if "weight" not in target:
                        raise ValueError(f"LoRA weights for layer {parentKey} cannot be loaded into the model.")
                    if "lora_A" in visited[parentKey] and "lora_B" in visited[parentKey]:
                        lora_a = visited[parentKey]["lora_A"]
                        lora_b = visited[parentKey]["lora_B"]
                        transWeight = target["weight"]
                        weight = transWeight + lora_scale * (lora_b @ lora_a)
                        target["weight"] = weight
