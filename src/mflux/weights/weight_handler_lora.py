import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.dreambooth.lora_layers.fused_linear_lora_layer import FusedLoRALinear
from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.weights.weight_handler import MetaData, WeightHandler


class WeightHandlerLoRA:
    def __init__(self, weight_handlers: list[WeightHandler]):
        self.weight_handlers = weight_handlers

    @staticmethod
    def load_lora_weights(
        transformer: nn.Module,
        lora_files: list[str],
        lora_scales: list[float] | None = None,
    ) -> list["WeightHandler"]:
        lora_weights = []
        if lora_files:
            lora_scales = WeightHandlerLoRA._validate_lora_scales(lora_files, lora_scales)
            for lora_file, lora_scale in zip(lora_files, lora_scales):
                weights, _ = WeightHandler.load_transformer(lora_path=lora_file)
                weights = dict(tree_flatten(weights))
                weights = {key.removesuffix(".weight"): value for key, value in weights.items()}
                weights = {f"transformer.{key}": value for key, value in weights.items()}
                lora_transformer_dict = LoRALayers.transformer_dict_from_template(weights, transformer, lora_scale)
                transformer_weights = tree_unflatten(list(lora_transformer_dict.items()))["transformer"]
                weights = WeightHandler(
                    clip_encoder=None,
                    t5_encoder=None,
                    vae=None,
                    transformer=transformer_weights,
                    meta_data=MetaData(
                        quantization_level=None,
                        scale=lora_scale,
                        is_lora=True,
                        is_mflux=False,
                    ),
                )
                lora_weights.append(weights)

        return lora_weights

    @staticmethod
    def set_lora_weights(transformer: nn.Module, loras: list["WeightHandler"]) -> None:
        if loras:
            fused_weights = WeightHandlerLoRA._fuse_lora_dicts(loras[0].transformer, loras[1].transformer)
            fused_weights = WeightHandler(
                meta_data=MetaData(),
                clip_encoder=None,
                t5_encoder=None,
                vae=None,
                transformer=fused_weights,
            )

            WeightHandlerLoRA.set_lora_layers(
                transformer_module=transformer,
                lora_layers=LoRALayers(weights=fused_weights)
            )  # fmt:off

    @staticmethod
    def _fuse_lora_dicts(dict1: dict, dict2: dict) -> dict:
        fused_dict = {}

        for key in dict1.keys():
            if key not in dict2:
                raise ValueError(f"Key {key} is missing in the second dictionary.")

            value1 = dict1[key]
            value2 = dict2[key]

            # Recursively handle nested dictionaries
            if (
                isinstance(value1, dict)
                and isinstance(value2, dict)
                and not isinstance(value1, LoRALinear)
                and not isinstance(value2, LoRALinear)
            ):
                fused_dict[key] = WeightHandlerLoRA._fuse_lora_dicts(value1, value2)

            # Handle LoRALinear layers
            elif isinstance(value1, LoRALinear) and isinstance(value2, LoRALinear):
                fused_layer = FusedLoRALinear(base_linear=value1.linear, loras=[value1, value2])
                fused_dict[key] = fused_layer

            # Handle lists
            elif isinstance(value1, list) and isinstance(value2, list):
                if len(value1) != len(value2):
                    raise ValueError(f"Lists for key {key} have different lengths.")
                fused_dict[key] = [
                    WeightHandlerLoRA._fuse_lora_dicts({str(idx): v1}, {str(idx): v2})[str(idx)]
                    if isinstance(v1, (dict, LoRALinear)) and isinstance(v2, (dict, LoRALinear))
                    else v1  # Or apply other rules for non-LoRALinear types
                    for idx, (v1, v2) in enumerate(zip(value1, value2))
                ]

            else:
                raise ValueError(f"Incompatible types for key {key}: {type(value1)} and {type(value2)}.")

        return fused_dict

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
    def set_lora_layers(transformer_module: nn.Module, lora_layers: LoRALayers) -> None:
        transformer = lora_layers.layers.transformer

        # Handle transformer_blocks
        transformer_blocks = transformer.get("transformer_blocks", [])
        for i, weights in enumerate(transformer_blocks):
            LoRALayers.set_transformer_block(
                transformer_block=transformer_module.transformer_blocks[i],
                dictionary=weights
            )  # fmt:off

        # Handle single_transformer_blocks
        single_transformer_blocks = transformer.get("single_transformer_blocks", [])
        for i, weights in enumerate(single_transformer_blocks):
            LoRALayers.set_single_transformer_block(
                single_transformer_block=transformer_module.single_transformer_blocks[i],
                dictionary=weights
            )  # fmt:off
