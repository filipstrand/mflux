import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

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
                lora_transformer_dict = LoRALayers.transformer_dict_from_template(weights, transformer)
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
            merged_weights = WeightHandlerLoRA.merge_lora_weights(
                transformer_module=transformer,
                loras=loras
            )  # fmt:off
            WeightHandlerLoRA.set_lora_layers(
                transformer_module=transformer,
                lora_layers=LoRALayers(weights=merged_weights)
            )  # fmt:off

    @staticmethod
    def merge_lora_weights(transformer_module: nn.Module, loras: list["WeightHandler"]) -> "WeightHandler":
        merged_dict = {}
        for lora_idx, lora in enumerate(loras):
            flattened = tree_flatten(lora.transformer)
            for name, value in flattened:
                if name.endswith(".lora_A") or name.endswith(".lora_B"):
                    if name not in merged_dict:
                        merged_dict[name] = lora.meta_data.scale * value
                    else:
                        merged_dict[name] += lora.meta_data.scale * value

        merged_dict = {f"transformer.{key}": value for key, value in merged_dict.items()}
        lora_transformer_dict = LoRALayers.transformer_dict_from_template(merged_dict, transformer_module)
        transformer_lora_weights = tree_unflatten(list(lora_transformer_dict.items()))["transformer"]

        arbitrary_lora = loras[0]
        return WeightHandler(
            meta_data=arbitrary_lora.meta_data,
            clip_encoder=arbitrary_lora.clip_encoder,
            t5_encoder=arbitrary_lora.t5_encoder,
            vae=arbitrary_lora.vae,
            transformer=transformer_lora_weights,
        )

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
