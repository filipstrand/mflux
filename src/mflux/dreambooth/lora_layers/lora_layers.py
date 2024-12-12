from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.state.training_spec import TrainingSpec
from mflux.dreambooth.state.zip_util import ZipUtil
from mflux.post_processing.generated_image import GeneratedImage

if TYPE_CHECKING:
    from mflux import Flux1


class LoRALayers:
    def __init__(self, lora_layers: dict[str, nn.Module]):
        self.layers = lora_layers

    @staticmethod
    def from_spec(flux: "Flux1", training_spec: TrainingSpec) -> "LoRALayers":
        block_spec = training_spec.lora_layers.single_transformer_blocks
        start = block_spec.block_range.start
        end = block_spec.block_range.end

        lora_layers = {}

        # Iterate through the specified blocks
        for i in range(start, end):
            block = flux.transformer.single_transformer_blocks[i]

            # For each specified layer type in the config
            for layer_type in block_spec.layer_types:
                # Get the original layer
                original_layer = LoRALayers._get_nested_attr(block, layer_type)

                # Create the LoRA version of the layer
                lora_layer = LoRALinear.from_linear(
                    linear=original_layer,
                    r=block_spec.lora_rank,
                )

                # Store the layer with its path
                layer_path = f"transformer.single_transformer_blocks.{i}.{layer_type}"
                lora_layers[layer_path] = lora_layer
                mx.eval(lora_layers)

        # Load from state if present in the spec
        if training_spec.lora_layers.state_path is not None:
            lora_weights = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda x: tree_unflatten(list(mx.load(x).items())),
            )
            weights = lora_weights["transformer"]["single_transformer_blocks"]
            for k in lora_layers.keys():
                parts = k.split(".")
                if len(parts) == 4:
                    layer_number = int(parts[-2])
                    module = parts[-1]
                    lora_layers[k].lora_A = weights[layer_number][module]["lora_A"]
                    lora_layers[k].lora_B = weights[layer_number][module]["lora_B"]
                if len(parts) == 5:
                    layer_number = int(parts[-3])
                    module = parts[-2]
                    name = parts[-1]
                    lora_layers[k].lora_A = weights[layer_number][module][name]["lora_A"]
                    lora_layers[k].lora_B = weights[layer_number][module][name]["lora_B"]

        return LoRALayers(lora_layers)

    @classmethod
    def from_transformer_template(cls, weights: dict, transformer: nn.Module) -> "LoRALayers":
        lora_layers = {}
        for key in weights.keys():
            if key.endswith(".lora_A"):
                base_path = key[: -len(".lora_A")]
                parts = base_path.split(".")

                if parts[1] == "transformer_blocks":
                    continue

                if parts[1] == "single_transformer_blocks":
                    LoRALayers.handle_single_transformer_blocks(
                        weights=weights,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                    )

        return cls(lora_layers)

    @classmethod
    def handle_single_transformer_blocks(cls, weights: dict, transformer: nn.Module, lora_layers: dict, base_path: str):
        parts = base_path.split(".")
        if len(parts) == 4:
            block_idx = int(parts[2])
            module_name = parts[3]
            original_layer = getattr(transformer.single_transformer_blocks[block_idx], module_name)
        elif len(parts) == 5:
            block_idx = int(parts[2])
            module_name = parts[3]
            attr_name = parts[4]
            block = transformer.single_transformer_blocks[block_idx]
            module = getattr(block, module_name)
            original_layer = getattr(module, attr_name)

        # Create LoRA layer
        lora_A = weights[f"{base_path}.lora_A"]
        rank = lora_A.shape[1]
        lora_layer = LoRALinear.from_linear(linear=original_layer, r=rank)

        # Set the weights
        lora_layer.lora_A = weights[f"{base_path}.lora_A"]
        lora_layer.lora_B = weights[f"{base_path}.lora_B"]

        # Store the layer
        lora_layers[base_path] = lora_layer

    @staticmethod
    def set_single_transformer_block(single_transformer_block, dictionary: dict):
        for key, val in dictionary.items():
            if key == "attn":
                single_transformer_block[key]["to_q"] = val["to_q"]
                single_transformer_block[key]["to_k"] = val["to_k"]
                single_transformer_block[key]["to_v"] = val["to_v"]
            elif key == "norm":
                single_transformer_block[key]["linear"] = val
            elif key == "proj_mlp" or key == "proj_out":
                single_transformer_block[key] = val
            else:
                raise Exception("Could not set LoRA weights")

    @staticmethod
    def _get_nested_attr(obj, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    def save(self, path: Path, training_spec: TrainingSpec) -> None:
        weights = {}
        for key, val in self.layers.items():
            weights[key] = val.trainable_parameters()

        mx.save_safetensors(
            str(path),
            dict(tree_flatten(weights)),
            metadata={
                "mflux_version": GeneratedImage.get_version(),
                "single_transformer_blocks": str(training_spec.lora_layers.single_transformer_blocks),
            },
        )
