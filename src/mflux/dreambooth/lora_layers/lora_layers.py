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
                    lora_layers[k].lora_a = weights[layer_number][module]["lora_a"]
                    lora_layers[k].lora_b = weights[layer_number][module]["lora_b"]
                if len(parts) == 5:
                    layer_number = int(parts[-3])
                    module = parts[-2]
                    name = parts[-1]
                    lora_layers[k].lora_a = weights[layer_number][module][name]["lora_a"]
                    lora_layers[k].lora_b = weights[layer_number][module][name]["lora_b"]

        return LoRALayers(lora_layers)

    @staticmethod
    def set_nested_attr(flux, attr_path, value):
        # Split the path and remove the first part ('transformer')
        parts = attr_path.split(".")

        # Start from the transformer
        current = flux.transformer

        # Navigate to the correct block
        block_num = int(parts[2])  # The block number is the third part
        current = current.single_transformer_blocks[block_num]

        # Handle the remaining parts (e.g., 'proj_out' or 'attn.to_q')
        remaining_path = ".".join(parts[3:])
        if "." in remaining_path:  # Handle nested attributes like 'attn.to_q'
            attr_parent, attr_name = remaining_path.split(".")
            current = getattr(current, attr_parent)
            setattr(current, attr_name, value)
        else:  # Handle direct attributes like 'proj_out'
            setattr(current, remaining_path, value)

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
                "quantize": str(training_spec.quantize),
                "single_transformer_blocks": str(training_spec.lora_layers.single_transformer_blocks),
            },
        )
