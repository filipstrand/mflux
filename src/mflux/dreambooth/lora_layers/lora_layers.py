from pathlib import Path
from typing import TYPE_CHECKING

import mlx
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.state.training_spec import SingleTransformerBlocks, TrainingSpec, TransformerBlocks
from mflux.dreambooth.state.zip_util import ZipUtil
from mflux.models.transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.transformer.single_transformer_block import SingleTransformerBlock
from mflux.post_processing.generated_image import GeneratedImage
from mflux.weights.weight_handler import MetaData, WeightHandler

if TYPE_CHECKING:
    from mflux import Flux1


class LoRALayers:
    def __init__(self, weights: "WeightHandler"):
        self.layers = weights

    @staticmethod
    def from_spec(flux: "Flux1", training_spec: TrainingSpec) -> "LoRALayers":
        if training_spec.lora_layers.state_path is not None:
            # Load from state if present in the spec
            from mflux.weights.weight_handler_lora import WeightHandlerLoRA

            weights = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda x: WeightHandlerLoRA.load_lora_weights(
                    transformer=flux.transformer, lora_files=[x], lora_scales=[1.0]
                ),
            )
            return LoRALayers(weights=weights[0])
        else:
            # Construct the LoRA weights from the spec
            transformer_lora_layers = {}
            single_transformer_lora_layers = {}

            if training_spec.lora_layers.transformer_blocks:
                transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.transformer_blocks,
                    block_spec=training_spec.lora_layers.transformer_blocks,
                    block_prefix="transformer.transformer_blocks",
                )

            if training_spec.lora_layers.single_transformer_blocks:
                single_transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.single_transformer_blocks,
                    block_spec=training_spec.lora_layers.single_transformer_blocks,
                    block_prefix="transformer.single_transformer_blocks",
                )

            lora_layers = {**transformer_lora_layers, **single_transformer_lora_layers}

            weights = WeightHandler(
                meta_data=MetaData(is_mflux=True),
                transformer=mlx.utils.tree_unflatten(list(lora_layers.items()))['transformer'],
            )  # fmt:off

            return LoRALayers(weights=weights)

    @staticmethod
    def _construct_layers(
        block_spec: TransformerBlocks | SingleTransformerBlocks,
        blocks: list[JointTransformerBlock] | list[SingleTransformerBlock],
        block_prefix: str,
    ) -> dict:
        start = block_spec.block_range.start
        end = block_spec.block_range.end
        lora_layers = {}
        for i in range(start, end):
            block = blocks[i]

            for layer_type in block_spec.layer_types:
                original_layer = LoRALayers._get_nested_attr(block, layer_type)
                is_list = isinstance(original_layer, list)

                lora_layer = LoRALinear.from_linear(
                    linear=original_layer[0] if is_list else original_layer,
                    r=block_spec.lora_rank,
                )
                layer_path = f"{block_prefix}.{i}.{layer_type}"

                lora_layers[layer_path] = [lora_layer] if is_list else lora_layer

        return lora_layers

    @staticmethod
    def transformer_dict_from_template(weights: dict, transformer: nn.Module, scale: float) -> dict:
        lora_layers = {}
        for key in weights.keys():
            if key.endswith(".lora_A"):
                base_path = key[: -len(".lora_A")]
                parts = base_path.split(".")

                if parts[1] == "transformer_blocks":
                    LoRALayers._handle_transformer_blocks(
                        weights=weights,
                        scale=scale,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                    )

                if parts[1] == "single_transformer_blocks":
                    LoRALayers._handle_single_transformer_blocks(
                        weights=weights,
                        scale=scale,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                    )

        return lora_layers

    @staticmethod
    def _handle_transformer_blocks(
        weights: dict, scale: float, transformer: nn.Module, lora_layers: dict, base_path: str
    ):
        parts = base_path.split(".")
        block_idx = int(parts[2])
        module_name = parts[3]
        attr_name = parts[4]
        block = transformer.transformer_blocks[block_idx]
        module = getattr(block, module_name)
        original_layer = getattr(module, attr_name)

        if len(parts) == 6:
            original_layer = original_layer[0]  # Special case here

        # Create LoRA layer
        lora_A = weights[f"{base_path}.lora_A"]
        rank = lora_A.shape[1]
        lora_layer = LoRALinear.from_linear(linear=original_layer, r=rank, scale=scale)

        # Set the weights
        lora_layer.lora_A = weights[f"{base_path}.lora_A"]
        lora_layer.lora_B = weights[f"{base_path}.lora_B"]

        # Store the layer
        lora_layers[base_path] = lora_layer

    @staticmethod
    def _handle_single_transformer_blocks(
        weights: dict, scale: float, transformer: nn.Module, lora_layers: dict, base_path: str
    ):
        parts = base_path.split(".")
        block_idx = int(parts[2])
        module_name = parts[3]
        if len(parts) == 4:
            original_layer = getattr(transformer.single_transformer_blocks[block_idx], module_name)
        elif len(parts) == 5:
            attr_name = parts[4]
            block = transformer.single_transformer_blocks[block_idx]
            module = getattr(block, module_name)
            original_layer = getattr(module, attr_name)

        # Create LoRA layer
        lora_A = weights[f"{base_path}.lora_A"]
        rank = lora_A.shape[1]
        lora_layer = LoRALinear.from_linear(linear=original_layer, r=rank, scale=scale)

        # Set the weights
        lora_layer.lora_A = weights[f"{base_path}.lora_A"]
        lora_layer.lora_B = weights[f"{base_path}.lora_B"]

        # Store the layer
        lora_layers[base_path] = lora_layer

    @staticmethod
    def set_transformer_block(transformer_block, dictionary: dict):
        for key, val in dictionary.items():
            if key == "attn":
                LoRALayers._set_attribute(transformer_block, key, val, "to_q")
                LoRALayers._set_attribute(transformer_block, key, val, "to_k")
                LoRALayers._set_attribute(transformer_block, key, val, "to_v")
                LoRALayers._set_attribute(transformer_block, key, val, "to_out")
                LoRALayers._set_attribute(transformer_block, key, val, "add_q_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "add_k_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "add_v_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "to_add_out")
            elif key == "ff" or key == "ff_context":
                LoRALayers._set_attribute(transformer_block, key, val, "linear1")
                LoRALayers._set_attribute(transformer_block, key, val, "linear2")
            elif key == "norm1" or key == "norm1_context":
                LoRALayers._set_attribute(transformer_block, key, val, "linear")
            else:
                raise Exception("Could not set LoRA weights")

    @staticmethod
    def set_single_transformer_block(single_transformer_block, dictionary: dict):
        for key, val in dictionary.items():
            if key == "attn":
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_q")
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_k")
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_v")
            elif key == "norm":
                LoRALayers._set_attribute(single_transformer_block, key, val, "linear")
            elif key == "proj_mlp" or key == "proj_out":
                single_transformer_block[key] = val
            else:
                raise Exception("Could not set LoRA weights")

    @staticmethod
    def _set_attribute(block, key: str, val: dict, name: str):
        if block[key].get(name, False) and val.get(name, False):
            block[key][name] = val[name]

    @staticmethod
    def _get_nested_attr(obj, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    def save(self, path: Path, training_spec: TrainingSpec) -> None:
        weights = {}
        for entry in tree_flatten(self.layers.transformer):
            name = entry[0]
            weight = entry[1]
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                weights[name] = weight

        weights = {key: mx.transpose(val) for key, val in weights.items()}
        weights = {"transformer": weights}
        mx.save_safetensors(
            str(path),
            dict(tree_flatten(weights)),
            metadata={
                "mflux_version": GeneratedImage.get_version(),
                "transformer_blocks": str(training_spec.lora_layers.transformer_blocks),
                "single_transformer_blocks": str(training_spec.lora_layers.single_transformer_blocks),
            },
        )
