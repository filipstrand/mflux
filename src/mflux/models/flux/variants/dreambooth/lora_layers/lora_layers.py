from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux.model.flux_transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.flux.model.flux_transformer.single_transformer_block import SingleTransformerBlock
from mflux.models.flux.variants.dreambooth.state.training_spec import (
    SingleTransformerBlocks,
    TrainingSpec,
    TransformerBlocks,
)
from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
from mflux.utils.version_util import VersionUtil

if TYPE_CHECKING:
    from mflux.models.flux.variants.txt2img.flux import Flux1


class LoRALayers:
    @staticmethod
    def from_spec(flux: "Flux1", training_spec: TrainingSpec) -> None:
        if training_spec.lora_layers.state_path is not None:
            # Load from checkpoint using the unified LoRALoader
            ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda lora_file: LoRALoader.load_and_apply_lora(
                    lora_mapping=FluxLoRAMapping.get_mapping(),
                    transformer=flux.transformer,
                    lora_paths=[lora_file],
                    lora_scales=[1.0],
                ),
            )
        else:
            # Construct fresh LoRA layers and apply directly to transformer
            if training_spec.lora_layers.transformer_blocks:
                LoRALayers._construct_and_apply_layers(
                    transformer=flux.transformer,
                    blocks=flux.transformer.transformer_blocks,
                    block_spec=training_spec.lora_layers.transformer_blocks,
                    block_attr="transformer_blocks",
                )

            if training_spec.lora_layers.single_transformer_blocks:
                LoRALayers._construct_and_apply_layers(
                    transformer=flux.transformer,
                    blocks=flux.transformer.single_transformer_blocks,
                    block_spec=training_spec.lora_layers.single_transformer_blocks,
                    block_attr="single_transformer_blocks",
                )

    @staticmethod
    def _construct_and_apply_layers(
        transformer,
        blocks: list[JointTransformerBlock] | list[SingleTransformerBlock],
        block_spec: TransformerBlocks | SingleTransformerBlocks,
        block_attr: str,
    ) -> None:
        block_indices = block_spec.block_range.get_blocks()
        for idx in block_indices:
            if idx >= len(blocks):
                raise IndexError(f"Index {idx} over range")

            block = blocks[idx]
            for layer_type in block_spec.layer_types:
                original_layer = LoRALayers._get_nested_attr(block, layer_type)
                is_list = isinstance(original_layer, list)

                lora_layer = LoRALinear.from_linear(
                    linear=original_layer[0] if is_list else original_layer,
                    r=block_spec.lora_rank,
                )

                # Apply the LoRA layer directly to the transformer
                LoRALayers._set_layer_at_path(
                    transformer=transformer,
                    block_attr=block_attr,
                    block_idx=idx,
                    layer_type=layer_type,
                    lora_layer=lora_layer,
                    is_list=is_list,
                )

    @staticmethod
    def _set_layer_at_path(
        transformer,
        block_attr: str,
        block_idx: int,
        layer_type: str,
        lora_layer: LoRALinear,
        is_list: bool,
    ) -> None:
        block = getattr(transformer, block_attr)[block_idx]
        parts = layer_type.split(".")

        # Navigate to parent
        current = block
        for part in parts[:-1]:
            current = getattr(current, part)

        # Set the final attribute
        final_attr = parts[-1]
        if is_list:
            getattr(current, final_attr)[0] = lora_layer
        else:
            setattr(current, final_attr, lora_layer)

    @staticmethod
    def _get_nested_attr(obj, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    @staticmethod
    def save(transformer, path: Path, training_spec: TrainingSpec) -> None:
        weights = {}
        for entry in tree_flatten(transformer):
            name = entry[0]
            weight = entry[1]
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                weights[name] = weight

        weights = {f"transformer.{key}": mx.transpose(val) for key, val in weights.items()}
        mx.save_safetensors(
            str(path),
            weights,
            metadata={
                "mflux_version": VersionUtil.get_mflux_version(),
                "transformer_blocks": str(training_spec.lora_layers.transformer_blocks),
                "single_transformer_blocks": str(training_spec.lora_layers.single_transformer_blocks),
            },
        )
