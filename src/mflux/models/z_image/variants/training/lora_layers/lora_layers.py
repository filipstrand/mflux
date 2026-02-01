from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.z_image.model.z_image_transformer.context_block import ZImageContextBlock
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock
from mflux.models.z_image.variants.training.state.training_spec import (
    TrainingSpec,
    ZImageTransformerBlocks,
)
from mflux.models.z_image.variants.training.state.zip_util import ZipUtil
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.utils.version_util import VersionUtil

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.z_image_base import ZImageBase


class ZImageLoRALayers:
    """LoRA layer management for Z-Image training."""

    @staticmethod
    def from_spec(model: "ZImageBase", training_spec: TrainingSpec) -> None:
        """Apply LoRA layers to transformer based on training spec."""
        if training_spec.lora_layers is None:
            return

        if training_spec.lora_layers.state_path is not None:
            # Load from checkpoint
            ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda lora_file: LoRALoader.load_and_apply_lora(
                    lora_mapping=ZImageLoRAMapping.get_mapping(),
                    transformer=model.transformer,
                    lora_paths=[lora_file],
                    lora_scales=[1.0],
                ),
            )
        else:
            # Construct fresh LoRA layers
            lora_spec = training_spec.lora_layers

            # Apply to main layers (30 transformer blocks)
            if lora_spec.main_layers:
                ZImageLoRALayers._construct_and_apply_layers(
                    transformer=model.transformer,
                    blocks=model.transformer.layers,
                    block_spec=lora_spec.main_layers,
                    block_attr="layers",
                )

            # Apply to noise refiner (2 blocks)
            if lora_spec.noise_refiner:
                ZImageLoRALayers._construct_and_apply_layers(
                    transformer=model.transformer,
                    blocks=model.transformer.noise_refiner,
                    block_spec=lora_spec.noise_refiner,
                    block_attr="noise_refiner",
                )

            # Apply to context refiner (2 blocks)
            if lora_spec.context_refiner:
                ZImageLoRALayers._construct_and_apply_layers(
                    transformer=model.transformer,
                    blocks=model.transformer.context_refiner,
                    block_spec=lora_spec.context_refiner,
                    block_attr="context_refiner",
                )

    @staticmethod
    def _construct_and_apply_layers(
        transformer: object,
        blocks: list[ZImageTransformerBlock] | list[ZImageContextBlock],
        block_spec: ZImageTransformerBlocks,
        block_attr: str,
    ) -> None:
        """Construct and apply LoRA layers to specified transformer blocks."""
        block_indices = block_spec.block_range.get_blocks()

        for idx in block_indices:
            if idx >= len(blocks):
                raise IndexError(f"Block index {idx} out of range for {block_attr} (max: {len(blocks) - 1})")

            block = blocks[idx]
            for layer_type in block_spec.layer_types:
                original_layer = ZImageLoRALayers._get_nested_attr(block, layer_type)
                is_list = isinstance(original_layer, list)

                # Validate LoRA rank
                if block_spec.lora_rank is None or block_spec.lora_rank <= 0:
                    raise ValueError(f"Invalid LoRA rank: {block_spec.lora_rank}. Must be a positive integer.")

                lora_layer = LoRALinear.from_linear(
                    linear=original_layer[0] if is_list else original_layer,
                    r=block_spec.lora_rank,
                )

                # Apply the LoRA layer
                ZImageLoRALayers._set_layer_at_path(
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
        """Set a LoRA layer at the specified path in the transformer."""
        block = getattr(transformer, block_attr)[block_idx]
        parts = layer_type.split(".")

        # Navigate to parent (handle numeric indices in path)
        current = block
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        # Set the final attribute (handle numeric index for list assignment)
        final_attr = parts[-1]
        if final_attr.isdigit():
            # List index assignment (e.g., to_out[0])
            current[int(final_attr)] = lora_layer
        elif is_list:
            getattr(current, final_attr)[0] = lora_layer
        else:
            setattr(current, final_attr, lora_layer)

    @staticmethod
    def _get_nested_attr(obj: object, attr_path: str) -> object:
        """Get a nested attribute by dot-separated path.

        Args:
            obj: The root object to traverse
            attr_path: Dot-separated attribute path (e.g., "attention.to_q" or "attention.to_out.0")

        Returns:
            The attribute value at the specified path
        """
        attrs = attr_path.split(".")
        for attr in attrs:
            # Handle numeric indices for list access (e.g., "to_out.0")
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj

    @staticmethod
    def save(transformer, path: Path, training_spec: TrainingSpec) -> None:
        """Save LoRA weights to safetensors format."""
        weights = {}
        for entry in tree_flatten(transformer):
            name = entry[0]
            weight = entry[1]
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                weights[name] = weight

        # Transpose and add prefix for compatibility
        weights = {f"transformer.{key}": mx.transpose(val) for key, val in weights.items()}

        # Build metadata
        metadata = {
            "mflux_version": VersionUtil.get_mflux_version(),
            "model": "z-image",
            "training_mode": "lora",
        }

        if training_spec.lora_layers:
            if training_spec.lora_layers.main_layers:
                metadata["main_layers"] = str(training_spec.lora_layers.main_layers)
            if training_spec.lora_layers.noise_refiner:
                metadata["noise_refiner"] = str(training_spec.lora_layers.noise_refiner)
            if training_spec.lora_layers.context_refiner:
                metadata["context_refiner"] = str(training_spec.lora_layers.context_refiner)

        mx.save_safetensors(str(path), weights, metadata=metadata)
