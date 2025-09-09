from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class LoRALoader:

    @staticmethod
    def load_and_apply_lora(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_files: list[str],
        lora_scales: list[float] | None = None,
    ) -> None:
        if not lora_files:
            return

        # Validate scales - handle both None and empty list cases
        if lora_scales is None or len(lora_scales) == 0:
            lora_scales = [1.0] * len(lora_files)
        elif len(lora_scales) != len(lora_files):
            raise ValueError(
                f"Number of LoRA scales ({len(lora_scales)}) must match number of LoRA files ({len(lora_files)})"
            )

        print(f"📦 Loading {len(lora_files)} LoRA file(s)...")

        for lora_file, scale in zip(lora_files, lora_scales):
            LoRALoader._apply_single_lora(transformer, lora_file, scale, lora_mapping)

        print("✅ All LoRA weights applied successfully")

    @staticmethod
    def _apply_single_lora(
        transformer: nn.Module,
        lora_file: str,
        scale: float,
        lora_mapping: list[LoRATarget]
    ) -> None:
        # Load the LoRA weights
        if not Path(lora_file).exists():
            print(f"❌ LoRA file not found: {lora_file}")
            return

        print(f"🔧 Applying LoRA: {Path(lora_file).name} (scale={scale})")

        try:
            weights = dict(mx.load(lora_file, return_metadata=True)[0].items())
            mx.eval(weights)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to load LoRA file: {e}")
            return

        print(f"🔍 DEBUG: Found {len(weights)} LoRA weights")

        # Apply LoRA using the provided mapping
        flat_mapping = LoRALoader._get_flat_mapping(lora_mapping)
        applied_count = LoRALoader._apply_lora_with_mapping(transformer, weights, scale, flat_mapping)

        print(f"   ✅ Applied to {applied_count} layers")

    @staticmethod
    def _apply_lora_with_mapping(
        transformer: nn.Module,
        weights: dict,
        scale: float,
        lora_mappings: Dict[str, Tuple[str, str, bool]]
    ) -> int:
        applied_count = 0
        lora_data_by_target = {}

        # Group LoRA weights by their target layers
        for weight_key, weight_value in weights.items():
            found_mapping = None
            block_idx = None

            # Pattern matching logic
            for pattern, mapping_info in lora_mappings.items():
                if "{block}" in pattern:
                    # Extract block number from the weight key
                    parts = weight_key.split(".")
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            try:
                                test_block_idx = int(part)
                                concrete_pattern = pattern.format(block=test_block_idx)
                                if weight_key == concrete_pattern:
                                    found_mapping = mapping_info
                                    block_idx = test_block_idx
                                    break
                            except (ValueError, KeyError):
                                continue
                    if found_mapping:
                        break
                else:
                    if weight_key == pattern:
                        found_mapping = mapping_info
                        break

            if found_mapping is None:
                continue

            target_path, matrix_name, transpose = found_mapping

            # Handle block substitution in target path
            if block_idx is not None and "{block}" in target_path:
                target_path = target_path.format(block=block_idx)

            if target_path not in lora_data_by_target:
                lora_data_by_target[target_path] = {}

            lora_data_by_target[target_path][matrix_name] = (weight_value, transpose)

        print(f"🔍 DEBUG: Found {len(lora_data_by_target)} LoRA target layers")

        # Apply LoRA to each target
        for target_path, lora_data in lora_data_by_target.items():
            if LoRALoader._apply_lora_matrices_to_target(transformer, target_path, lora_data, scale):
                applied_count += 1

        return applied_count

    @staticmethod
    def _apply_lora_matrices_to_target(
        transformer: nn.Module,
        target_path: str,
        lora_data: dict,
        scale: float
    ) -> bool:
        # Navigate to the target layer
        current_module = transformer
        path_parts = target_path.split(".")

        try:
            for part in path_parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                else:
                    current_module = getattr(current_module, part)
        except (AttributeError, IndexError, KeyError):
            print(f"❌ Could not find target path: {target_path}")
            return False

        # Check if we have the required matrices
        if "lora_A" not in lora_data or "lora_B" not in lora_data:
            print(f"❌ Missing required LoRA matrices for {target_path}")
            return False

        lora_A, transpose_A = lora_data["lora_A"]
        lora_B, transpose_B = lora_data["lora_B"]

        # Handle transposition
        if transpose_A:
            lora_A = lora_A.T
        if transpose_B:
            lora_B = lora_B.T

        # Handle alpha scaling
        alpha_scale = 1.0
        if "alpha" in lora_data:
            alpha_value, _ = lora_data["alpha"]
            rank = lora_A.shape[1]
            alpha_scale = float(alpha_value) / rank

        # Calculate final scale - only use user scale, matching Diffusers approach
        effective_scale = scale

        # Replace the linear layer with LoRA layer
        if hasattr(current_module, 'weight'):
            original_layer = current_module

            # Create LoRA layer
            lora_layer = LoRALinear.from_linear(
                original_layer,
                r=lora_A.shape[1],
                scale=effective_scale
            )

            # Set the LoRA matrices - use the correct dimensions from the LoRA file
            # This overrides any incorrect dimension calculations in from_linear
            lora_layer.lora_A = lora_A
            lora_layer.lora_B = lora_B

            # Apply alpha scaling to the matrices if present
            if "alpha" in lora_data:
                lora_layer.lora_B = lora_layer.lora_B * alpha_scale

            # Replace the layer in the parent module
            parent_module = transformer
            for part in path_parts[:-1]:
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)

            final_attr = path_parts[-1]
            if final_attr.isdigit():
                parent_module[int(final_attr)] = lora_layer
            else:
                setattr(parent_module, final_attr, lora_layer)

            return True
        else:
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

    @staticmethod
    def _get_flat_mapping(targets: list[LoRATarget]) -> Dict[str, Tuple[str, str, bool]]:
        flat_mapping = {}

        for target in targets:
            # Add up weight patterns (lora_B, transposed)
            for pattern in target.possible_up_patterns:
                flat_mapping[pattern] = (target.model_path, "lora_B", True)

            # Add down weight patterns (lora_A, transposed)
            for pattern in target.possible_down_patterns:
                flat_mapping[pattern] = (target.model_path, "lora_A", True)

            # Add alpha patterns (no transpose)
            for pattern in target.possible_alpha_patterns:
                flat_mapping[pattern] = (target.model_path, "alpha", False)

        return flat_mapping
