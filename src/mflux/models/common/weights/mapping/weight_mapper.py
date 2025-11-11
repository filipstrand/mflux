"""
Weight Mapper - Applies declarative weight mappings to transform HF weights to MLX structure.

Similar to LoRALoader, but for weight mapping instead of LoRA application.
"""

import re
from typing import Callable, Dict, List, Optional

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class WeightMapper:
    """Maps HuggingFace weights to MLX nested structure using declarative mappings."""

    @staticmethod
    def apply_mapping(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> Dict:
        """
        Apply weight mapping to transform HF weights to MLX structure.

        Args:
            hf_weights: Raw HuggingFace weights (flat dict with dot-notation keys)
            mapping: List of WeightTarget mappings
            num_blocks: Number of transformer blocks (auto-detected if None)
            num_layers: Number of text encoder layers (auto-detected if None)

        Returns:
            Nested dict structure matching MLX model
        """
        # Auto-detect number of blocks if not provided
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)

        # Auto-detect number of layers if not provided
        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)

        # Build flat mapping: HF pattern -> (MLX path, transform)
        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)

        # Map weights
        mapped_weights = {}
        mapped_count = 0
        skipped_count = 0

        for hf_key, hf_tensor in hf_weights.items():
            # Try to find matching mapping
            mlx_path, transform = WeightMapper._find_mapping(hf_key, flat_mapping)

            if mlx_path:
                # Apply transform if specified
                tensor = hf_tensor
                if transform:
                    tensor = transform(tensor)

                # Build nested structure
                WeightMapper._set_nested_value(mapped_weights, mlx_path, tensor)
                mapped_count += 1
            else:
                # Weight not in mapping - might be intentionally skipped (e.g., lm_head)
                # or optional weight (e.g., conv_shortcut) - that's OK
                skipped_count += 1

        # Optional: uncomment for debugging
        # print(f"✅ Mapped {mapped_count} weights, skipped {skipped_count}")

        return mapped_weights

    @staticmethod
    def _detect_num_blocks(hf_weights: Dict[str, mx.array]) -> int:
        """Detect number of transformer blocks from weight keys."""
        block_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: transformer_blocks.{number}.something
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))

        if block_numbers:
            return max(block_numbers) + 1  # Blocks are 0-indexed
        return 0

    @staticmethod
    def _detect_num_layers(hf_weights: Dict[str, mx.array]) -> int:
        """Detect number of text encoder layers from weight keys."""
        layer_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: model.layers.{number}.something
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match:
                layer_numbers.add(int(match.group(1)))

        if layer_numbers:
            return max(layer_numbers) + 1  # Layers are 0-indexed
        return 28  # Default 28 layers for Qwen text encoder

    @staticmethod
    def _build_flat_mapping(
        mapping: List[WeightTarget], num_blocks: int = 0, num_layers: int = 28
    ) -> Dict[str, tuple[str, Optional[Callable[[mx.array], mx.array]]]]:
        """
        Build flat mapping from declarative targets.

        Returns:
            Dict mapping HF pattern -> (MLX path, transform)
        """
        flat = {}

        for target in mapping:
            # Expand placeholders for each pattern
            for hf_pattern in target.hf_patterns:
                # Check which placeholders are present
                has_block = "{block}" in hf_pattern or "{block}" in target.mlx_path
                has_i = "{i}" in hf_pattern or "{i}" in target.mlx_path
                has_res = "{res}" in hf_pattern or "{res}" in target.mlx_path
                has_layer = "{layer}" in hf_pattern or "{layer}" in target.mlx_path

                # Handle multiple placeholders together
                if has_block and has_res:
                    # Up blocks: expand both {block} and {res}
                    max_blocks = num_blocks if num_blocks > 0 else 4  # Default 4 for up_blocks
                    for block_num in range(max_blocks):
                        for res in range(3):  # 3 resnets per up_block
                            concrete_hf = hf_pattern.replace("{block}", str(block_num)).replace("{res}", str(res))
                            concrete_mlx = target.mlx_path.replace("{block}", str(block_num)).replace("{res}", str(res))
                            flat[concrete_hf] = (concrete_mlx, target.transform)
                elif has_block:
                    # Expand {block} only (for transformer blocks or visual blocks)
                    # Check if target has max_blocks override
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    # Check if this is for visual blocks (32 blocks) or transformer blocks
                    elif "visual.blocks" in hf_pattern or "visual.blocks" in target.mlx_path:
                        max_blocks = 32  # Visual blocks are always 32
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 4  # Default 4 for up_blocks
                    for block_num in range(max_blocks):
                        concrete_hf = hf_pattern.replace("{block}", str(block_num))
                        concrete_mlx = target.mlx_path.replace("{block}", str(block_num))
                        flat[concrete_hf] = (concrete_mlx, target.transform)
                elif has_layer:
                    # Expand {layer} for text encoder layers or visual blocks
                    max_layers = num_layers if num_layers > 0 else 28  # Default 28 for text encoder
                    for layer_num in range(max_layers):
                        concrete_hf = hf_pattern.replace("{layer}", str(layer_num))
                        concrete_mlx = target.mlx_path.replace("{layer}", str(layer_num))
                        flat[concrete_hf] = (concrete_mlx, target.transform)
                elif has_i:
                    # Expand {i} only (for mid_block resnets)
                    for i in range(2):  # 2 resnets in mid_block
                        concrete_hf = hf_pattern.replace("{i}", str(i))
                        concrete_mlx = target.mlx_path.replace("{i}", str(i))
                        flat[concrete_hf] = (concrete_mlx, target.transform)
                elif has_res:
                    # This shouldn't happen for VAE (encoder down_blocks are explicit)
                    # But handle it just in case
                    if "up_block" in hf_pattern:
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{res}", str(res))
                            concrete_mlx = target.mlx_path.replace("{res}", str(res))
                            flat[concrete_hf] = (concrete_mlx, target.transform)
                else:
                    # No placeholder, use as-is
                    flat[hf_pattern] = (target.mlx_path, target.transform)

        return flat

    @staticmethod
    def _find_mapping(
        hf_key: str, flat_mapping: Dict[str, tuple[str, Optional[Callable[[mx.array], mx.array]]]]
    ) -> tuple[Optional[str], Optional[Callable[[mx.array], mx.array]]]:
        """Find MLX path and transform for a given HF key."""
        return flat_mapping.get(hf_key, (None, None))

    @staticmethod
    def _set_nested_value(d: Dict, path: str, value: mx.array):
        """
        Set value in nested dict using dot-notation path.

        Creates nested structure as needed.
        Handles both dict keys and list indices.
        """
        parts = path.split(".")
        current = d
        i = 0

        while i < len(parts) - 1:
            part = parts[i]

            # Check if next part is a digit (list index)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                # This is a list, ensure it exists
                if part not in current:
                    current[part] = []
                # Ensure list is large enough
                idx = int(parts[i + 1])
                while len(current[part]) <= idx:
                    current[part].append({})
                current = current[part][idx]
                # Skip both the key and the index
                i += 2
            else:
                # Regular dict key
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        # Set final value
        final_key = parts[-1]
        current[final_key] = value
