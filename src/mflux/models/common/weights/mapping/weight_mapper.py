import re
from typing import Callable, Dict, List, Optional

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class WeightMapper:
    @staticmethod
    def apply_mapping(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> Dict:
        # Auto-detect number of blocks if not provided
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)

        # Auto-detect number of layers if not provided
        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)

        # Build flat mapping: HF pattern -> [(MLX path, transform), ...] (supports one-to-many)
        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)

        # Map weights
        mapped_weights = {}
        mapped_count = 0
        skipped_count = 0

        for hf_key, hf_tensor in hf_weights.items():
            # Try to find matching mappings (can be multiple targets for one source)
            targets = flat_mapping.get(hf_key, [])

            if targets:
                for mlx_path, transform in targets:
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
        block_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: transformer_blocks.{number}.something
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))
                continue
            # Match pattern: single_transformer_blocks.{number}.something
            match = re.search(r"single_transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))

        if block_numbers:
            return max(block_numbers) + 1  # Blocks are 0-indexed
        return 0

    @staticmethod
    def _detect_num_layers(hf_weights: Dict[str, mx.array]) -> int:
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
    ) -> Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]]:
        flat: Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]] = {}

        def add_mapping(hf_key: str, mlx_path: str, transform: Optional[Callable[[mx.array], mx.array]]):
            if hf_key not in flat:
                flat[hf_key] = []
            flat[hf_key].append((mlx_path, transform))

        for target in mapping:
            # Expand placeholders for each pattern
            for hf_pattern in target.from_pattern:
                # Check which placeholders are present in BOTH patterns
                hf_has_block = "{block}" in hf_pattern
                to_has_block = "{block}" in target.to_pattern
                has_i = "{i}" in hf_pattern or "{i}" in target.to_pattern
                has_res = "{res}" in hf_pattern or "{res}" in target.to_pattern
                has_layer = "{layer}" in hf_pattern or "{layer}" in target.to_pattern

                # Handle multiple placeholders together
                if (hf_has_block or to_has_block) and has_res:
                    # Up blocks: expand both {block} and {res}
                    max_blocks = num_blocks if num_blocks > 0 else 4  # Default 4 for up_blocks
                    for block_num in range(max_blocks):
                        for res in range(3):  # 3 resnets per up_block
                            concrete_hf = hf_pattern.replace("{block}", str(block_num)).replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{block}", str(block_num)).replace(
                                "{res}", str(res)
                            )
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif hf_has_block and to_has_block:
                    # Both have {block} - standard one-to-one expansion
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    elif "visual.blocks" in hf_pattern or "visual.blocks" in target.to_pattern:
                        max_blocks = 32  # Visual blocks are always 32
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 4
                    for block_num in range(max_blocks):
                        concrete_hf = hf_pattern.replace("{block}", str(block_num))
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif to_has_block and not hf_has_block:
                    # One-to-many: single HF key maps to multiple MLX targets (e.g., relative_attention_bias)
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 24  # Default for T5
                    for block_num in range(max_blocks):
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(hf_pattern, concrete_mlx, target.transform)
                elif has_layer:
                    # Expand {layer} for text encoder layers or visual blocks
                    max_layers = num_layers if num_layers > 0 else 28  # Default 28 for text encoder
                    for layer_num in range(max_layers):
                        concrete_hf = hf_pattern.replace("{layer}", str(layer_num))
                        concrete_mlx = target.to_pattern.replace("{layer}", str(layer_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_i:
                    # Expand {i} only (for mid_block resnets)
                    for i in range(2):  # 2 resnets in mid_block
                        concrete_hf = hf_pattern.replace("{i}", str(i))
                        concrete_mlx = target.to_pattern.replace("{i}", str(i))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_res:
                    # This shouldn't happen for VAE (encoder down_blocks are explicit)
                    # But handle it just in case
                    if "up_block" in hf_pattern:
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{res}", str(res))
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                else:
                    # No placeholder, use as-is
                    add_mapping(hf_pattern, target.to_pattern, target.transform)

        return flat

    @staticmethod
    def _set_nested_value(d: Dict, path: str, value: mx.array):
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
