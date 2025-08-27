"""
Weight loader for Qwen text encoder from safetensor files.
"""

import json
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import load_file

from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder


class QwenTextEncoderLoader:
    """Loads Qwen text encoder weights from safetensor files."""

    @staticmethod
    def load_weights(text_encoder_path: Path) -> dict[str, mx.array]:
        """
        Load all text encoder weights from safetensor files.

        Args:
            text_encoder_path: Path to flux_text_encoder directory

        Returns:
            Dictionary mapping MLX parameter names to weight arrays
        """
        print("🔍 Loading Qwen text encoder weights...")

        # Load the index to find which file contains which weights
        index_path = text_encoder_path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        # Load weights from each file
        all_weights = {}
        for file_name, param_names in files_to_load.items():
            file_path = text_encoder_path / file_name
            print(f"   Loading {file_name} ({len(param_names)} parameters)...")

            # Load the safetensor file
            try:
                file_weights = load_file(str(file_path))
            except Exception as e:
                # If MLX can't load directly, try with torch and convert
                print(f"   MLX loading failed, trying torch conversion: {e}")
                import torch
                from safetensors.torch import load_file as torch_load_file

                torch_weights = torch_load_file(str(file_path))
                file_weights = {}
                for name, tensor in torch_weights.items():
                    # Convert to float32 if bfloat16, then to MLX
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    file_weights[name] = mx.array(tensor.numpy())

            # Add to combined weights
            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        print(f"✅ Loaded {len(all_weights)} text encoder parameters")

        # Convert to MLX parameter naming
        mlx_weights = QwenTextEncoderLoader._convert_to_mlx_names(all_weights)

        return mlx_weights

    @staticmethod
    def _convert_to_mlx_names(hf_weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """
        Convert HuggingFace parameter names to MLX parameter names.

        HF: model.embed_tokens.weight -> MLX: encoder.embed_tokens.weight
        HF: model.layers.0.input_layernorm.weight -> MLX: encoder.layers.0.input_layernorm.weight
        etc.
        """
        print("🔄 Converting parameter names from HF to MLX format...")

        mlx_weights = {}
        converted_count = 0
        skipped_count = 0

        for hf_name, weight in hf_weights.items():
            # Skip LM head and vision encoder weights - we only need the text encoder
            if hf_name.startswith("lm_head") or hf_name.startswith("visual."):
                skipped_count += 1
                continue

            # Convert model.* to encoder.*
            if hf_name.startswith("model."):
                mlx_name = hf_name.replace("model.", "encoder.")

                # HF stores linear weights as (out_features, in_features)
                # MLX nn.Linear expects weights in the same format (out_features, in_features)
                # and does the transposition internally with weight.T during computation
                # So we keep weights as-is from HF format
                mlx_weights[mlx_name] = weight
                converted_count += 1
            else:
                # Keep other names as-is (shouldn't happen for text encoder)
                print(f"   ⚠️  Unexpected parameter name: {hf_name}")
                mlx_weights[hf_name] = weight
                converted_count += 1

        print(f"   ✅ Converted {converted_count} parameters")
        print(f"   ⏭️  Skipped {skipped_count} parameters (LM head + vision encoder)")

        return mlx_weights

    @staticmethod
    def convert_to_nested_dict(flat_weights: dict[str, mx.array]) -> dict:
        """
        Convert flat parameter names to nested dictionary structure.
        Handle special case for layers which should be lists in MLX.

        Example:
        'encoder.embed_tokens.weight' -> {'encoder': {'embed_tokens': {'weight': array}}}
        'encoder.layers.0.input_layernorm.weight' -> {'encoder': {'layers': [{'input_layernorm': {'weight': array}}]}}
        """
        nested = {}

        # First pass: collect all parameters
        for param_name, param_value in flat_weights.items():
            parts = param_name.split(".")

            # Navigate/create the nested structure
            current = nested
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    # Check if this part is a layer index (numeric and previous part is 'layers')
                    if part.isdigit() and i > 0 and parts[i - 1] == "layers":
                        # Initialize as empty dict for now, we'll convert to list later
                        current[part] = {}
                    else:
                        current[part] = {}
                current = current[part]

            # Set the final value
            current[parts[-1]] = param_value

        # Second pass: convert layers dictionaries to lists
        def convert_layers_to_lists(obj):
            if isinstance(obj, dict):
                # Check if this dict has 'layers' key with numeric string keys
                if "layers" in obj and isinstance(obj["layers"], dict):
                    layers_dict = obj["layers"]
                    # Check if all keys are numeric strings
                    if all(k.isdigit() for k in layers_dict.keys()):
                        # Convert to list
                        max_idx = max(int(k) for k in layers_dict.keys())
                        layers_list = [None] * (max_idx + 1)
                        for k, v in layers_dict.items():
                            layers_list[int(k)] = convert_layers_to_lists(v)
                        obj["layers"] = layers_list
                        return obj

                # Recursively process other dict values
                for k, v in obj.items():
                    obj[k] = convert_layers_to_lists(v)

            return obj

        nested = convert_layers_to_lists(nested)
        return nested

    @staticmethod
    def apply_weights(text_encoder: QwenTextEncoder, text_encoder_path: Path) -> None:
        """
        Load and apply weights to a QwenTextEncoder instance.

        Args:
            text_encoder: The MLX text encoder to load weights into
            text_encoder_path: Path to the flux_text_encoder directory with safetensor files
        """
        print("📝 Applying Qwen text encoder weights...")

        # Load weights
        weights = QwenTextEncoderLoader.load_weights(text_encoder_path)

        # Apply weights to the model using update() with nested structure (like Flux)
        try:
            print("🔧 Converting flat weights to nested structure...")
            nested_weights = QwenTextEncoderLoader.convert_to_nested_dict(weights)

            print("🔧 Applying weights with update(strict=False)...")
            text_encoder.update(nested_weights, strict=False)
            print("✅ Text encoder weights applied successfully")

            # Verify some weights were loaded
            embed_weights = text_encoder.encoder.embed_tokens.weight
            print(
                f"   📊 Embedding weights: mean={mx.mean(embed_weights).item():.6f}, std={mx.std(embed_weights).item():.6f}"
            )

            # Check if weights look reasonable (not all zeros)
            if mx.all(embed_weights == 0):
                print("⚠️  WARNING: All embedding weights are zero - weights may not have loaded correctly")
            else:
                print("✅ Weights appear to have loaded correctly")

        except Exception as e:
            print(f"❌ Error applying weights: {e}")
            raise

    @staticmethod
    def verify_weight_shapes(text_encoder: QwenTextEncoder, text_encoder_path: Path) -> bool:
        """
        Verify that the weight shapes match between saved weights and model.

        Returns:
            True if all shapes match, False otherwise
        """
        print("🔍 Verifying weight shapes...")

        weights = QwenTextEncoderLoader.load_weights(text_encoder_path)

        # Get model parameter shapes
        model_params = text_encoder.parameters()

        mismatches = []
        matches = 0

        for name, saved_weight in weights.items():
            # Navigate to the parameter using the name path
            try:
                # Split name into parts: encoder.layers.0.self_attn.q_proj.weight
                parts = name.split(".")
                current = text_encoder
                for part in parts:
                    if part.isdigit():
                        current = current[int(part)]
                    else:
                        current = getattr(current, part)

                model_weight = current
                if saved_weight.shape != model_weight.shape:
                    mismatches.append(f"{name}: saved {saved_weight.shape} vs model {model_weight.shape}")
                else:
                    matches += 1

            except (AttributeError, IndexError, KeyError):
                mismatches.append(f"{name}: not found in model")

        if mismatches:
            print(f"❌ Found {len(mismatches)} shape mismatches:")
            for mismatch in mismatches[:10]:  # Show first 10
                print(f"   {mismatch}")
            if len(mismatches) > 10:
                print(f"   ... and {len(mismatches) - 10} more")
            return False
        else:
            print(f"✅ All {matches} weight shapes match correctly")
            return True
