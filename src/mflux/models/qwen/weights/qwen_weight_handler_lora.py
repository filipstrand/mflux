import mlx.core as mx
import mlx.nn as nn

from mflux.models.qwen.weights.qwen_weight_handler import MetaData, QwenWeightHandler


class QwenLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear | nn.QuantizedLinear, r: int = 16, scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.scale = scale

        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        # Initialize LoRA matrices
        std = 1.0 / (input_dims**0.5)
        self.lora_A = mx.random.uniform(low=-std, high=std, shape=(input_dims, r))
        self.lora_B = mx.random.uniform(low=-std, high=std, shape=(r, output_dims))

    def __call__(self, x):
        base_out = self.linear(x)
        # LoRA computation: x @ lora_A.T @ lora_B.T (transpose to match expected dimensions)
        lora_out = mx.matmul(mx.matmul(x, self.lora_A.T), self.lora_B.T)
        return base_out + self.scale * lora_out


class QwenWeightHandlerLoRA:
    def __init__(self, weight_handlers: list[QwenWeightHandler]):
        self.weight_handlers = weight_handlers

    @staticmethod
    def load_lora_weights(
        transformer: nn.Module,
        lora_files: list[str],
        lora_scales: list[float] | None = None,
    ) -> list["QwenWeightHandler"]:
        lora_weights = []
        if not lora_files:
            return lora_weights

        lora_scales = QwenWeightHandlerLoRA._validate_lora_scales(lora_files, lora_scales)

        for lora_file, lora_scale in zip(lora_files, lora_scales):
            try:
                # Load LoRA weights from safetensors file
                weights_data = mx.load(lora_file, return_metadata=True)
                weights = dict(weights_data[0].items())

                print(f"\n🔄 Loading LoRA: {lora_file}")
                print(f"📏 Scale: {lora_scale}")
                print(f"📊 Total weight tensors loaded: {len(weights)}")

                # Print first 10 weight keys to see the structure
                print("🔍 First 10 weight keys:")
                for i, (key, value) in enumerate(list(weights.items())[:10]):
                    print(f"  {i + 1:2d}. {key}: {value.shape}")

                if len(weights) > 10:
                    print(f"  ... and {len(weights) - 10} more weights")

                # Process and create LoRA layers for Qwen transformer
                transformer_weights = QwenWeightHandlerLoRA._create_qwen_lora_layers(weights, transformer, lora_scale)

                qwen_weights = QwenWeightHandler(
                    meta_data=MetaData(
                        quantization_level=None,
                        mflux_version="dev",
                    ),
                    qwen_text_encoder=None,
                    vae=None,
                    transformer=transformer_weights,
                )
                lora_weights.append(qwen_weights)

            except (FileNotFoundError, ValueError, KeyError) as e:  # noqa: PERF203
                print(f"Failed to load LoRA from {lora_file}: {e}")
                continue

        return lora_weights

    @staticmethod
    def set_lora_weights(transformer: nn.Module, loras: list["QwenWeightHandler"]) -> None:
        if not loras:
            return

        print(f"Applying {len(loras)} LoRA weights to Qwen transformer")

        # For simplicity, we'll apply each LoRA sequentially
        # In practice, you might want to fuse multiple LoRAs
        for lora_handler in loras:
            QwenWeightHandlerLoRA._apply_lora_to_transformer(transformer, lora_handler)

    @staticmethod
    def _create_qwen_lora_layers(weights: dict, transformer: nn.Module, scale: float) -> dict:
        print(f"\n🔧 Creating LoRA layers for Qwen transformer (scale: {scale})")
        transformer_dict = {}
        processed_count = 0
        skipped_count = 0

        # Get the actual number of blocks in the transformer
        max_blocks = len(transformer.transformer_blocks) if hasattr(transformer, "transformer_blocks") else 48
        print(f"📋 Transformer has {max_blocks} blocks (0-{max_blocks - 1})")

        # Look for transformer block weights
        for key, value in weights.items():
            if "transformer_blocks." in key:
                parts = key.split(".")
                if len(parts) >= 3:
                    try:
                        block_idx = int(parts[1])  # transformer_blocks.{idx}.{layer}
                        layer_path = ".".join(parts[2:])  # remaining path

                        # Skip invalid block indices
                        if block_idx >= max_blocks:
                            print(f"  ⚠️  Skipping Block {block_idx}: OUT OF BOUNDS (max is {max_blocks - 1})")
                            skipped_count += 1
                            continue

                        print(f"  📍 Processing: Block {block_idx}, Layer: {layer_path}, Shape: {value.shape}")

                        # Initialize block structure
                        block_key = f"transformer_blocks.{block_idx}"
                        if block_key not in transformer_dict:
                            transformer_dict[block_key] = {}

                        # Handle different layer types
                        if "attn" in layer_path:
                            QwenWeightHandlerLoRA._process_attention_lora(
                                transformer_dict[block_key], layer_path, value, transformer, block_idx, scale
                            )
                            processed_count += 1
                        elif "ff" in layer_path or "mlp" in layer_path:
                            QwenWeightHandlerLoRA._process_ff_lora(
                                transformer_dict[block_key], layer_path, value, transformer, block_idx, scale
                            )
                            processed_count += 1
                        else:
                            print(f"    ⚠️  Skipped unknown layer type: {layer_path}")
                            skipped_count += 1

                    except (ValueError, IndexError) as e:
                        print(f"    ❌ Skipped malformed key {key}: {e}")
                        skipped_count += 1
                        continue
            else:
                print(f"  ⏩ Skipping non-transformer weight: {key}")
                skipped_count += 1

        print(f"✅ LoRA layer creation complete: {processed_count} processed, {skipped_count} skipped")
        print(f"🏗️  Created transformer blocks: {list(transformer_dict.keys())}")

        return {"transformer": transformer_dict}

    @staticmethod
    def _process_attention_lora(
        block_dict: dict,
        layer_path: str,
        value: mx.array,
        transformer: nn.Module,
        block_idx: int,
        scale: float,
    ):
        if "attn" not in block_dict:
            block_dict["attn"] = {}

        print(f"    🔍 Processing attention LoRA: {layer_path}")

        # Extract up and down matrices (LoRA naming convention)
        if layer_path.endswith(".lora_down.weight"):
            base_key = layer_path[:-17]  # Remove .lora_down.weight
            block_dict["attn"][f"{base_key}_lora_down"] = value
            print(f"      💾 Stored lora_down for {base_key}: {value.shape}")

        elif layer_path.endswith(".lora_up.weight"):
            base_key = layer_path[:-15]  # Remove .lora_up.weight
            block_dict["attn"][f"{base_key}_lora_up"] = value
            print(f"      💾 Stored lora_up for {base_key}: {value.shape}")

            # Try to create LoRA layer if we have both up and down
            lora_down_key = f"{base_key}_lora_down"
            if lora_down_key in block_dict["attn"]:
                lora_A = block_dict["attn"][lora_down_key]  # down = A
                lora_B = value  # up = B

                print(f"      🏗️  Creating LoRA layer for {base_key}")
                print(f"         A shape: {lora_A.shape}, B shape: {lora_B.shape}")

                try:
                    # Determine the original layer
                    original_layer = QwenWeightHandlerLoRA._get_attention_layer(transformer, block_idx, base_key)

                    if original_layer is not None:
                        rank = lora_A.shape[1]
                        lora_layer = QwenLoRALinear(linear=original_layer, r=rank, scale=scale)
                        lora_layer.lora_A = lora_A
                        lora_layer.lora_B = lora_B

                        # Store the LoRA layer with the full base_key for proper mapping
                        block_dict["attn"][base_key] = lora_layer

                        print(f"      ✅ Created LoRA layer: {base_key} (rank={rank}, scale={scale})")

                        # Clean up temporary storage
                        if lora_down_key in block_dict["attn"]:
                            del block_dict["attn"][lora_down_key]
                        if f"{base_key}_lora_up" in block_dict["attn"]:
                            del block_dict["attn"][f"{base_key}_lora_up"]

                    else:
                        print(f"      ❌ Original layer not found for {base_key}")

                except (AttributeError, ValueError) as e:
                    print(f"      ❌ Failed to create attention LoRA for {base_key}: {e}")
            else:
                print(f"      ⏳ Waiting for lora_down to pair with lora_up for {base_key}")

    @staticmethod
    def _process_ff_lora(
        block_dict: dict,
        layer_path: str,
        value: mx.array,
        transformer: nn.Module,
        block_idx: int,
        scale: float,
    ):
        if "ff" not in block_dict:
            block_dict["ff"] = {}

        print(f"    🔧 Processing FF LoRA: {layer_path}")

        # Similar logic to attention layers
        if layer_path.endswith(".lora_down.weight"):
            base_key = layer_path[:-17]  # Remove .lora_down.weight
            block_dict["ff"][f"{base_key}_lora_down"] = value
            print(f"      💾 Stored lora_down for {base_key}: {value.shape}")

        elif layer_path.endswith(".lora_up.weight"):
            base_key = layer_path[:-15]  # Remove .lora_up.weight
            block_dict["ff"][f"{base_key}_lora_up"] = value
            print(f"      💾 Stored lora_up for {base_key}: {value.shape}")

            lora_down_key = f"{base_key}_lora_down"
            if lora_down_key in block_dict["ff"]:
                lora_A = block_dict["ff"][lora_down_key]  # down = A
                lora_B = value  # up = B

                print(f"      🏗️  Creating LoRA layer for {base_key}")
                print(f"         A shape: {lora_A.shape}, B shape: {lora_B.shape}")

                try:
                    original_layer = QwenWeightHandlerLoRA._get_ff_layer(transformer, block_idx, base_key)

                    if original_layer is not None:
                        rank = lora_A.shape[1]
                        lora_layer = QwenLoRALinear(linear=original_layer, r=rank, scale=scale)
                        lora_layer.lora_A = lora_A
                        lora_layer.lora_B = lora_B

                        # Store the LoRA layer with the full base_key for proper mapping
                        block_dict["ff"][base_key] = lora_layer

                        print(f"      ✅ Created LoRA layer: {base_key} (rank={rank}, scale={scale})")

                        # Clean up temporary storage
                        if lora_down_key in block_dict["ff"]:
                            del block_dict["ff"][lora_down_key]
                        if f"{base_key}_lora_up" in block_dict["ff"]:
                            del block_dict["ff"][f"{base_key}_lora_up"]

                    else:
                        print(f"      ❌ Original layer not found for {base_key}")

                except (AttributeError, ValueError) as e:
                    print(f"      ❌ Failed to create FF LoRA for {base_key}: {e}")
            else:
                print(f"      ⏳ Waiting for lora_down to pair with lora_up for {base_key}")

    @staticmethod
    def _get_attention_layer(transformer: nn.Module, block_idx: int, layer_path: str):
        try:
            block = transformer.transformer_blocks[block_idx]

            # Map LoRA layer names to actual Qwen transformer attribute names
            if layer_path.startswith("attn.to_q"):
                return getattr(block, "to_q", None)
            elif layer_path.startswith("attn.to_k"):
                return getattr(block, "to_k", None)
            elif layer_path.startswith("attn.to_v"):
                return getattr(block, "to_v", None)
            elif layer_path.startswith("attn.add_q_proj"):
                return getattr(block, "add_q_proj", None)
            elif layer_path.startswith("attn.add_k_proj"):
                return getattr(block, "add_k_proj", None)
            elif layer_path.startswith("attn.add_v_proj"):
                return getattr(block, "add_v_proj", None)
            elif layer_path.startswith("attn.to_out.0"):
                return getattr(block, "attn_to_out", [None])[0] if hasattr(block, "attn_to_out") else None
            elif layer_path.startswith("attn.to_add_out"):
                return getattr(block, "to_add_out", None)

        except (AttributeError, IndexError):
            pass

        return None

    @staticmethod
    def _get_ff_layer(transformer: nn.Module, block_idx: int, layer_path: str):
        try:
            block = transformer.transformer_blocks[block_idx]

            # Map LoRA FF layer names to actual Qwen transformer attribute names
            if layer_path.startswith("img_mlp.net.0.proj"):
                return getattr(block, "img_mlp_in", None)
            elif layer_path.startswith("img_mlp.net.2"):
                return getattr(block, "img_mlp_out", None)
            elif layer_path.startswith("txt_mlp.net.0.proj"):
                return getattr(block, "txt_mlp_in", None)
            elif layer_path.startswith("txt_mlp.net.2"):
                return getattr(block, "txt_mlp_out", None)

        except (AttributeError, IndexError):
            pass

        return None

    @staticmethod
    def _apply_lora_to_transformer(transformer: nn.Module, lora_handler: QwenWeightHandler):
        print("\n🔗 Applying LoRA weights to transformer")

        if not lora_handler.transformer:
            print("❌ No transformer weights found in LoRA handler")
            return

        transformer_weights = lora_handler.transformer
        applied_count = 0
        failed_count = 0

        print(f"🔍 Found transformer blocks: {list(transformer_weights.keys())}")

        # Handle nested transformer structure
        if "transformer" in transformer_weights:
            transformer_blocks = transformer_weights["transformer"]
            print(f"🔍 Found nested transformer blocks: {list(transformer_blocks.keys())}")
        else:
            transformer_blocks = transformer_weights

        for block_key, block_weights in transformer_blocks.items():
            if "transformer_blocks." in block_key:
                print(f"  🎯 Processing block {block_key}: {list(block_weights.keys())}")
                try:
                    block_idx = int(block_key.split(".")[-1])
                    block = transformer.transformer_blocks[block_idx]

                    print(f"  🎯 Applying LoRA to block {block_idx}")

                    # Apply attention LoRAs
                    if "attn" in block_weights:
                        print(f"    🔍 Attention layers found: {list(block_weights['attn'].keys())}")
                        for layer_name, lora_layer in block_weights["attn"].items():
                            if isinstance(lora_layer, QwenLoRALinear):
                                # Map LoRA layer names to actual Qwen transformer attribute names
                                actual_attr_name = QwenWeightHandlerLoRA._map_attention_layer_name(layer_name)
                                if actual_attr_name and hasattr(block, actual_attr_name):
                                    # Special case for attn_to_out which is a list
                                    if actual_attr_name == "attn_to_out":
                                        original_layer = block.attn_to_out[0]
                                        lora_layer.linear = original_layer
                                        block.attn_to_out[0] = lora_layer
                                    else:
                                        original_layer = getattr(block, actual_attr_name)
                                        # Replace with LoRA layer
                                        lora_layer.linear = original_layer  # Set original layer reference
                                        setattr(block, actual_attr_name, lora_layer)
                                    applied_count += 1
                                    print(f"      ✅ Applied LoRA to {actual_attr_name}")
                                    print(f"         Original: {type(original_layer).__name__}")
                                    print(
                                        f"         LoRA: A={lora_layer.lora_A.shape}, B={lora_layer.lora_B.shape}, scale={lora_layer.scale}"
                                    )
                                else:
                                    print(
                                        f"      ❌ Cannot map layer {layer_name} to block attribute (mapped to: {actual_attr_name})"
                                    )
                                    failed_count += 1
                            else:
                                print(f"      ⚠️  Non-LoRA layer found: {layer_name} ({type(lora_layer)})")

                    # Apply FF LoRAs
                    if "ff" in block_weights:
                        print(f"    🔧 FF layers found: {list(block_weights['ff'].keys())}")
                        for layer_name, lora_layer in block_weights["ff"].items():
                            if isinstance(lora_layer, QwenLoRALinear):
                                # Map LoRA layer names to actual Qwen transformer attribute names
                                actual_attr_name = QwenWeightHandlerLoRA._map_ff_layer_name(layer_name)
                                if actual_attr_name and hasattr(block, actual_attr_name):
                                    original_layer = getattr(block, actual_attr_name)
                                    # Replace with LoRA layer
                                    lora_layer.linear = original_layer  # Set original layer reference
                                    setattr(block, actual_attr_name, lora_layer)
                                    applied_count += 1
                                    print(f"      ✅ Applied LoRA to {actual_attr_name}")
                                    print(f"         Original: {type(original_layer).__name__}")
                                    print(
                                        f"         LoRA: A={lora_layer.lora_A.shape}, B={lora_layer.lora_B.shape}, scale={lora_layer.scale}"
                                    )
                                else:
                                    print(
                                        f"      ❌ Cannot map layer {layer_name} to block attribute (mapped to: {actual_attr_name})"
                                    )
                                    failed_count += 1
                            else:
                                print(f"      ⚠️  Non-LoRA layer found: {layer_name} ({type(lora_layer)})")

                except (ValueError, IndexError, AttributeError) as e:
                    print(f"    ❌ Failed to apply LoRA to block {block_key}: {e}")
                    failed_count += 1

        print(f"🎉 LoRA application complete: {applied_count} applied, {failed_count} failed")

    @staticmethod
    def _map_attention_layer_name(lora_layer_name: str) -> str | None:
        """Map LoRA attention layer names to actual Qwen transformer attribute names."""
        # Remove any _lora_up/_lora_down suffixes to get base name
        base_name = lora_layer_name.replace("_lora_up", "").replace("_lora_down", "")

        # Remove "attn." prefix if present (we'll add it back for nested structure)
        if base_name.startswith("attn."):
            base_name = base_name[5:]  # Remove "attn."

        # Map LoRA names to nested Qwen transformer layer names (now under attn.)
        mapping = {
            "to_q": "attn.to_q",
            "to_k": "attn.to_k",
            "to_v": "attn.to_v",
            "add_q_proj": "attn.add_q_proj",
            "add_k_proj": "attn.add_k_proj",
            "add_v_proj": "attn.add_v_proj",
            "to_out.0": "attn.attn_to_out",  # Special case: this is a list in Qwen
            "to_add_out": "attn.to_add_out",
        }

        return mapping.get(base_name)

    @staticmethod
    def _map_ff_layer_name(lora_layer_name: str) -> str | None:
        """Map LoRA FF layer names to actual Qwen transformer attribute names."""
        # Remove any _lora_up/_lora_down suffixes to get base name
        base_name = lora_layer_name.replace("_lora_up", "").replace("_lora_down", "")

        # Map LoRA names to nested Qwen transformer layer names (now under img_ff/txt_ff)
        mapping = {
            "img_mlp.net.0.proj": "img_ff.mlp_in",
            "img_mlp.net.2": "img_ff.mlp_out",
            "txt_mlp.net.0.proj": "txt_ff.mlp_in",
            "txt_mlp.net.2": "txt_ff.mlp_out",
        }

        return mapping.get(base_name)

    @staticmethod
    def _validate_lora_scales(lora_files: list[str], lora_scales: list[float] | None) -> list[float]:
        if lora_scales is None:
            return [1.0] * len(lora_files)

        if len(lora_files) == 1:
            if len(lora_scales) > 1:
                raise ValueError("Please provide a single scale for the LoRA, or skip it to default to 1")
            return lora_scales if lora_scales else [1.0]

        elif len(lora_files) > 1:
            if len(lora_files) != len(lora_scales):
                raise ValueError("When providing multiple LoRAs, be sure to specify a scale for each one respectively")
            return lora_scales

        return [1.0]
