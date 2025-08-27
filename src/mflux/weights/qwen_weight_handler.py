"""
QwenImage Weight Handler for mflux

This module handles loading and managing weights for the Qwen Image model.
For now, it provides a simplified structure that can be expanded when we implement
actual weight loading from pretrained models.
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.weights.download import snapshot_download


@dataclass
class QwenImageMetaData:
    """Metadata for Qwen Image model weights."""

    quantization_level: int | None = None
    mflux_version: str | None = None


class QwenImageWeightHandler:
    """
    Weight handler for Qwen Image models.

    Unlike Flux which has separate text encoders (CLIP + T5), Qwen uses a single VL model.
    The architecture consists of:
    - Qwen2.5-VL text encoder (unified vision-language model)
    - QwenImage transformer (dual-stream architecture)
    - QwenImage VAE (3D temporal processing)
    """

    def __init__(
        self,
        meta_data: QwenImageMetaData,
        qwen_text_encoder: dict | None = None,
        transformer: dict | None = None,
        vae: dict | None = None,
    ):
        self.qwen_text_encoder = qwen_text_encoder
        self.transformer = transformer
        self.vae = vae
        self.meta_data = meta_data

    @staticmethod
    def load_pretrained_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "QwenImageWeightHandler":
        """
        Load pretrained Qwen Image weights from the cached repository.

        Args:
            repo_id: Repository ID (e.g., "Qwen/Qwen-Image")
            local_path: Local path to weights (if any)
        Returns:
            QwenImageWeightHandler with loaded weights
        """
        print(f"🔄 Loading Qwen Image weights from: {repo_id or local_path}")

        # Get the weights path (from cache or local)
        if local_path:
            root_path = Path(local_path)
        else:
            # This will use the cached weights without re-downloading
            root_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        "vae/*.safetensors",
                        "transformer/*.safetensors",
                        "text_encoder/*.safetensors",
                    ],
                )
            )

        print(f"📁 Using weights from: {root_path}")

        # Load VAE weights (Phase 1 - only VAE implemented)
        vae_weights = None
        vae_path = root_path / "vae"
        if vae_path.exists():
            print("🔍 Loading VAE weights...")
            vae_weights = QwenImageWeightHandler._load_qwen_vae(vae_path)
            print(f"✅ VAE weights loaded: {len(vae_weights) if vae_weights else 0} parameters")

        # Phase 2: Load transformer weights (organized, not yet applied)
        transformer_weights = None
        transformer_path = root_path / "transformer"
        if transformer_path.exists():
            print("🔍 Loading Transformer weights...")
            transformer_weights = QwenImageWeightHandler._load_qwen_transformer(transformer_path)
            # Count parameters for quick health check
            num_params = QwenImageWeightHandler._count_nested_tensors(transformer_weights)
            print(f"✅ Transformer weights loaded: {num_params} tensors")

        # Phase 3: Load text encoder weights
        text_encoder_weights = None
        text_encoder_path = root_path / "text_encoder"
        if text_encoder_path.exists():
            print("🔍 Loading Text Encoder weights...")
            from mflux.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
            text_encoder_weights = QwenTextEncoderLoader.load_weights(text_encoder_path)
            print(f"✅ Text encoder weights loaded: {len(text_encoder_weights) if text_encoder_weights else 0} parameters")

        return QwenImageWeightHandler(
            meta_data=QwenImageMetaData(
                quantization_level=None,
                mflux_version="dev",
            ),
            qwen_text_encoder=text_encoder_weights,
            transformer=transformer_weights,  # Phase 2: Structured transformer weights
            vae=vae_weights,  # Phase 1: Actual VAE weights loaded
        )

    @staticmethod
    def _load_qwen_vae(vae_path: Path) -> dict:
        """
        Load Qwen VAE weights from the safetensors file.

        Args:
            vae_path: Path to the VAE directory containing safetensors files
        Returns:
            Dictionary containing the processed VAE weights
        """
        # Find the safetensors file
        safetensors_files = list(vae_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {vae_path}")

        # Load the weights
        weights_file = safetensors_files[0]  # Should be diffusion_pytorch_model.safetensors
        print(f"   Loading from: {weights_file.name}")

        # Load using MLX
        data = mx.load(str(weights_file), return_metadata=True)
        weights = dict(data[0].items())

        print(f"   Raw weights loaded: {len(weights)} tensors")

        # Debug: Print first few weight keys and shapes
        print("   🔍 First 5 weight keys and shapes:")
        for i, (key, value) in enumerate(list(weights.items())[:5]):
            print(f"      {i + 1}. {key}: {value.shape}")

        # Map diffusers weight names to our MLX QwenImageVAE structure
        # Instead of generic processing, we need specific mapping for Qwen VAE
        mapped_weights = QwenImageWeightHandler._map_diffusers_to_mlx_vae(weights)

        print(f"   Mapped to MLX structure: {len(mapped_weights)} top-level components")
        return mapped_weights

    @staticmethod
    def _map_diffusers_to_mlx_vae(diffusers_weights: dict) -> dict:
        """
        Map diffusers VAE weight names to our MLX QwenImageVAE structure.

        Diffusers structure:
        - decoder.conv_in.weight -> decoder.conv_in.conv3d.weight
        - decoder.mid_block.resnets.0.conv1.weight -> decoder.mid_block.conv1.conv3d.weight
        - etc.

        Args:
            diffusers_weights: Raw weights from diffusers safetensors
        Returns:
            Dictionary with MLX-compatible structure
        """
        mlx_weights = {}

        print("   🔧 Mapping diffusers weights to MLX structure...")

        # Process both decoder weights and post_quant_conv for Phase 1
        decoder_weights = {}
        vae_weights = {}

        for key, value in diffusers_weights.items():
            # Handle post_quant_conv (not under decoder prefix)
            if key.startswith("post_quant_conv."):
                mlx_key = key  # Keep full key

                # Reshape convolution weights for MLX (post_quant_conv is always conv)
                if "weight" in mlx_key:  # Only transpose weight tensors
                    if len(value.shape) == 5:  # 3D conv: (out_ch, in_ch, d, h, w) -> (out_ch, d, h, w, in_ch)
                        value = mx.transpose(value, (0, 2, 3, 4, 1))
                    elif len(value.shape) == 4:  # 2D conv: (out_ch, in_ch, h, w) -> (out_ch, h, w, in_ch)
                        value = mx.transpose(value, (0, 2, 3, 1))

                # Map post_quant_conv
                if mlx_key == "post_quant_conv.weight":
                    vae_weights["post_quant_conv"] = {"conv3d": {"weight": value}}
                elif mlx_key == "post_quant_conv.bias":
                    if "post_quant_conv" not in vae_weights:
                        vae_weights["post_quant_conv"] = {"conv3d": {}}
                    vae_weights["post_quant_conv"]["conv3d"]["bias"] = value

            elif key.startswith("decoder."):
                # Remove 'decoder.' prefix and map to our structure
                mlx_key = key[8:]  # Remove 'decoder.'

                # Reshape convolution weights for MLX (but NOT norm weights)
                # Check if this is a conv weight (not a norm weight like gamma/beta)
                is_conv_weight = (
                    "weight" in mlx_key
                    and not ("norm" in mlx_key or "gamma" in mlx_key or "beta" in mlx_key)
                    and ("conv" in mlx_key or "to_qkv" in mlx_key or "proj" in mlx_key)
                )

                if is_conv_weight:
                    if len(value.shape) == 5:  # 3D conv: (out_ch, in_ch, d, h, w) -> (out_ch, d, h, w, in_ch)
                        value = mx.transpose(value, (0, 2, 3, 4, 1))
                    elif len(value.shape) == 4:  # 2D conv: (out_ch, in_ch, h, w) -> (out_ch, h, w, in_ch)
                        value = mx.transpose(value, (0, 2, 3, 1))

                # Map specific layer names to our MLX structure
                if mlx_key == "conv_in.weight":
                    if "conv_in" not in decoder_weights:
                        decoder_weights["conv_in"] = {"conv3d": {}}
                    decoder_weights["conv_in"]["conv3d"]["weight"] = value
                elif mlx_key == "conv_in.bias":
                    if "conv_in" not in decoder_weights:
                        decoder_weights["conv_in"] = {"conv3d": {}}
                    decoder_weights["conv_in"]["conv3d"]["bias"] = value
                elif mlx_key == "conv_out.weight":
                    if "conv_out" not in decoder_weights:
                        decoder_weights["conv_out"] = {"conv3d": {}}
                    decoder_weights["conv_out"]["conv3d"]["weight"] = value
                elif mlx_key == "conv_out.bias":
                    if "conv_out" not in decoder_weights:
                        decoder_weights["conv_out"] = {"conv3d": {}}
                    decoder_weights["conv_out"]["conv3d"]["bias"] = value
                # Add post_quant_conv mapping
                elif mlx_key == "post_quant_conv.weight":
                    decoder_weights["post_quant_conv"] = {"conv3d": {"weight": value}}
                elif mlx_key == "post_quant_conv.bias":
                    if "post_quant_conv" not in decoder_weights:
                        decoder_weights["post_quant_conv"] = {"conv3d": {}}
                    decoder_weights["post_quant_conv"]["conv3d"]["bias"] = value
                elif mlx_key.startswith("norm_out."):
                    # Map normalization weights
                    param_name = mlx_key.split(".")[-1]  # 'gamma' or 'beta'
                    if "norm_out" not in decoder_weights:
                        decoder_weights["norm_out"] = {}
                    if param_name == "gamma":
                        decoder_weights["norm_out"]["weight"] = value
                    elif param_name == "beta":
                        decoder_weights["norm_out"]["bias"] = value
                elif mlx_key.startswith("mid_block."):
                    # Map mid_block weights to nested structure matching MLX model
                    QwenImageWeightHandler._map_mid_block_weights_to_lists(mlx_key, value, decoder_weights)
                elif mlx_key.startswith("up_blocks."):
                    # Map up_blocks weights to nested structure matching MLX model
                    QwenImageWeightHandler._map_up_blocks_weights_to_lists(mlx_key, value, decoder_weights)
                else:
                    # For now, store unmapped weights for debugging
                    decoder_weights[mlx_key] = value

        # Create flat weights dictionary for tree_unflatten
        flat_weights = {}

        # Add post_quant_conv weights
        for key, value in vae_weights.items():
            if key == "post_quant_conv":
                flat_weights["post_quant_conv.conv3d.weight"] = value["conv3d"]["weight"]
                if "bias" in value["conv3d"]:
                    flat_weights["post_quant_conv.conv3d.bias"] = value["conv3d"]["bias"]

        # Add decoder weights with flat naming
        for key, value in decoder_weights.items():
            if key.startswith("mid_block"):
                # Skip the nested mid_block structure - we'll handle it differently
                continue
            elif isinstance(value, dict) and "conv3d" in value:
                # Handle conv layers
                flat_weights[f"decoder.{key}.conv3d.weight"] = value["conv3d"]["weight"]
                if "bias" in value["conv3d"]:
                    flat_weights[f"decoder.{key}.conv3d.bias"] = value["conv3d"]["bias"]
            elif isinstance(value, dict):
                # Handle norm layers
                if "weight" in value:
                    flat_weights[f"decoder.{key}.weight"] = value["weight"]
                if "bias" in value:
                    flat_weights[f"decoder.{key}.bias"] = value["bias"]
            else:
                # Direct values
                flat_weights[f"decoder.{key}"] = value

        # Handle mid_block with proper list-based naming for MLX
        mid_block_weights = decoder_weights.get("mid_block", {})
        if "resnets" in mid_block_weights:
            for i, resnet in enumerate(mid_block_weights["resnets"]):
                for component, comp_weights in resnet.items():
                    if isinstance(comp_weights, dict):
                        for param_name, param_value in comp_weights.items():
                            if component in ["conv1", "conv2"]:
                                # Handle the conv3d structure created by the mapping function
                                if param_name == "conv3d" and isinstance(param_value, dict):
                                    # param_value is {"weight": tensor, "bias": tensor}
                                    for sub_param, sub_value in param_value.items():
                                        flat_weights[
                                            f"decoder.mid_block.resnets[{i}].{component}.conv3d.{sub_param}"
                                        ] = sub_value
                                else:
                                    # Direct parameter (shouldn't happen for conv layers after our fix)
                                    flat_weights[f"decoder.mid_block.resnets[{i}].{component}.conv3d.{param_name}"] = (
                                        param_value
                                    )
                            else:  # norm1, norm2
                                flat_weights[f"decoder.mid_block.resnets[{i}].{component}.{param_name}"] = param_value

        if "attentions" in mid_block_weights:
            for i, attention in enumerate(mid_block_weights["attentions"]):
                for component, comp_weights in attention.items():
                    if isinstance(comp_weights, dict):
                        for param_name, param_value in comp_weights.items():
                            if component in ["to_qkv", "proj"]:
                                # Use list index notation for MLX - attention uses 2D convolutions
                                flat_weights[f"decoder.mid_block.attentions[{i}].{component}.{param_name}"] = (
                                    param_value
                                )
                            else:  # norm
                                flat_weights[f"decoder.mid_block.attentions[{i}].{component}.{param_name}"] = (
                                    param_value
                                )

        # Create the proper nested structure with actual Python lists for MLX
        mlx_weights = {}

        # Handle post_quant_conv
        if "post_quant_conv.conv3d.weight" in flat_weights:
            mlx_weights["post_quant_conv"] = {"conv3d": {"weight": flat_weights["post_quant_conv.conv3d.weight"]}}
            if "post_quant_conv.conv3d.bias" in flat_weights:
                mlx_weights["post_quant_conv"]["conv3d"]["bias"] = flat_weights["post_quant_conv.conv3d.bias"]

        # Handle decoder with proper list structures for mid_block
        decoder_weights_final = {}

        # Handle non-mid_block decoder weights
        for key, value in flat_weights.items():
            if key.startswith("decoder.") and "mid_block" not in key:
                # Remove "decoder." prefix
                decoder_key = key[8:]

                # Parse the key to build nested structure
                parts = decoder_key.split(".")
                current = decoder_weights_final

                # Navigate/create nested structure
                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the final value
                current[parts[-1]] = value

        # Handle mid_block with actual Python lists
        mid_block_structure = {}

        # Initialize lists
        max_resnet_idx = -1
        max_attention_idx = -1

        for key in flat_weights.keys():
            if "mid_block.resnets[" in key:
                idx_start = key.find("resnets[") + 8
                idx_end = key.find("]", idx_start)
                idx = int(key[idx_start:idx_end])
                max_resnet_idx = max(max_resnet_idx, idx)
            elif "mid_block.attentions[" in key:
                idx_start = key.find("attentions[") + 11
                idx_end = key.find("]", idx_start)
                idx = int(key[idx_start:idx_end])
                max_attention_idx = max(max_attention_idx, idx)

        # Create lists with proper size
        if max_resnet_idx >= 0:
            mid_block_structure["resnets"] = [{} for _ in range(max_resnet_idx + 1)]
        if max_attention_idx >= 0:
            mid_block_structure["attentions"] = [{} for _ in range(max_attention_idx + 1)]

        # Populate the lists
        for key, value in flat_weights.items():
            if key.startswith("decoder.mid_block."):
                # Parse mid_block keys
                if "resnets[" in key:
                    idx_start = key.find("resnets[") + 8
                    idx_end = key.find("]", idx_start)
                    idx = int(key[idx_start:idx_end])

                    # Get the rest of the path after the index
                    rest = key[idx_end + 2 :]  # +2 to skip "]."
                    parts = rest.split(".")

                    # Navigate to the right place in the resnet
                    current = mid_block_structure["resnets"][idx]
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value

                elif "attentions[" in key:
                    idx_start = key.find("attentions[") + 11
                    idx_end = key.find("]", idx_start)
                    idx = int(key[idx_start:idx_end])

                    # Get the rest of the path after the index
                    rest = key[idx_end + 2 :]  # +2 to skip "]."
                    parts = rest.split(".")

                    # Navigate to the right place in the attention
                    current = mid_block_structure["attentions"][idx]
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value

        if mid_block_structure:
            decoder_weights_final["mid_block"] = mid_block_structure

        # Handle up_blocks with actual Python lists
        for up_block_key, up_block_data in decoder_weights.items():
            if up_block_key.startswith("up_block"):
                up_block_structure = {}

                # Handle resnets as actual Python list
                if "resnets" in up_block_data:
                    # Deep-copy and normalize resnet structures to match model attributes
                    normalized_resnets = []
                    for res in up_block_data["resnets"]:
                        res_norm = {}
                        for comp_name, comp_val in res.items():
                            # Ensure skip_conv uses conv3d container like conv1/conv2
                            if comp_name == "skip_conv":
                                if isinstance(comp_val, dict):
                                    conv3d = comp_val.get("conv3d", {})
                                    # Move flat weight/bias under conv3d if present
                                    if "weight" in comp_val:
                                        conv3d["weight"] = comp_val["weight"]
                                    if "bias" in comp_val:
                                        conv3d["bias"] = comp_val["bias"]
                                    res_norm["skip_conv"] = {"conv3d": conv3d}
                                else:
                                    res_norm["skip_conv"] = comp_val
                            else:
                                res_norm[comp_name] = comp_val
                        normalized_resnets.append(res_norm)
                    up_block_structure["resnets"] = normalized_resnets

                # Handle upsamplers if present
                if "upsamplers" in up_block_data:
                    up_block_structure["upsamplers"] = up_block_data["upsamplers"]

                decoder_weights_final[up_block_key] = up_block_structure

        mlx_weights["decoder"] = decoder_weights_final

        decoder_count = len([k for k in diffusers_weights.keys() if k.startswith("decoder.")])
        post_quant_count = len([k for k in diffusers_weights.keys() if k.startswith("post_quant_conv.")])
        print(f"   ✅ Mapped {decoder_count} decoder weights and {post_quant_count} post_quant_conv weights")
        return mlx_weights

    @staticmethod
    def _count_nested_tensors(d: dict) -> int:
        count = 0
        if isinstance(d, dict):
            for v in d.values():
                count += QwenImageWeightHandler._count_nested_tensors(v)
        elif isinstance(d, list):
            for v in d:
                count += QwenImageWeightHandler._count_nested_tensors(v)
        else:
            try:
                import mlx.core as mx  # local import to avoid cycles

                if isinstance(d, mx.array):
                    return 1
            except Exception:
                pass
        return count

    @staticmethod
    def _load_qwen_transformer(transformer_path: Path) -> dict:
        """
        Load and organize Qwen transformer weights from all safetensor shards.

        Creates a structured dictionary with proper Python lists for blocks and to_out.
        Does NOT transpose linear weights; use assignment consistent with MLX Linear expectations.
        """
        import mlx.core as mx

        # Merge all shards (excluding hidden/metadata files)
        shard_files = sorted([f for f in transformer_path.glob("*.safetensors") if not f.name.startswith("._")])
        if not shard_files:
            raise FileNotFoundError(f"No transformer safetensors found in {transformer_path}")

        flat = {}
        for shard in shard_files:
            data, metadata = mx.load(str(shard), return_metadata=True)
            flat.update(dict(data.items()))

        # Build structured dict
        tf = {}

        def set_linear(tf_parent: dict, name: str, weight_key: str, bias_key: str) -> None:
            if name not in tf_parent:
                tf_parent[name] = {}
            if weight_key in flat:
                tf_parent[name]["weight"] = flat[weight_key]
            if bias_key in flat:
                tf_parent[name]["bias"] = flat[bias_key]

        # Top-level components
        set_linear(tf, "img_in", "img_in.weight", "img_in.bias")

        if "txt_norm.weight" in flat:
            tf.setdefault("txt_norm", {})["weight"] = flat["txt_norm.weight"]
        set_linear(tf, "txt_in", "txt_in.weight", "txt_in.bias")

        # time_text_embed.timestep_embedder
        tte = tf.setdefault("time_text_embed", {}).setdefault("timestep_embedder", {})
        if "time_text_embed.timestep_embedder.linear_1.weight" in flat:
            tte["linear_1"] = {
                "weight": flat["time_text_embed.timestep_embedder.linear_1.weight"],
                "bias": flat["time_text_embed.timestep_embedder.linear_1.bias"],
            }
        if "time_text_embed.timestep_embedder.linear_2.weight" in flat:
            tte["linear_2"] = {
                "weight": flat["time_text_embed.timestep_embedder.linear_2.weight"],
                "bias": flat["time_text_embed.timestep_embedder.linear_2.bias"],
            }

        # Blocks
        blocks = []

        def ensure_block(idx: int) -> dict:
            while len(blocks) <= idx:
                blocks.append({})
            return blocks[idx]

        def ensure_path(d: dict, parts: list[str]):
            cur = d
            for p in parts:
                if p not in cur:
                    cur[p] = {}
                cur = cur[p]
            return cur

        # Iterate over flat keys for transformer_blocks
        prefix = "transformer_blocks."
        for k, v in flat.items():
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix) :]
            # rest example: "58.attn.to_k.weight"
            try:
                block_idx_str, subpath = rest.split(".", 1)
                block_idx = int(block_idx_str)
            except Exception:
                continue
            block = ensure_block(block_idx)

            # Handle to_out.N.* as list
            parts = subpath.split(".")
            if parts[0] == "attn" and parts[1] == "to_out":
                # parts: ["attn","to_out","0","weight"]
                if "to_out" not in block.setdefault("attn", {}):
                    block["attn"]["to_out"] = []
                try:
                    idx = int(parts[2])
                except Exception:
                    continue
                while len(block["attn"]["to_out"]) <= idx:
                    block["attn"]["to_out"].append({})
                block["attn"]["to_out"][idx][parts[3]] = v
                continue

            # Generic nested set for other keys
            # Map path segments directly, e.g., attn.to_q.weight -> block["attn"]["to_q"]["weight"]
            container = block
            for i, p in enumerate(parts[:-1]):
                container = ensure_path(container, [p])
            container[parts[-1]] = v

        if blocks:
            tf["transformer_blocks"] = blocks

        # Output projection
        out = tf.setdefault("output", {})
        # norm_out.linear
        if "norm_out.linear.weight" in flat:
            out.setdefault("norm_out", {})["linear.weight"] = flat["norm_out.linear.weight"]
        if "norm_out.linear.bias" in flat:
            out.setdefault("norm_out", {})["linear.bias"] = flat["norm_out.linear.bias"]
        # proj_out
        set_linear(out, "proj_out", "proj_out.weight", "proj_out.bias")

        return tf

    @staticmethod
    def _map_mid_block_weights_to_lists(mlx_key: str, value: mx.array, decoder_weights: dict):
        """
        Map mid_block weights to structure matching MLX model's list-based architecture.

        MLX model has:
        - self.resnets = [ResNet0, ResNet1, ...]  (list)
        - self.attentions = [Attention0, ...]     (list)

        So we need to create nested dictionaries that match this structure.
        """
        # mlx_key format: "mid_block.resnets.0.norm1.gamma" or "mid_block.attentions.0.norm.gamma"
        parts = mlx_key.split(".")

        if "mid_block" not in decoder_weights:
            decoder_weights["mid_block"] = {"resnets": [], "attentions": []}

        if parts[1] == "resnets":
            # ResNet block: mid_block.resnets.{idx}.{component}.{param}
            resnet_idx = int(parts[2])
            component = parts[3]  # 'norm1', 'conv1', 'norm2', 'conv2'
            param = parts[4]  # 'weight', 'bias', 'gamma', 'beta'

            # Ensure we have enough ResNet dictionaries in the list
            resnets_list = decoder_weights["mid_block"]["resnets"]
            while len(resnets_list) <= resnet_idx:
                resnets_list.append({})

            resnet_dict = resnets_list[resnet_idx]

            # Create component dictionary if needed
            if component not in resnet_dict:
                resnet_dict[component] = {}

            # Map parameter names (gamma -> weight, beta -> bias)
            if param == "gamma":
                resnet_dict[component]["weight"] = value
            elif param == "beta":
                resnet_dict[component]["bias"] = value
            elif component in ["conv1", "conv2", "skip_conv"] and param in ["weight", "bias"]:
                # For conv layers, add the conv3d sub-structure to match MLX model
                # Note: weights were already transposed earlier via is_conv_weight
                if "conv3d" not in resnet_dict[component]:
                    resnet_dict[component]["conv3d"] = {}
                resnet_dict[component]["conv3d"][param] = value
            else:
                resnet_dict[component][param] = value

        elif parts[1] == "attentions":
            # Attention block: mid_block.attentions.{idx}.{component}.{param}
            attn_idx = int(parts[2])
            component = parts[3]  # 'norm', 'to_qkv', 'proj'
            param = parts[4]  # 'weight', 'bias', 'gamma'

            # Ensure we have enough Attention dictionaries in the list
            attentions_list = decoder_weights["mid_block"]["attentions"]
            while len(attentions_list) <= attn_idx:
                attentions_list.append({})

            attention_dict = attentions_list[attn_idx]

            # Create component dictionary if needed
            if component not in attention_dict:
                attention_dict[component] = {}

            # Map parameter names (gamma -> weight, beta -> bias)
            if param == "gamma":
                attention_dict[component]["weight"] = value
            elif param == "beta":
                attention_dict[component]["bias"] = value
            else:
                attention_dict[component][param] = value

    @staticmethod
    def _map_up_blocks_weights_to_lists(mlx_key: str, value: mx.array, decoder_weights: dict):
        """
        Map up_blocks weights to structure matching MLX model's list-based architecture.

        MLX model has:
        - self.up_block0, self.up_block1, self.up_block2, self.up_block3
        Each up_block has:
        - self.res_blocks = [ResNet0, ResNet1, ResNet2]  (list)
        - potentially upsamplers (complex structure)

        Weight key format: "up_blocks.{block_idx}.resnets.{resnet_idx}.{component}.{param}"
        Example: "up_blocks.0.resnets.0.conv1.weight"
        """
        # mlx_key format: "up_blocks.0.resnets.0.conv1.weight" or "up_blocks.0.upsamplers.0.resample.1.weight"
        parts = mlx_key.split(".")

        block_idx = int(parts[1])  # 0, 1, 2, or 3
        block_key = f"up_block{block_idx}"

        # Initialize up_block structure if needed
        if block_key not in decoder_weights:
            decoder_weights[block_key] = {"resnets": []}

        if parts[2] == "resnets":
            # ResNet block: up_blocks.{block_idx}.resnets.{resnet_idx}.{component}.{param}
            resnet_idx = int(parts[3])
            component = parts[4]  # 'norm1', 'conv1', 'norm2', 'conv2', 'conv_shortcut'
            # Align naming with MLX module attribute
            if component == "conv_shortcut":
                component = "skip_conv"
            param = parts[5]  # 'weight', 'bias', 'gamma', 'beta'

            # Ensure we have enough ResNet dictionaries in the list
            resnets_list = decoder_weights[block_key]["resnets"]
            while len(resnets_list) <= resnet_idx:
                resnets_list.append({})

            resnet_dict = resnets_list[resnet_idx]

            # Create component dictionary if needed
            if component not in resnet_dict:
                resnet_dict[component] = {}

            # Map parameter names (gamma -> weight, beta -> bias for norms)
            if param == "gamma":
                resnet_dict[component]["weight"] = value
            elif param == "beta":
                resnet_dict[component]["bias"] = value
            elif component in ["conv1", "conv2"] and param in ["weight", "bias"]:
                # For conv layers, add the conv3d sub-structure to match MLX model
                if "conv3d" not in resnet_dict[component]:
                    resnet_dict[component]["conv3d"] = {}
                resnet_dict[component]["conv3d"][param] = value
            else:
                resnet_dict[component][param] = value

        elif parts[2] == "upsamplers":
            # Upsampler: up_blocks.{block_idx}.upsamplers.{upsampler_idx}.{component}.{param}
            upsampler_idx = int(parts[3])
            remaining_path = ".".join(parts[4:])

            # Initialize upsamplers structure if needed
            if "upsamplers" not in decoder_weights[block_key]:
                decoder_weights[block_key]["upsamplers"] = []

            upsamplers_list = decoder_weights[block_key]["upsamplers"]
            while len(upsamplers_list) <= upsampler_idx:
                upsamplers_list.append({})

            upsampler_dict = upsamplers_list[upsampler_idx]

            # Map upsampler weight paths to MLX structure
            if remaining_path == "resample.1.weight":
                # Map resample.1.weight -> resample_conv.weight
                # PyTorch Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
                # MLX Conv2d: (out_channels, kernel_h, kernel_w, in_channels)
                # Transpose from (192, 384, 3, 3) to (192, 3, 3, 384)
                transposed_weight = mx.transpose(value, (0, 2, 3, 1))
                if "resample_conv" not in upsampler_dict:
                    upsampler_dict["resample_conv"] = {}
                upsampler_dict["resample_conv"]["weight"] = transposed_weight
            elif remaining_path == "resample.1.bias":
                # Map resample.1.bias -> resample_conv.bias
                if "resample_conv" not in upsampler_dict:
                    upsampler_dict["resample_conv"] = {}
                upsampler_dict["resample_conv"]["bias"] = value
            elif remaining_path.startswith("time_conv."):
                # Map time_conv weights to conv3d structure (like ResNet blocks)
                param = remaining_path.split(".")[-1]  # weight or bias
                if "time_conv" not in upsampler_dict:
                    upsampler_dict["time_conv"] = {"conv3d": {}}
                upsampler_dict["time_conv"]["conv3d"][param] = value
            else:
                # Store other weights with their original path
                upsampler_dict[remaining_path] = value

    def num_transformer_blocks(self) -> int:
        """
        Return the number of transformer blocks.
        Based on the diffusers implementation, Qwen uses a different architecture than Flux.

        TODO: Determine the actual number from the pretrained model.
        """
        # Placeholder - will be determined from actual model architecture
        return 24  # This is a guess based on typical transformer sizes

    def has_pretrained_weights(self) -> bool:
        """Check if this handler contains actual pretrained weights or just placeholders."""
        return self.qwen_text_encoder is not None and self.transformer is not None and self.vae is not None
