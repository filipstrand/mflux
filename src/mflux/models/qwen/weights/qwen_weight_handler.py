"""
QwenImage Weight Handler for mflux

This module handles loading and managing weights for the Qwen Image model.
For now, it provides a simplified structure that can be expanded when we implement
actual weight loading from pretrained models.
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.models.qwen.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
from mflux.utils.download import snapshot_download


@dataclass
class QwenImageMetaData:
    """Metadata for Qwen Image model weights."""

    quantization_level: int | None = None
    mflux_version: str | None = None
    vae_path: str | None = None  # Path to VAE directory for critical weight fixes


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
            text_encoder_weights = QwenTextEncoderLoader.load_weights(text_encoder_path)
            print(
                f"✅ Text encoder weights loaded: {len(text_encoder_weights) if text_encoder_weights else 0} parameters"
            )

        return QwenImageWeightHandler(
            meta_data=QwenImageMetaData(
                quantization_level=None,
                mflux_version="dev",
                vae_path=str(vae_path) if vae_path.exists() else None,
            ),
            qwen_text_encoder=text_encoder_weights,
            transformer=transformer_weights,  # Phase 2: Structured transformer weights
            vae=vae_weights,  # Phase 1: Actual VAE weights loaded
        )

    @staticmethod
    def _load_qwen_vae(vae_path: Path) -> dict:
        safetensors_files = list(vae_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {vae_path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        weights = dict(data[0].items())
        mapped_weights = QwenImageWeightHandler._manual_flux_style_mapping(weights)
        return mapped_weights

    @staticmethod
    def _manual_flux_style_mapping(diffusers_weights: dict) -> dict:
        weights = {}
        
        # 1. Simple direct mappings (like Flux does)
        weights["decoder"] = {}
        
        # conv_in: decoder.conv_in.weight -> decoder.conv_in.conv3d.weight
        weights["decoder"]["conv_in"] = {"conv3d": {
            "weight": diffusers_weights["decoder.conv_in.weight"],
            "bias": diffusers_weights["decoder.conv_in.bias"]
        }}
        
        # conv_out: decoder.conv_out.weight -> decoder.conv_out.conv3d.weight  
        weights["decoder"]["conv_out"] = {"conv3d": {
            "weight": diffusers_weights["decoder.conv_out.weight"],
            "bias": diffusers_weights["decoder.conv_out.bias"]
        }}
        
        # norm_out: decoder.norm_out.gamma -> decoder.norm_out.weight
        weights["decoder"]["norm_out"] = {
            "weight": diffusers_weights["decoder.norm_out.gamma"]
        }
        
        # post_quant_conv: post_quant_conv.weight -> post_quant_conv.conv3d.weight
        weights["post_quant_conv"] = {"conv3d": {
            "weight": diffusers_weights["post_quant_conv.weight"],
            "bias": diffusers_weights["post_quant_conv.bias"]
        }}
        
        # 2. Mid block (manual structure building)
        weights["decoder"]["mid_block"] = {}
        
        # mid_block.resnets (list of 2 resnets)
        weights["decoder"]["mid_block"]["resnets"] = [{}, {}]
        for i in range(2):
            resnet = weights["decoder"]["mid_block"]["resnets"][i]
            # conv1.weight -> conv1.conv3d.weight
            resnet["conv1"] = {"conv3d": {
                "weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.weight"],
                "bias": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.bias"]
            }}
            # conv2.weight -> conv2.conv3d.weight
            resnet["conv2"] = {"conv3d": {
                "weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv2.weight"],
                "bias": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv2.bias"]
            }}
            # norm1.gamma -> norm1.weight
            resnet["norm1"] = {"weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.norm1.gamma"]}
            resnet["norm2"] = {"weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.norm2.gamma"]}
        
        # mid_block.attentions (list of 1 attention)
        weights["decoder"]["mid_block"]["attentions"] = [{}]
        attn = weights["decoder"]["mid_block"]["attentions"][0]
        attn["norm"] = {"weight": diffusers_weights["decoder.mid_block.attentions.0.norm.gamma"]}
        # Note: to_qkv and proj are Conv2d, not Conv3d - no conv3d wrapper
        attn["to_qkv"] = {
            "weight": diffusers_weights["decoder.mid_block.attentions.0.to_qkv.weight"],
            "bias": diffusers_weights["decoder.mid_block.attentions.0.to_qkv.bias"]
        }
        attn["proj"] = {
            "weight": diffusers_weights["decoder.mid_block.attentions.0.proj.weight"],
            "bias": diffusers_weights["decoder.mid_block.attentions.0.proj.bias"]
        }
        
        # 3. Up blocks (manual structure building)
        for block_idx in range(4):
            up_block_key = f"up_block{block_idx}"
            weights["decoder"][up_block_key] = {}
            
            # resnets (list of 3 resnets)
            weights["decoder"][up_block_key]["resnets"] = [{}, {}, {}]
            for res_idx in range(3):
                resnet = weights["decoder"][up_block_key]["resnets"][res_idx]
                
                # conv1.weight -> conv1.conv3d.weight
                resnet["conv1"] = {"conv3d": {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv1.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv1.bias"]
                }}
                # conv2.weight -> conv2.conv3d.weight
                resnet["conv2"] = {"conv3d": {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv2.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv2.bias"]
                }}
                # norm1.gamma -> norm1.weight
                resnet["norm1"] = {"weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.norm1.gamma"]}
                resnet["norm2"] = {"weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.norm2.gamma"]}
                
                # Handle optional conv_shortcut -> skip_conv (only exists for some resnets)
                shortcut_key = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv_shortcut.weight"
                if shortcut_key in diffusers_weights:
                    resnet["skip_conv"] = {"conv3d": {
                        "weight": diffusers_weights[shortcut_key],
                        "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv_shortcut.bias"]
                    }}
            
            # upsamplers (only for blocks 0, 1, 2)
            if block_idx <= 2:
                weights["decoder"][up_block_key]["upsamplers"] = [{}]
                upsampler = weights["decoder"][up_block_key]["upsamplers"][0]
                
                # resample.1.weight -> resample_conv.weight (Conv2d, no conv3d wrapper)
                upsampler["resample_conv"] = {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.resample.1.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.resample.1.bias"]
                }
                
                # time_conv (only for blocks 0, 1)
                if block_idx <= 1:
                    upsampler["time_conv"] = {"conv3d": {
                        "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.time_conv.weight"],
                        "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.time_conv.bias"]
                    }}
        
        print("   ✨ Manual Flux-style mapping complete!")
        return weights

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
                # Keep weights in original PyTorch format - transposes handled in computation
                if "resample_conv" not in upsampler_dict:
                    upsampler_dict["resample_conv"] = {}
                upsampler_dict["resample_conv"]["weight"] = value
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
