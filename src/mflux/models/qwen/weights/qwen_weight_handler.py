from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.models.qwen.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
from mflux.utils.download import snapshot_download


@dataclass
class QwenImageMetaData:
    quantization_level: int | None = None
    mflux_version: str | None = None
    vae_path: str | None = None


class QwenImageWeightHandler:
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
        if local_path:
            root_path = Path(local_path)
        else:
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

        # Load VAE weights (Phase 1 - only VAE implemented)
        vae_weights = None
        vae_path = root_path / "vae"
        if vae_path.exists():
            vae_weights = QwenImageWeightHandler._load_qwen_vae(vae_path)

        # Phase 2: Load transformer weights (organized, not yet applied)
        transformer_weights = None
        transformer_path = root_path / "transformer"
        if transformer_path.exists():
            transformer_weights = QwenImageWeightHandler._load_qwen_transformer(transformer_path)

        # Phase 3: Load text encoder weights
        text_encoder_weights = None
        text_encoder_path = root_path / "text_encoder"
        if text_encoder_path.exists():
            text_encoder_weights = QwenTextEncoderLoader.load_weights(text_encoder_path)

        return QwenImageWeightHandler(
            meta_data=QwenImageMetaData(
                quantization_level=None,
                mflux_version="dev",
                vae_path=str(vae_path) if vae_path.exists() else None,
            ),
            qwen_text_encoder=text_encoder_weights,
            transformer=transformer_weights,
            vae=vae_weights,
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

        return weights

    @staticmethod
    def _load_qwen_transformer(transformer_path: Path) -> dict:
        # Merge all shards (excluding hidden/metadata files)
        shard_files = sorted([f for f in transformer_path.glob("*.safetensors") if not f.name.startswith("._")])
        if not shard_files:
            raise FileNotFoundError(f"No transformer safetensors found in {transformer_path}")

        flat = {}
        for shard in shard_files:
            data, metadata = mx.load(str(shard), return_metadata=True)
            flat.update(dict(data.items()))

        # Use manual Flux-style mapping instead of complex automation
        tf = QwenImageWeightHandler._manual_transformer_mapping(flat)
        return tf

    @staticmethod 
    def _manual_transformer_mapping(diffusers_weights: dict) -> dict:
        weights = {}
        
        # 1. Top-level mappings (exact MLX parameter names)
        weights["img_in"] = {
            "weight": diffusers_weights["img_in.weight"],
            "bias": diffusers_weights["img_in.bias"]
        }
        weights["txt_norm"] = {"weight": diffusers_weights["txt_norm.weight"]}
        weights["txt_in"] = {
            "weight": diffusers_weights["txt_in.weight"], 
            "bias": diffusers_weights["txt_in.bias"]
        }
        
        # 2. Time text embedder (exact MLX structure)
        weights["time_text_embed"] = {"timestep_embedder": {
            "linear_1": {
                "weight": diffusers_weights["time_text_embed.timestep_embedder.linear_1.weight"],
                "bias": diffusers_weights["time_text_embed.timestep_embedder.linear_1.bias"]
            },
            "linear_2": {
                "weight": diffusers_weights["time_text_embed.timestep_embedder.linear_2.weight"],
                "bias": diffusers_weights["time_text_embed.timestep_embedder.linear_2.bias"]
            }
        }}
        
        # 3. Output head (exact MLX structure)
        weights["norm_out"] = {
            "linear": {
                "weight": diffusers_weights["norm_out.linear.weight"],
                "bias": diffusers_weights["norm_out.linear.bias"]
            }
        }
        weights["proj_out"] = {
            "weight": diffusers_weights["proj_out.weight"],
            "bias": diffusers_weights["proj_out.bias"]
        }
        
        # 4. Transformer blocks (exact MLX parameter names - no applier needed!)
        transformer_blocks = []
        for block_idx in range(60):  # 60 blocks based on debug output
            block = {}
            
            # Modulation layers (exact MLX parameter names)
            block["img_mod_linear"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mod.1.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mod.1.bias"]
            }
            block["txt_mod_linear"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mod.1.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mod.1.bias"]
            }
            
            # Attention layers (exact MLX parameter names - no nesting under "attn")
            block["to_q"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_q.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_q.bias"]
            }
            block["to_k"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_k.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_k.bias"]
            }
            block["to_v"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_v.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_v.bias"]
            }
            block["add_q_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_q_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_q_proj.bias"]
            }
            block["add_k_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_k_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_k_proj.bias"]
            }
            block["add_v_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_v_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_v_proj.bias"]
            }
            
            # Attention norms (exact MLX parameter names)
            block["norm_q"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_q.weight"]}
            block["norm_k"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_k.weight"]}
            block["norm_added_q"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_added_q.weight"]}
            block["norm_added_k"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_added_k.weight"]}
            
            # Attention outputs (exact MLX parameter names)
            block["attn_to_out"] = [{
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_out.0.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_out.0.bias"]
            }]
            block["to_add_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_add_out.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_add_out.bias"]
            }
            
            # MLP layers (exact MLX parameter names)
            block["img_mlp_in"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.0.proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.0.proj.bias"]
            }
            block["img_mlp_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.2.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.2.bias"]
            }
            block["txt_mlp_in"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.0.proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.0.proj.bias"]
            }
            block["txt_mlp_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.2.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.2.bias"]
            }
            
            transformer_blocks.append(block)
        
        weights["transformer_blocks"] = transformer_blocks
        return weights
