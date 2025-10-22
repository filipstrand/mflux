"""
Declarative weight mapping for Qwen models.

This replaces the manual mapping logic in qwen_weight_handler.py with a declarative structure,
similar to how qwen_lora_mapping.py works.
"""

from typing import List

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


def reshape_gamma_to_1d(tensor: mx.array) -> mx.array:
    """Reshape gamma tensor to 1D if needed (for LayerNorm weights)."""
    if len(tensor.shape) > 1:
        return mx.reshape(tensor, (tensor.shape[0],))
    return tensor


def transpose_patch_embed(tensor: mx.array) -> mx.array:
    """Transpose patch embedding weight for MLX Conv3d: (O,I,D,H,W) -> (O,D,H,W,I)."""
    if len(tensor.shape) == 5:
        return tensor.transpose(0, 2, 3, 4, 1)
    return tensor


def transpose_conv3d_weight(tensor: mx.array) -> mx.array:
    """Transpose Conv3d weight for MLX: (O,I,D,H,W) -> (O,D,H,W,I)."""
    if len(tensor.shape) == 5:
        return tensor.transpose(0, 2, 3, 4, 1)
    return tensor


def transpose_conv2d_weight(tensor: mx.array) -> mx.array:
    """Transpose Conv2d weight for MLX: (O,I,H,W) -> (O,H,W,I)."""
    if len(tensor.shape) == 4:
        return tensor.transpose(0, 2, 3, 1)
    return tensor


class QwenWeightMapping(WeightMapping):
    """Declarative weight mappings for Qwen models (transformer + VAE)."""

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """Get weight mappings for transformer component."""
        return [
            # Top-level mappings
            WeightTarget(
                mlx_path="img_in.weight",
                hf_patterns=["img_in.weight"],
            ),
            WeightTarget(
                mlx_path="img_in.bias",
                hf_patterns=["img_in.bias"],
            ),
            WeightTarget(
                mlx_path="txt_norm.weight",
                hf_patterns=["txt_norm.weight"],
            ),
            WeightTarget(
                mlx_path="txt_in.weight",
                hf_patterns=["txt_in.weight"],
            ),
            WeightTarget(
                mlx_path="txt_in.bias",
                hf_patterns=["txt_in.bias"],
            ),
            # Time text embedder
            WeightTarget(
                mlx_path="time_text_embed.timestep_embedder.linear_1.weight",
                hf_patterns=["time_text_embed.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                mlx_path="time_text_embed.timestep_embedder.linear_1.bias",
                hf_patterns=["time_text_embed.timestep_embedder.linear_1.bias"],
            ),
            WeightTarget(
                mlx_path="time_text_embed.timestep_embedder.linear_2.weight",
                hf_patterns=["time_text_embed.timestep_embedder.linear_2.weight"],
            ),
            WeightTarget(
                mlx_path="time_text_embed.timestep_embedder.linear_2.bias",
                hf_patterns=["time_text_embed.timestep_embedder.linear_2.bias"],
            ),
            # Output head
            WeightTarget(
                mlx_path="norm_out.linear.weight",
                hf_patterns=["norm_out.linear.weight"],
            ),
            WeightTarget(
                mlx_path="norm_out.linear.bias",
                hf_patterns=["norm_out.linear.bias"],
            ),
            WeightTarget(
                mlx_path="proj_out.weight",
                hf_patterns=["proj_out.weight"],
            ),
            WeightTarget(
                mlx_path="proj_out.bias",
                hf_patterns=["proj_out.bias"],
            ),
            # Transformer blocks - Attention
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_q.weight",
                hf_patterns=["transformer_blocks.{block}.attn.to_q.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_q.bias",
                hf_patterns=["transformer_blocks.{block}.attn.to_q.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_k.weight",
                hf_patterns=["transformer_blocks.{block}.attn.to_k.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_k.bias",
                hf_patterns=["transformer_blocks.{block}.attn.to_k.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_v.weight",
                hf_patterns=["transformer_blocks.{block}.attn.to_v.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_v.bias",
                hf_patterns=["transformer_blocks.{block}.attn.to_v.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_q_proj.weight",
                hf_patterns=["transformer_blocks.{block}.attn.add_q_proj.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_q_proj.bias",
                hf_patterns=["transformer_blocks.{block}.attn.add_q_proj.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_k_proj.weight",
                hf_patterns=["transformer_blocks.{block}.attn.add_k_proj.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_k_proj.bias",
                hf_patterns=["transformer_blocks.{block}.attn.add_k_proj.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_v_proj.weight",
                hf_patterns=["transformer_blocks.{block}.attn.add_v_proj.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.add_v_proj.bias",
                hf_patterns=["transformer_blocks.{block}.attn.add_v_proj.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_q.weight",
                hf_patterns=["transformer_blocks.{block}.attn.norm_q.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_k.weight",
                hf_patterns=["transformer_blocks.{block}.attn.norm_k.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_added_q.weight",
                hf_patterns=["transformer_blocks.{block}.attn.norm_added_q.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_added_k.weight",
                hf_patterns=["transformer_blocks.{block}.attn.norm_added_k.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.attn_to_out.0.weight",
                hf_patterns=["transformer_blocks.{block}.attn.to_out.0.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.attn_to_out.0.bias",
                hf_patterns=["transformer_blocks.{block}.attn.to_out.0.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_add_out.weight",
                hf_patterns=["transformer_blocks.{block}.attn.to_add_out.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_add_out.bias",
                hf_patterns=["transformer_blocks.{block}.attn.to_add_out.bias"],
            ),
            # Transformer blocks - Modulation
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_mod_linear.weight",
                hf_patterns=["transformer_blocks.{block}.img_mod.1.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_mod_linear.bias",
                hf_patterns=["transformer_blocks.{block}.img_mod.1.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_mod_linear.weight",
                hf_patterns=["transformer_blocks.{block}.txt_mod.1.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_mod_linear.bias",
                hf_patterns=["transformer_blocks.{block}.txt_mod.1.bias"],
            ),
            # Transformer blocks - Feed Forward
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_ff.mlp_in.weight",
                hf_patterns=["transformer_blocks.{block}.img_mlp.net.0.proj.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_ff.mlp_in.bias",
                hf_patterns=["transformer_blocks.{block}.img_mlp.net.0.proj.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_ff.mlp_out.weight",
                hf_patterns=["transformer_blocks.{block}.img_mlp.net.2.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.img_ff.mlp_out.bias",
                hf_patterns=["transformer_blocks.{block}.img_mlp.net.2.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_ff.mlp_in.weight",
                hf_patterns=["transformer_blocks.{block}.txt_mlp.net.0.proj.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_ff.mlp_in.bias",
                hf_patterns=["transformer_blocks.{block}.txt_mlp.net.0.proj.bias"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_ff.mlp_out.weight",
                hf_patterns=["transformer_blocks.{block}.txt_mlp.net.2.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.txt_ff.mlp_out.bias",
                hf_patterns=["transformer_blocks.{block}.txt_mlp.net.2.bias"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Get weight mappings for VAE component."""
        return [
            # ========== Decoder ==========
            # Decoder conv_in
            WeightTarget(
                mlx_path="decoder.conv_in.conv3d.weight",
                hf_patterns=["decoder.conv_in.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.conv_in.conv3d.bias",
                hf_patterns=["decoder.conv_in.bias"],
            ),
            # Decoder conv_out
            WeightTarget(
                mlx_path="decoder.conv_out.conv3d.weight",
                hf_patterns=["decoder.conv_out.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.conv_out.conv3d.bias",
                hf_patterns=["decoder.conv_out.bias"],
            ),
            # Decoder norm_out (gamma -> weight with reshape)
            WeightTarget(
                mlx_path="decoder.norm_out.weight",
                hf_patterns=["decoder.norm_out.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Post quant conv
            WeightTarget(
                mlx_path="post_quant_conv.conv3d.weight",
                hf_patterns=["post_quant_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="post_quant_conv.conv3d.bias",
                hf_patterns=["post_quant_conv.bias"],
            ),
            # Decoder mid_block resnets
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.conv1.conv3d.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.conv1.conv3d.bias",
                hf_patterns=["decoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.conv2.conv3d.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.conv2.conv3d.bias",
                hf_patterns=["decoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.norm1.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.norm2.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Decoder mid_block attention
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.0.norm.weight",
                hf_patterns=["decoder.mid_block.attentions.0.norm.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.0.to_qkv.weight",
                hf_patterns=["decoder.mid_block.attentions.0.to_qkv.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.0.to_qkv.bias",
                hf_patterns=["decoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.0.proj.weight",
                hf_patterns=["decoder.mid_block.attentions.0.proj.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.0.proj.bias",
                hf_patterns=["decoder.mid_block.attentions.0.proj.bias"],
            ),
            # Decoder up_blocks resnets
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.conv1.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.conv1.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.conv2.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.conv2.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.norm1.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.norm2.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Decoder up_blocks optional conv_shortcut -> skip_conv
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.skip_conv.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=transpose_conv3d_weight,
                required=False,  # Optional weight
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.resnets.{res}.skip_conv.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,  # Optional weight
            ),
            # Decoder up_blocks upsamplers (blocks 0, 1, 2)
            WeightTarget(
                mlx_path="decoder.up_block{block}.upsamplers.0.resample_conv.weight",
                hf_patterns=["decoder.up_blocks.{block}.upsamplers.0.resample.1.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.upsamplers.0.resample_conv.bias",
                hf_patterns=["decoder.up_blocks.{block}.upsamplers.0.resample.1.bias"],
            ),
            # Decoder up_blocks time_conv (blocks 0, 1)
            WeightTarget(
                mlx_path="decoder.up_block{block}.upsamplers.0.time_conv.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.upsamplers.0.time_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_block{block}.upsamplers.0.time_conv.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.upsamplers.0.time_conv.bias"],
            ),
            # ========== Encoder ==========
            # Encoder conv_in
            WeightTarget(
                mlx_path="encoder.conv_in.conv3d.weight",
                hf_patterns=["encoder.conv_in.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.conv_in.conv3d.bias",
                hf_patterns=["encoder.conv_in.bias"],
            ),
            # Encoder conv_out
            WeightTarget(
                mlx_path="encoder.conv_out.conv3d.weight",
                hf_patterns=["encoder.conv_out.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.conv_out.conv3d.bias",
                hf_patterns=["encoder.conv_out.bias"],
            ),
            # Encoder norm_out
            WeightTarget(
                mlx_path="encoder.norm_out.weight",
                hf_patterns=["encoder.norm_out.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Encoder mid_block attention
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.0.norm.weight",
                hf_patterns=["encoder.mid_block.attentions.0.norm.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.0.to_qkv.weight",
                hf_patterns=["encoder.mid_block.attentions.0.to_qkv.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.0.to_qkv.bias",
                hf_patterns=["encoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.0.proj.weight",
                hf_patterns=["encoder.mid_block.attentions.0.proj.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.0.proj.bias",
                hf_patterns=["encoder.mid_block.attentions.0.proj.bias"],
            ),
            # Encoder mid_block resnets
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.conv1.conv3d.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.conv1.conv3d.bias",
                hf_patterns=["encoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.conv2.conv3d.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.conv2.conv3d.bias",
                hf_patterns=["encoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.norm1.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.norm2.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Encoder down_blocks - Stage 0 (flat_idx 0,1 -> resnets[0,1]; flat_idx 2 -> downsampler)
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.0.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.0.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.0.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.0.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.norm1.weight",
                hf_patterns=["encoder.down_blocks.0.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.0.norm2.weight",
                hf_patterns=["encoder.down_blocks.0.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.1.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.1.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.1.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.1.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.norm1.weight",
                hf_patterns=["encoder.down_blocks.1.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.resnets.1.norm2.weight",
                hf_patterns=["encoder.down_blocks.1.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.downsamplers.0.resample_conv.weight",
                hf_patterns=["encoder.down_blocks.2.resample.1.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.0.downsamplers.0.resample_conv.bias",
                hf_patterns=["encoder.down_blocks.2.resample.1.bias"],
            ),
            # Encoder down_blocks - Stage 1 (flat_idx 3,4 -> resnets[0,1]; flat_idx 5 -> downsampler)
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.3.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.3.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.3.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.3.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.norm1.weight",
                hf_patterns=["encoder.down_blocks.3.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.norm2.weight",
                hf_patterns=["encoder.down_blocks.3.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.skip_conv.conv3d.weight",
                hf_patterns=["encoder.down_blocks.3.conv_shortcut.weight"],
                transform=transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.0.skip_conv.conv3d.bias",
                hf_patterns=["encoder.down_blocks.3.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.4.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.4.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.4.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.4.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.norm1.weight",
                hf_patterns=["encoder.down_blocks.4.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.resnets.1.norm2.weight",
                hf_patterns=["encoder.down_blocks.4.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.downsamplers.0.resample_conv.weight",
                hf_patterns=["encoder.down_blocks.5.resample.1.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.1.downsamplers.0.resample_conv.bias",
                hf_patterns=["encoder.down_blocks.5.resample.1.bias"],
            ),
            # Encoder down_blocks - Stage 2 (flat_idx 6,7 -> resnets[0,1]; flat_idx 8 -> downsampler)
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.6.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.6.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.6.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.6.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.norm1.weight",
                hf_patterns=["encoder.down_blocks.6.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.norm2.weight",
                hf_patterns=["encoder.down_blocks.6.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.skip_conv.conv3d.weight",
                hf_patterns=["encoder.down_blocks.6.conv_shortcut.weight"],
                transform=transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.0.skip_conv.conv3d.bias",
                hf_patterns=["encoder.down_blocks.6.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.7.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.7.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.7.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.7.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.norm1.weight",
                hf_patterns=["encoder.down_blocks.7.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.resnets.1.norm2.weight",
                hf_patterns=["encoder.down_blocks.7.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.downsamplers.0.resample_conv.weight",
                hf_patterns=["encoder.down_blocks.8.resample.1.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.downsamplers.0.resample_conv.bias",
                hf_patterns=["encoder.down_blocks.8.resample.1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.downsamplers.0.time_conv.conv3d.weight",
                hf_patterns=["encoder.down_blocks.8.time_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.2.downsamplers.0.time_conv.conv3d.bias",
                hf_patterns=["encoder.down_blocks.8.time_conv.bias"],
            ),
            # Encoder down_blocks - Stage 3 (flat_idx 9,10 -> resnets[0,1])
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.9.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.9.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.9.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.9.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.norm1.weight",
                hf_patterns=["encoder.down_blocks.9.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.0.norm2.weight",
                hf_patterns=["encoder.down_blocks.9.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.10.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.10.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.10.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.10.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.norm1.weight",
                hf_patterns=["encoder.down_blocks.10.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.3.resnets.1.norm2.weight",
                hf_patterns=["encoder.down_blocks.10.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            # Quant conv
            WeightTarget(
                mlx_path="quant_conv.conv3d.weight",
                hf_patterns=["quant_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="quant_conv.conv3d.bias",
                hf_patterns=["quant_conv.bias"],
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """Get weight mappings for text encoder component."""
        return [
            # Top-level embeddings
            WeightTarget(
                mlx_path="encoder.embed_tokens.weight",
                hf_patterns=["model.embed_tokens.weight"],
            ),
            # Final norm
            WeightTarget(
                mlx_path="encoder.norm.weight",
                hf_patterns=["model.norm.weight"],
            ),
            # Encoder layers (28 layers)
            WeightTarget(
                mlx_path="encoder.layers.{layer}.input_layernorm.weight",
                hf_patterns=["model.layers.{layer}.input_layernorm.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.post_attention_layernorm.weight",
                hf_patterns=["model.layers.{layer}.post_attention_layernorm.weight"],
            ),
            # Self attention
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.q_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.q_proj.bias",
                hf_patterns=["model.layers.{layer}.self_attn.q_proj.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.k_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.k_proj.bias",
                hf_patterns=["model.layers.{layer}.self_attn.k_proj.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.v_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.v_proj.bias",
                hf_patterns=["model.layers.{layer}.self_attn.v_proj.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.self_attn.o_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.o_proj.weight"],
            ),
            # MLP
            WeightTarget(
                mlx_path="encoder.layers.{layer}.mlp.gate_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.mlp.up_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                mlx_path="encoder.layers.{layer}.mlp.down_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.down_proj.weight"],
            ),
            # Visual weights (optional, only present in Edit models)
            # Patch embedding (with transpose transform)
            WeightTarget(
                mlx_path="encoder.visual.patch_embed.proj.weight",
                hf_patterns=["visual.patch_embed.proj.weight"],
                transform=transpose_patch_embed,
                required=False,
            ),
            # Vision transformer blocks (32 blocks) - use {block} placeholder
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.attn.qkv.weight",
                hf_patterns=["visual.blocks.{block}.attn.qkv.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.attn.qkv.bias",
                hf_patterns=["visual.blocks.{block}.attn.qkv.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.attn.proj.weight",
                hf_patterns=["visual.blocks.{block}.attn.proj.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.attn.proj.bias",
                hf_patterns=["visual.blocks.{block}.attn.proj.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.gate_proj.weight",
                hf_patterns=["visual.blocks.{block}.mlp.gate_proj.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.gate_proj.bias",
                hf_patterns=["visual.blocks.{block}.mlp.gate_proj.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.up_proj.weight",
                hf_patterns=["visual.blocks.{block}.mlp.up_proj.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.up_proj.bias",
                hf_patterns=["visual.blocks.{block}.mlp.up_proj.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.down_proj.weight",
                hf_patterns=["visual.blocks.{block}.mlp.down_proj.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.mlp.down_proj.bias",
                hf_patterns=["visual.blocks.{block}.mlp.down_proj.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.norm1.weight",
                hf_patterns=["visual.blocks.{block}.norm1.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.blocks.{block}.norm2.weight",
                hf_patterns=["visual.blocks.{block}.norm2.weight"],
                required=False,
            ),
            # Patch merger
            WeightTarget(
                mlx_path="encoder.visual.merger.ln_q.weight",
                hf_patterns=["visual.merger.ln_q.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.merger.mlp_0.weight",
                hf_patterns=["visual.merger.mlp.0.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.merger.mlp_0.bias",
                hf_patterns=["visual.merger.mlp.0.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.merger.mlp_1.weight",
                hf_patterns=["visual.merger.mlp.2.weight"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.visual.merger.mlp_1.bias",
                hf_patterns=["visual.merger.mlp.2.bias"],
                required=False,
            ),
        ]

    @staticmethod
    def get_mapping() -> List[WeightTarget]:
        """Get all weight mappings (currently just transformer)."""
        return QwenWeightMapping.get_transformer_mapping()
