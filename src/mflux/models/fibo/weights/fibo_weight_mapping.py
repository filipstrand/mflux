from typing import List

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


def reshape_gamma_to_1d(tensor: mx.array) -> mx.array:
    if len(tensor.shape) > 1:
        return mx.reshape(tensor, (tensor.shape[0],))
    return tensor


def transpose_conv3d_weight(tensor: mx.array) -> mx.array:
    if len(tensor.shape) == 5:
        return tensor.transpose(0, 2, 3, 4, 1)
    return tensor


def transpose_conv2d_weight(tensor: mx.array) -> mx.array:
    if len(tensor.shape) == 4:
        return tensor.transpose(0, 2, 3, 1)
    return tensor


class FIBOWeightMapping(WeightMapping):
    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return [
            # ========== Encoder conv_in ==========
            WeightTarget(
                mlx_path="encoder.conv_in.conv3d.weight",
                hf_patterns=["encoder.conv_in.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.conv_in.conv3d.bias",
                hf_patterns=["encoder.conv_in.bias"],
            ),
            # ========== Encoder down_blocks ==========
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.norm1.weight",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv1.conv3d.weight",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv1.conv3d.bias",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.norm2.weight",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv2.conv3d.weight",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv2.conv3d.bias",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.conv3d.weight",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.conv3d.bias",
                hf_patterns=["encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.downsampler.resample_conv.weight",
                hf_patterns=["encoder.down_blocks.{block}.downsampler.resample.1.weight"],
                transform=transpose_conv2d_weight,
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.downsampler.resample_conv.bias",
                hf_patterns=["encoder.down_blocks.{block}.downsampler.resample.1.bias"],
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.downsampler.time_conv.conv3d.weight",
                hf_patterns=["encoder.down_blocks.{block}.downsampler.time_conv.weight"],
                transform=transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                mlx_path="encoder.down_blocks.{block}.downsampler.time_conv.conv3d.bias",
                hf_patterns=["encoder.down_blocks.{block}.downsampler.time_conv.bias"],
                required=False,
            ),
            # ========== Encoder mid_block ==========
            WeightTarget(
                mlx_path="encoder.mid_block.resnets.{i}.norm1.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
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
                mlx_path="encoder.mid_block.resnets.{i}.norm2.weight",
                hf_patterns=["encoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
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
                mlx_path="encoder.mid_block.attentions.{i}.norm.weight",
                hf_patterns=["encoder.mid_block.attentions.{i}.norm.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.{i}.to_qkv.weight",
                hf_patterns=["encoder.mid_block.attentions.{i}.to_qkv.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.{i}.to_qkv.bias",
                hf_patterns=["encoder.mid_block.attentions.{i}.to_qkv.bias"],
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.{i}.proj.weight",
                hf_patterns=["encoder.mid_block.attentions.{i}.proj.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.mid_block.attentions.{i}.proj.bias",
                hf_patterns=["encoder.mid_block.attentions.{i}.proj.bias"],
            ),
            # ========== Encoder output ==========
            WeightTarget(
                mlx_path="encoder.norm_out.weight",
                hf_patterns=["encoder.norm_out.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="encoder.conv_out.conv3d.weight",
                hf_patterns=["encoder.conv_out.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="encoder.conv_out.conv3d.bias",
                hf_patterns=["encoder.conv_out.bias"],
            ),
            # ========== Decoder conv_in ==========
            WeightTarget(
                mlx_path="decoder.conv_in.conv3d.weight",
                hf_patterns=["decoder.conv_in.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.conv_in.conv3d.bias",
                hf_patterns=["decoder.conv_in.bias"],
            ),
            # ========== Decoder mid_block ==========
            # Mid block resnets
            WeightTarget(
                mlx_path="decoder.mid_block.resnets.{i}.norm1.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
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
                mlx_path="decoder.mid_block.resnets.{i}.norm2.weight",
                hf_patterns=["decoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
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
            # Mid block attention
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.{i}.norm.weight",
                hf_patterns=["decoder.mid_block.attentions.{i}.norm.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.{i}.to_qkv.weight",
                hf_patterns=["decoder.mid_block.attentions.{i}.to_qkv.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.{i}.to_qkv.bias",
                hf_patterns=["decoder.mid_block.attentions.{i}.to_qkv.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.{i}.proj.weight",
                hf_patterns=["decoder.mid_block.attentions.{i}.proj.weight"],
                transform=transpose_conv2d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.mid_block.attentions.{i}.proj.bias",
                hf_patterns=["decoder.mid_block.attentions.{i}.proj.bias"],
            ),
            # ========== Decoder up_blocks ==========
            # Up blocks resnets
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.norm1.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.norm1.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv1.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv1.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.norm2.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.norm2.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv2.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv2.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            # Up blocks resnets - conv_shortcut (optional, when in_dim != out_dim)
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=transpose_conv3d_weight,
                required=False,  # Optional - only exists when dimensions differ
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,  # Optional - only exists when dimensions differ
            ),
            # Up blocks upsamplers - time_conv (blocks 0, 1 only)
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.upsampler.time_conv.conv3d.weight",
                hf_patterns=["decoder.up_blocks.{block}.upsampler.time_conv.weight"],
                transform=transpose_conv3d_weight,
                required=False,  # Only exists for blocks 0, 1
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.upsampler.time_conv.conv3d.bias",
                hf_patterns=["decoder.up_blocks.{block}.upsampler.time_conv.bias"],
                required=False,  # Only exists for blocks 0, 1
            ),
            # Up blocks upsamplers - resample_conv (blocks 0, 1, 2)
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.upsampler.resample_conv.weight",
                hf_patterns=["decoder.up_blocks.{block}.upsampler.resample.1.weight"],
                transform=transpose_conv2d_weight,
                required=False,  # Only exists for blocks 0, 1, 2
            ),
            WeightTarget(
                mlx_path="decoder.up_blocks.{block}.upsampler.resample_conv.bias",
                hf_patterns=["decoder.up_blocks.{block}.upsampler.resample.1.bias"],
                required=False,  # Only exists for blocks 0, 1, 2
            ),
            # ========== Decoder output ==========
            WeightTarget(
                mlx_path="decoder.norm_out.weight",
                hf_patterns=["decoder.norm_out.gamma"],
                transform=reshape_gamma_to_1d,
            ),
            WeightTarget(
                mlx_path="decoder.conv_out.conv3d.weight",
                hf_patterns=["decoder.conv_out.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="decoder.conv_out.conv3d.bias",
                hf_patterns=["decoder.conv_out.bias"],
            ),
            # ========== Quant conv ==========
            WeightTarget(
                mlx_path="quant_conv.conv3d.weight",
                hf_patterns=["quant_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="quant_conv.conv3d.bias",
                hf_patterns=["quant_conv.bias"],
            ),
            # ========== Post quant conv ==========
            WeightTarget(
                mlx_path="post_quant_conv.conv3d.weight",
                hf_patterns=["post_quant_conv.weight"],
                transform=transpose_conv3d_weight,
            ),
            WeightTarget(
                mlx_path="post_quant_conv.conv3d.bias",
                hf_patterns=["post_quant_conv.bias"],
            ),
        ]
