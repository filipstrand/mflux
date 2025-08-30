import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_down_block_3d import QwenImageDownBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageEncoder3D(nn.Module):

    def __init__(
        self,
        dim=96,
        z_dim=32,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=[2, 2, 2, 2],
        attn_scales=[],
        temporal_downsample=[False, False, True, True],
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.dropout = dropout

        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = QwenImageCausalConv3D(3, dims[0], 3, 1, 1)

        down_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < len(self.temporal_downsample):
                downsample_mode = "downsample3d" if self.temporal_downsample[i] else "downsample2d"
            else:
                downsample_mode = "downsample2d"
            # Do not downsample on the final stage only (match diffusers)
            if i == len(dims) - 2:
                downsample_mode = None
            # Use per-stage num_res_blocks
            stage_res_blocks = num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks
            down_block = QwenImageDownBlock3D(
                in_dim, out_dim,
                num_res_blocks=stage_res_blocks,
                downsample_mode=downsample_mode
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        self.mid_block = QwenImageMidBlock3D(dims[-1], num_layers=1)
        self.norm_out = QwenImageRMSNorm(dims[-1], images=False)  # Encoder norm_out uses images=False
        self.conv_out = QwenImageCausalConv3D(dims[-1], 32, 3, 1, 1) # Changed z_dim to 32

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
