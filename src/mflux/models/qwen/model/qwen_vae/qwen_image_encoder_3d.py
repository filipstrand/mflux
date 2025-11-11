import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_down_block_3d import QwenImageDownBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 96
        self.z_dim = 32
        self.dim_mult = [1, 2, 4, 4]
        self.num_res_blocks = [2, 2, 2, 2]
        self.attn_scales = []
        self.temporal_downsample = [False, False, True, True]
        self.dropout = 0.0

        dims = [self.dim * u for u in [1] + self.dim_mult]
        self.conv_in = QwenImageCausalConv3D(3, dims[0], 3, 1, 1)

        down_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            downsample_mode = "downsample3d" if self.temporal_downsample[i] else "downsample2d"
            # Do not downsample on the final stage only
            if i == len(dims) - 2:
                downsample_mode = None
            # Use per-stage num_res_blocks
            stage_res_blocks = self.num_res_blocks[i] if isinstance(self.num_res_blocks, list) else self.num_res_blocks
            down_block = QwenImageDownBlock3D(
                in_dim, out_dim, num_res_blocks=stage_res_blocks, downsample_mode=downsample_mode
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        self.mid_block = QwenImageMidBlock3D(dims[-1], num_layers=1)
        self.norm_out = QwenImageRMSNorm(dims[-1], images=False)
        self.conv_out = QwenImageCausalConv3D(dims[-1], 32, 3, 1, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for stage_idx, down_block in enumerate(self.down_blocks):
            if stage_idx == 3:
                for resnet in down_block.resnets:
                    residual = x
                    n1 = resnet.norm1(x)
                    a1 = nn.silu(n1)
                    c1 = resnet.conv1(a1)
                    n2 = resnet.norm2(c1)
                    a2 = nn.silu(n2)
                    c2 = resnet.conv2(a2)
                    x = c2 + residual
                if down_block.downsamplers is not None:
                    x = down_block.downsamplers[0](x)
            else:
                x = down_block(x)

        x = self.mid_block(x)
        norm_in = x
        x = self.norm_out(norm_in)
        x = nn.silu(x)
        encoded = self.conv_out(x)
        return encoded
