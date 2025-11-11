import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
from mflux.models.qwen.model.qwen_vae.qwen_image_up_block_3d import QwenImageUpBlock3D


class QwenImageDecoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = QwenImageCausalConv3D(16, 384, 3, 1, 1)
        self.mid_block = QwenImageMidBlock3D(384, num_layers=1)
        self.up_block0 = QwenImageUpBlock3D(384, 384, num_res_blocks=2, upsample_mode="upsample3d")
        self.up_block1 = QwenImageUpBlock3D(192, 384, num_res_blocks=2, upsample_mode="upsample3d")
        self.up_block2 = QwenImageUpBlock3D(192, 192, num_res_blocks=2, upsample_mode="upsample2d")
        self.up_block3 = QwenImageUpBlock3D(96, 96, num_res_blocks=2, upsample_mode=None)
        self.norm_out = QwenImageRMSNorm(96, images=False)
        self.conv_out = QwenImageCausalConv3D(96, 3, 3, 1, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        x = self.up_block0(x)
        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.up_block3(x)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
