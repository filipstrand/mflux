import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = QwenImageRMSNorm(in_channels, images=False)
        self.conv1 = QwenImageCausalConv3D(in_channels, out_channels, 3, 1, 1)
        self.norm2 = QwenImageRMSNorm(out_channels, images=False)
        self.conv2 = QwenImageCausalConv3D(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.skip_conv = QwenImageCausalConv3D(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip_conv = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        return x + residual
