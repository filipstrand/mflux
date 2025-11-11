import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_res_block_3d import QwenImageResBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_resample_3d import QwenImageResample3D


class QwenImageDownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, downsample_mode: str = None):
        super().__init__()
        self.resnets = []
        current_dim = in_channels
        for _ in range(num_res_blocks):
            self.resnets.append(QwenImageResBlock3D(current_dim, out_channels))
            current_dim = out_channels

        self.downsamplers = None
        if downsample_mode is not None:
            self.downsamplers = [QwenImageResample3D(out_channels, mode=downsample_mode)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x
