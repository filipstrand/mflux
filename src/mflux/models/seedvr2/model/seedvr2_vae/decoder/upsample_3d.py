import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d


class Upsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        temporal_up: bool = False,
    ):
        super().__init__()
        spatial_factor = 2
        temporal_factor = 2 if temporal_up else 1
        total_factor = (spatial_factor**2) * temporal_factor

        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_padding_causal=True,
        )
        self.upscale_conv = CausalConv3d(
            channels,
            channels * total_factor,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        x = self.upscale_conv(x)
        sf = self.spatial_factor
        tf = self.temporal_factor
        x = x.reshape(B, sf, sf, tf, C, T, H, W)
        x = x.transpose(0, 4, 5, 3, 6, 1, 7, 2)
        x = x.reshape(B, C, T * tf, H * sf, W * sf)
        if T == 1 and tf > 1:
            x = x[:, :, :1, :, :]
        x = self.conv(x)
        return x
