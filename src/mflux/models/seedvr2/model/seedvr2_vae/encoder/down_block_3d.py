import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.encoder.downsample_3d import Downsample3D
from mflux.models.seedvr2.model.seedvr2_vae.encoder.resnet_block_3d import ResnetBlock3D


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        temporal_down: bool = False,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock3D(in_channels=in_ch, out_channels=out_channels))

        self.downsamplers = []
        if add_downsample:
            self.downsamplers.append(Downsample3D(channels=out_channels, spatial_only=not temporal_down))

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x
