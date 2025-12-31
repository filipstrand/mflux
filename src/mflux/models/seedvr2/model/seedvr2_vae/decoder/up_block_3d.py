import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.decoder.decoder_resnet_block_3d import ResnetBlock3D
from mflux.models.seedvr2.model.seedvr2_vae.decoder.upsample_3d import Upsample3D


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        temporal_up: bool = False,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                )
            )

        self.upsamplers = []
        if add_upsample:
            self.upsamplers.append(Upsample3D(channels=out_channels, temporal_up=temporal_up))

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x
