import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.attention_3d import Attention3D
from mflux.models.seedvr2.model.seedvr2_vae.decoder.decoder_resnet_block_3d import ResnetBlock3D


class MidBlock3D(nn.Module):
    def __init__(
        self,
        channels: int = 512,
    ):
        super().__init__()
        self.attentions = [Attention3D(channels=channels)]
        self.resnets = [
            ResnetBlock3D(
                in_channels=channels,
                out_channels=channels,
            ),
            ResnetBlock3D(
                in_channels=channels,
                out_channels=channels,
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x
