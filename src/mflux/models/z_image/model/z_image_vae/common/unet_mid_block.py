import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.common.attention import Attention
from mflux.models.z_image.model.z_image_vae.common.resnet_block_2d import ResnetBlock2D


class UNetMidBlock(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.attentions = [Attention(channels=channels)]
        self.resnets = [
            ResnetBlock2D(in_channels=channels, out_channels=channels),
            ResnetBlock2D(in_channels=channels, out_channels=channels),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states
