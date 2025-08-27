import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.attention import Attention
from mflux.models.flux.model.flux_vae.common.resnet_block_2d import ResnetBlock2D


class UnetMidBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = [Attention()]
        self.resnets = [
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512),
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512),
        ]

    def __call__(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0](input_array)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states
