import mlx.core as mx
from mlx import nn

from flux_1.models.vae.common.resnet_block_2d import ResnetBlock2D


class DownBlock4(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512),
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512),
        ]

    def forward(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0].forward(input_array)
        hidden_states = self.resnets[1].forward(hidden_states)
        return hidden_states
