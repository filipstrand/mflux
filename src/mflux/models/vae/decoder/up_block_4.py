import mlx.core as mx
from mlx import nn

from mflux.models.vae.common.resnet_block_2d import ResnetBlock2D


class UpBlock4(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(norm1=256, conv1_in=256, conv1_out=128, norm2=128, conv2_in=128, conv2_out=128, is_conv_shortcut=True, conv_shortcut_in=256, conv_shortcut_out=128),
            ResnetBlock2D(norm1=128, conv1_in=128, conv1_out=128, norm2=128, conv2_in=128, conv2_out=128),
            ResnetBlock2D(norm1=128, conv1_in=128, conv1_out=128, norm2=128, conv2_in=128, conv2_out=128),
        ]

    def forward(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0].forward(input_array)
        hidden_states = self.resnets[1].forward(hidden_states)
        hidden_states = self.resnets[2].forward(hidden_states)
        return hidden_states
