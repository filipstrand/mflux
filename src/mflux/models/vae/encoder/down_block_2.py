import mlx.core as mx
from mlx import nn

from mflux.models.vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.vae.encoder.down_sampler import DownSampler


class DownBlock2(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(norm1=128, conv1_in=128, conv1_out=256, norm2=256, conv2_in=256, conv2_out=256, is_conv_shortcut=True, conv_shortcut_in=128, conv_shortcut_out=256),
            ResnetBlock2D(norm1=256, conv1_in=256, conv1_out=256, norm2=256, conv2_in=256, conv2_out=256),
        ]
        self.downsamplers = [DownSampler(conv_in=256, conv_out=256)]

    def forward(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0].forward(input_array)
        hidden_states = self.resnets[1].forward(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0].forward(hidden_states)

        return hidden_states
