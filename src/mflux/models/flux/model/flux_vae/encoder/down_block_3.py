import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.flux.model.flux_vae.encoder.down_sampler import DownSampler


class DownBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(norm1=256, conv1_in=256, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512, is_conv_shortcut=True, conv_shortcut_in=256, conv_shortcut_out=512),
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=512, norm2=512, conv2_in=512, conv2_out=512),
        ]  # fmt: off
        self.downsamplers = [DownSampler(conv_in=512, conv_out=512)]

    def __call__(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0](input_array)
        hidden_states = self.resnets[1](hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)

        return hidden_states
