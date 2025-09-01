import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.flux.model.flux_vae.decoder.up_sampler import UpSampler


class UpBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(norm1=512, conv1_in=512, conv1_out=256, norm2=256, conv2_in=256, conv2_out=256, is_conv_shortcut=True, conv_shortcut_in=512, conv_shortcut_out=256),
            ResnetBlock2D(norm1=256, conv1_in=256, conv1_out=256, norm2=256, conv2_in=256, conv2_out=256),
            ResnetBlock2D(norm1=256, conv1_in=256, conv1_out=256, norm2=256, conv2_in=256, conv2_out=256),
        ]  # fmt: off
        self.upsamplers = [UpSampler(conv_in=256, conv_out=256)]

    def __call__(self, input_array: mx.array) -> mx.array:
        hidden_states = self.resnets[0](input_array)
        hidden_states = self.resnets[1](hidden_states)
        hidden_states = self.resnets[2](hidden_states)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states
