import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.z_image.model.z_image_vae.decoder.up_sampler import UpSampler


class UpDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.add_upsample = add_upsample

        # Build ResNet blocks
        resnets = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            use_shortcut = (i == 0) and (in_channels != out_channels)
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_ch,
                    out_channels=out_channels,
                    use_conv_shortcut=use_shortcut,
                )
            )
        self.resnets = resnets

        # Upsampler
        if add_upsample:
            self.upsamplers = [UpSampler(in_channels=out_channels, out_channels=out_channels)]
        else:
            self.upsamplers = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
