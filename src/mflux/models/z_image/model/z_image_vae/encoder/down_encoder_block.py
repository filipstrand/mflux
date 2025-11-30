import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.z_image.model.z_image_vae.encoder.down_sampler import DownSampler


class DownEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()

        # Create ResNet blocks
        self.resnets = []
        for i in range(num_layers):
            res_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=res_in_channels,
                    out_channels=out_channels,
                )
            )

        if add_downsample:
            self.downsamplers = [DownSampler(in_channels=out_channels, out_channels=out_channels)]
        else:
            self.downsamplers = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)

        return hidden_states
