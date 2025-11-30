import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.common.unet_mid_block import UNetMidBlock
from mflux.models.z_image.model.z_image_vae.encoder.conv_in import ConvIn
from mflux.models.z_image.model.z_image_vae.encoder.conv_norm_out import ConvNormOut
from mflux.models.z_image.model.z_image_vae.encoder.conv_out import ConvOut
from mflux.models.z_image.model.z_image_vae.encoder.down_encoder_block import DownEncoderBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = ConvIn(in_channels=3, out_channels=128)
        self.down_blocks = [
            DownEncoderBlock(in_channels=128, out_channels=128, num_layers=2, add_downsample=True),
            DownEncoderBlock(in_channels=128, out_channels=256, num_layers=2, add_downsample=True),
            DownEncoderBlock(in_channels=256, out_channels=512, num_layers=2, add_downsample=True),
            DownEncoderBlock(in_channels=512, out_channels=512, num_layers=2, add_downsample=False),
        ]
        self.mid_block = UNetMidBlock(channels=512)
        self.conv_norm_out = ConvNormOut(num_channels=512, num_groups=32)
        self.conv_out = ConvOut(in_channels=512, out_channels=32)  # 2 * 16 latent channels

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        return sample
