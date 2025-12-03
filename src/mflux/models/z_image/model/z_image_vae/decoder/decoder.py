import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.common.unet_mid_block import UNetMidBlock
from mflux.models.z_image.model.z_image_vae.decoder.conv_in import ConvIn
from mflux.models.z_image.model.z_image_vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.z_image.model.z_image_vae.decoder.conv_out import ConvOut
from mflux.models.z_image.model.z_image_vae.decoder.up_decoder_block import UpDecoderBlock


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = ConvIn(in_channels=16, out_channels=512)
        self.mid_block = UNetMidBlock(channels=512)
        self.up_blocks = [
            UpDecoderBlock(in_channels=512, out_channels=512, num_layers=3, add_upsample=True),
            UpDecoderBlock(in_channels=512, out_channels=512, num_layers=3, add_upsample=True),
            UpDecoderBlock(in_channels=512, out_channels=256, num_layers=3, add_upsample=True),
            UpDecoderBlock(in_channels=256, out_channels=128, num_layers=3, add_upsample=False),
        ]

        self.conv_norm_out = ConvNormOut(channels=128)
        self.conv_out = ConvOut(in_channels=128, out_channels=3)

    def __call__(self, latents: mx.array) -> mx.array:
        hidden_states = self.conv_in(latents)
        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
