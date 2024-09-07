import mlx.core as mx
from mlx import nn

from mflux.models.vae.decoder.conv_in import ConvIn
from mflux.models.vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.vae.decoder.conv_out import ConvOut
from mflux.models.vae.common.unet_mid_block import UnetMidBlock
from mflux.models.vae.decoder.up_block_1_or_2 import UpBlock1Or2
from mflux.models.vae.decoder.up_block_3 import UpBlock3
from mflux.models.vae.decoder.up_block_4 import UpBlock4


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_in = ConvIn()
        self.mid_block = UnetMidBlock()
        self.up_blocks = [
            UpBlock1Or2(),
            UpBlock1Or2(),
            UpBlock3(),
            UpBlock4(),
        ]
        self.conv_norm_out = ConvNormOut()
        self.conv_out = ConvOut()

    def decode(self, latents: mx.array) -> mx.array:
        latents = self.conv_in.forward(latents)
        latents = self.mid_block.forward(latents)
        for up_block in self.up_blocks:
            latents = up_block.forward(latents)
        latents = self.conv_norm_out.forward(latents)
        latents = nn.silu(latents)
        latents = self.conv_out.forward(latents)
        return latents
