import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.unet_mid_block import UnetMidBlock
from mflux.models.flux.model.flux_vae.encoder.conv_in import ConvIn
from mflux.models.flux.model.flux_vae.encoder.conv_norm_out import ConvNormOut
from mflux.models.flux.model.flux_vae.encoder.conv_out import ConvOut
from mflux.models.flux.model.flux_vae.encoder.down_block_1 import DownBlock1
from mflux.models.flux.model.flux_vae.encoder.down_block_2 import DownBlock2
from mflux.models.flux.model.flux_vae.encoder.down_block_3 import DownBlock3
from mflux.models.flux.model.flux_vae.encoder.down_block_4 import DownBlock4


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = ConvIn()
        self.mid_block = UnetMidBlock()
        self.down_blocks = [
            DownBlock1(),
            DownBlock2(),
            DownBlock3(),
            DownBlock4(),
        ]
        self.conv_norm_out = ConvNormOut()
        self.conv_out = ConvOut()

    def __call__(self, latents: mx.array) -> mx.array:
        latents = self.conv_in(latents)
        for down_block in self.down_blocks:
            latents = down_block(latents)
        latents = self.mid_block(latents)
        latents = self.conv_norm_out(latents)
        latents = nn.silu(latents)
        latents = self.conv_out(latents)
        return latents
