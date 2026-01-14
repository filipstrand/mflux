"""FLUX.2 VAE Encoder with 32 latent channels.

Same architecture as FLUX.1 but outputs 64 channels (32 mean + 32 variance).
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.unet_mid_block import UnetMidBlock
from mflux.models.flux.model.flux_vae.encoder.conv_in import ConvIn
from mflux.models.flux.model.flux_vae.encoder.conv_norm_out import ConvNormOut
from mflux.models.flux.model.flux_vae.encoder.down_block_1 import DownBlock1
from mflux.models.flux.model.flux_vae.encoder.down_block_2 import DownBlock2
from mflux.models.flux.model.flux_vae.encoder.down_block_3 import DownBlock3
from mflux.models.flux.model.flux_vae.encoder.down_block_4 import DownBlock4


class Flux2ConvOut(nn.Module):
    """Final convolution for FLUX.2 encoder (64 output channels = 32 mean + 32 var)."""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=512,
            out_channels=64,  # FLUX.2: 32 mean + 32 variance
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        return mx.transpose(self.conv2d(input_array), (0, 3, 1, 2))


class Flux2Encoder(nn.Module):
    """FLUX.2 VAE Encoder.

    Same architecture as FLUX.1 encoder but outputs 64 channels (32 mean + 32 variance).
    Reuses common components (DownBlocks, MidBlock, etc.) from FLUX.1.
    """

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
        self.conv_out = Flux2ConvOut()

    def __call__(self, image: mx.array) -> mx.array:
        """Encode image to latent distribution parameters.

        Args:
            image: Image tensor [B, 3, H, W]

        Returns:
            Latent distribution parameters [B, 64, H//8, W//8]
            (first 32 channels = mean, last 32 = log variance)
        """
        latents = self.conv_in(image)
        for down_block in self.down_blocks:
            latents = down_block(latents)
        latents = self.mid_block(latents)
        latents = self.conv_norm_out(latents)
        latents = nn.silu(latents)
        latents = self.conv_out(latents)
        return latents
