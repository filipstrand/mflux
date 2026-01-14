"""FLUX.2 VAE Decoder with 32 latent channels.

Same architecture as FLUX.1 but accepts 32 input channels.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.common.unet_mid_block import UnetMidBlock
from mflux.models.flux.model.flux_vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.flux.model.flux_vae.decoder.conv_out import ConvOut
from mflux.models.flux.model.flux_vae.decoder.up_block_1_or_2 import UpBlock1Or2
from mflux.models.flux.model.flux_vae.decoder.up_block_3 import UpBlock3
from mflux.models.flux.model.flux_vae.decoder.up_block_4 import UpBlock4


class Flux2ConvIn(nn.Module):
    """Initial convolution for FLUX.2 decoder (32 input channels)."""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=32,  # FLUX.2: 32 channels
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        return mx.transpose(self.conv2d(input_array), (0, 3, 1, 2))


class Flux2Decoder(nn.Module):
    """FLUX.2 VAE Decoder.

    Same architecture as FLUX.1 decoder but with 32-channel input.
    Reuses common components (UpBlocks, MidBlock, etc.) from FLUX.1.
    """

    def __init__(self, enable_tiling: bool = False, split_direction: str = "horizontal"):
        super().__init__()
        self.conv_in = Flux2ConvIn()
        self.mid_block = UnetMidBlock()
        self.up_blocks = [
            UpBlock1Or2(),
            UpBlock1Or2(),
            UpBlock3(),
            UpBlock4(),
        ]
        self.conv_norm_out = ConvNormOut()
        self.conv_out = ConvOut()
        self.enable_tiling = enable_tiling
        self.split_direction = split_direction

    def __call__(self, latents: mx.array) -> mx.array:
        """Decode latents to image.

        Args:
            latents: Latent tensor [B, 32, H, W]

        Returns:
            Decoded image [B, 3, H*8, W*8]
        """
        latents = self.conv_in(latents)
        latents = self.mid_block(latents)
        for i, up_block in enumerate(self.up_blocks):
            latents = self._apply_up_block(i, up_block, latents)
        latents = self.conv_norm_out(latents)
        latents = nn.silu(latents)
        latents = self.conv_out(latents)
        return latents

    def _apply_up_block(
        self,
        block_id: int,
        up_block: nn.Module,
        latents: mx.array,
    ) -> mx.array:
        if self.enable_tiling:
            return Flux2Decoder._apply_up_block_with_tiling(block_id, up_block, latents, self.split_direction)
        else:
            return up_block(latents)

    @staticmethod
    def _apply_up_block_with_tiling(
        block_id: int,
        up_block: nn.Module,
        latents: mx.array,
        split_direction: str = "horizontal",
    ) -> mx.array:
        if block_id == 2:
            latents = Flux2Decoder._process_block_3_in_tiles(latents, up_block, split_direction)
        else:
            latents = up_block(latents)
        return latents

    @staticmethod
    def _process_block_3_in_tiles(
        latents: mx.array, up_block: nn.Module, split_direction: str = "horizontal"
    ) -> mx.array:
        B, C, H, W = latents.shape
        if split_direction == "horizontal":
            return Flux2Decoder._process_horizontal(H, latents, up_block)
        else:
            return Flux2Decoder._process_vertical(W, latents, up_block)

    @staticmethod
    def _process_vertical(width: int, latents: mx.array, up_block: nn.Module) -> mx.array:
        left_tile_input = latents[:, :, :, : width // 2]
        right_tile_input = latents[:, :, :, width // 2 :]
        processed_left_tile = up_block(left_tile_input)
        processed_right_tile = up_block(right_tile_input)
        return mx.concatenate([processed_left_tile, processed_right_tile], axis=3)

    @staticmethod
    def _process_horizontal(height: int, latents: mx.array, up_block: nn.Module) -> mx.array:
        top_tile_input = latents[:, :, : height // 2, :]
        bottom_tile_input = latents[:, :, height // 2 :, :]
        processed_top_tile = up_block(top_tile_input)
        processed_bottom_tile = up_block(bottom_tile_input)
        return mx.concatenate([processed_top_tile, processed_bottom_tile], axis=2)
