import mlx.core as mx
from mlx import nn

from mflux.models.vae.common.unet_mid_block import UnetMidBlock
from mflux.models.vae.decoder.conv_in import ConvIn
from mflux.models.vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.vae.decoder.conv_out import ConvOut
from mflux.models.vae.decoder.up_block_1_or_2 import UpBlock1Or2
from mflux.models.vae.decoder.up_block_3 import UpBlock3
from mflux.models.vae.decoder.up_block_4 import UpBlock4


class Decoder(nn.Module):
    def __init__(self, enable_tiling: bool = False):
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
        self.enable_tiling = enable_tiling

    def __call__(self, latents: mx.array) -> mx.array:
        latents = self.conv_in(latents)
        latents = self.mid_block(latents)
        for i, up_block in enumerate(self.up_blocks):
            latents = self._apply_up_block(i, up_block, latents)
        latents = self.conv_norm_out(latents)
        latents = nn.silu(latents)
        latents = self.conv_out(latents)
        return latents

    def _apply_up_block(self, block_id: int, up_block: nn.Module, latents: mx.array) -> mx.array:
        if self.enable_tiling:
            return Decoder._apply_up_block_with_tiling(block_id, up_block, latents)
        else:
            return up_block(latents)

    @staticmethod
    def _apply_up_block_with_tiling(block_id: int, up_block: nn.Module, latents: mx.array) -> mx.array:
        # Turns out that the third block is the first to cause OOM issues for larger resolutions.
        # This trick is a "lossy optimization" that splits the latents into a top/bottom half and process them separately.
        # The resulting image may have a slightly visible horizontal line at half the image height.
        if block_id == 2:
            latents = Decoder._process_block_3_in_tiles(latents, up_block)
        else:
            latents = up_block(latents)
        return latents

    @staticmethod
    def _process_block_3_in_tiles(latents: mx.array, up_block: nn.Module) -> mx.array:
        B, C, H, W = latents.shape

        # 1. Tile top and bottom
        top_tile_input = latents[:, :, : H // 2, :]
        bottom_tile_input = latents[:, :, H // 2 :, :]

        # 2. Process each tile individually
        processed_top_tile = up_block(top_tile_input)
        processed_bottom_tile = up_block(bottom_tile_input)

        # 3. Concatenate along height
        return mx.concatenate([processed_top_tile, processed_bottom_tile], axis=2)
