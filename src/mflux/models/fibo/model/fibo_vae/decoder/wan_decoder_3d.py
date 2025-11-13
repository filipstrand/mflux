"""Wan 3D Decoder for FIBO VAE.

This implements WanDecoder3d from diffusers AutoencoderKLWan.
For now, using placeholder config values that we'll refine based on testing.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_mid_block import WanMidBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm
from mflux.models.fibo.model.fibo_vae.decoder.wan_up_block import WanUpBlock
from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_save


class WanDecoder3d(nn.Module):
    """3D Decoder for WanVAE.

    Structure: conv_in -> mid_block -> up_blocks -> norm_out -> conv_out

    TODO: Get exact config values from FIBO model:
    - dim (decoder_base_dim)
    - z_dim (latent channels)
    - dim_mult (channel multipliers)
    - num_res_blocks
    - temporal_upsample (which blocks upsample temporally)
    """

    def __init__(
        self,
        dim: int = 256,  # decoder_base_dim from FIBO config
        z_dim: int = 48,  # z_dim from FIBO config
        dim_mult: list[int] = [1, 2, 4, 4],  # dim_mult from FIBO config
        num_res_blocks: int = 2,  # num_res_blocks from FIBO config
        temporal_upsample: list[bool] | None = None,  # None = no temporal upsampling
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 12,  # FIBO outputs 12 channels, not 3
    ):
        """Initialize decoder.

        Args:
            dim: Base number of channels (decoder_base_dim)
            z_dim: Latent space dimension (48 for FIBO)
            dim_mult: Channel multipliers for each block
            num_res_blocks: Number of residual blocks per up block
            temporal_upsample: Which blocks upsample temporally (None = no temporal upsampling)
            dropout: Dropout rate
            non_linearity: Activation function
            out_channels: Output channels (12 for FIBO)
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_upsample = temporal_upsample or []  # Default to empty list if None

        # Compute channel dimensions: reverse dim_mult for decoder (going up)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # Input convolution
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle block
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # Upsample blocks
        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Determine upsampling mode
            # ISSUE: With dim_mult = [1, 2, 4, 4], we have 4 up_blocks but only upsample on 3 of them
            # This gives us 256x256 output, but FIBO needs 512x512 (16x upsampling from 32x32)
            # The diffusers implementation uses `i != len(dim_mult) - 1` which skips the last block
            # TODO: Investigate - maybe FIBO VAE has different config, or last block should upsample?
            # For now, keeping the diffusers logic to match their implementation
            up_flag = i != len(dim_mult) - 1  # Don't upsample on last block (matches diffusers)
            upsample_mode = None
            if up_flag:
                # If temporal_upsample is None or empty, use 2D upsampling only
                if self.temporal_upsample and i < len(self.temporal_upsample) and self.temporal_upsample[i]:
                    upsample_mode = "upsample3d"
                else:
                    upsample_mode = "upsample2d"

            up_block = WanUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

        # Output layers
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Decode latents to image.

        Args:
            x: Input latents of shape (batch, z_dim, time, height, width)
               For FIBO: (batch, 48, 1, height, width)

        Returns:
            Decoded image of shape (batch, out_channels, time, height, width)
            For FIBO: (batch, 12, 1, height, width)
        """
        # Debug checkpoint: decoder input
        debug_checkpoint("mlx_decoder_input", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_input")

        x = self.conv_in(x)
        debug_checkpoint("mlx_decoder_after_conv_in", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_after_conv_in")

        x = self.mid_block(x)
        debug_checkpoint("mlx_decoder_after_mid_block", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_after_mid_block")

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            debug_checkpoint(
                f"mlx_decoder_after_up_block_{i}",
                metadata={"shape": list(x.shape), "dtype": str(x.dtype), "block_idx": i},
            )
            debug_save(x, f"mlx_decoder_after_up_block_{i}")

        x = self.norm_out(x)
        debug_checkpoint("mlx_decoder_after_norm_out", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_after_norm_out")

        x = nn.silu(x)
        debug_checkpoint("mlx_decoder_after_silu", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_after_silu")

        x = self.conv_out(x)
        debug_checkpoint("mlx_decoder_output", metadata={"shape": list(x.shape), "dtype": str(x.dtype)})
        debug_save(x, "mlx_decoder_output")

        return x
