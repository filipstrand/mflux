"""Wan Up Block for FIBO VAE decoder.

Simplified version based on WanUpBlock structure.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_resample import WanResample
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanUpBlock(nn.Module):
    """Upsampling block for decoder.

    Contains residual blocks followed by optional upsampling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        """Initialize up block.

        Args:
            in_dim: Input channels
            out_dim: Output channels
            num_res_blocks: Number of residual blocks
            dropout: Dropout rate (not used)
            upsample_mode: Upsampling mode ('upsample2d', 'upsample3d', or None)
            non_linearity: Activation (always "silu")
        """
        super().__init__()
        # Create residual blocks
        self.resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        # Add upsampling if needed
        self.upsampler = None
        if upsample_mode is not None:
            # CRITICAL: Pass upsample_out_dim=out_dim to match PyTorch behavior
            # PyTorch: WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
            # Without this, MLX defaults to dim // 2, causing channel mismatch
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        """Apply up block.

        Args:
            x: Input tensor
            block_idx: Optional block index for debugging

        Returns:
            Output tensor
        """
        # Apply residual blocks (skip input checkpoint - we know it matches)
        # if block_idx == 0:
        #     from mflux_debugger.tensor_debug import debug_save
        #     debug_save(x, "mlx_up_block_0_input")

        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)
            # Debug checkpoint after each residual block (keep main ones, skip detailed)
            # if block_idx == 0:
            #     from mflux_debugger.tensor_debug import debug_save
            #     debug_save(x, f"mlx_up_block_0_resnet_{i}_after")

        # Apply upsampling if present
        if self.upsampler is not None:
            # Skip before_upsample - we know resnets are working
            # if block_idx == 0:
            #     from mflux_debugger.tensor_debug import debug_save
            #     debug_save(x, "mlx_up_block_0_before_upsample")
            x = self.upsampler(x, block_idx=block_idx)
            # Skip after_upsample - same as resample output
            # if block_idx == 0:
            #     from mflux_debugger.tensor_debug import debug_save
            #     debug_save(x, "mlx_up_block_0_after_upsample")

        return x
