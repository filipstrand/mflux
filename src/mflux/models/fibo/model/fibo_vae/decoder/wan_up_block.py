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
        self.resnets = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        # Add upsampling if needed
        self.upsampler = None
        if upsample_mode is not None:
            self.upsampler = WanResample(out_dim, mode=upsample_mode)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply up block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Apply residual blocks
        for resnet in self.resnets:
            x = resnet(x)

        # Apply upsampling if present
        if self.upsampler is not None:
            x = self.upsampler(x)

        return x
