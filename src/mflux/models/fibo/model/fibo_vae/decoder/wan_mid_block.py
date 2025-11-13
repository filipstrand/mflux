"""Wan Mid Block for FIBO VAE decoder.

Simplified version - for now we skip attention and just use residual blocks.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanMidBlock(nn.Module):
    """Middle block for decoder.

    For now, simplified to just residual blocks (no attention).
    Can add attention later if needed.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        """Initialize mid block.

        Args:
            dim: Number of channels
            dropout: Dropout rate (not used)
            non_linearity: Activation (always "silu")
            num_layers: Number of residual blocks (typically 1)
        """
        super().__init__()
        # First residual block
        self.resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        # Additional residual blocks if num_layers > 1
        for _ in range(num_layers):
            self.resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply mid block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for resnet in self.resnets:
            x = resnet(x)
        return x
