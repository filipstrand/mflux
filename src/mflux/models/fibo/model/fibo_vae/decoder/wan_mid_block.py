"""Wan Mid Block for FIBO VAE decoder.

Middle block with residual blocks and attention layers.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_attention_block import WanAttentionBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanMidBlock(nn.Module):
    """Middle block for decoder.

    Structure: resnet0 -> attention -> resnet1 (for num_layers=1)
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        """Initialize mid block.

        Args:
            dim: Number of channels
            dropout: Dropout rate (not used)
            non_linearity: Activation (always "silu")
            num_layers: Number of attention/residual block pairs (typically 1)
        """
        super().__init__()
        # First residual block
        self.resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        # Attention blocks and additional residual blocks
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(WanAttentionBlock(dim))
            self.resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply mid block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # First residual block
        x = self.resnets[0](x)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)

        return x
