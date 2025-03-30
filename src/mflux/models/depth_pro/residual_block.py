import mlx.core as mx
import mlx.nn as nn


class ResidualBlock(nn.Module):
    """Generic implementation of residual blocks.

    Adapted from "He et al. - Identity Mappings in Deep Residual Networks (2016)"
    """

    def __init__(self, residual: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual connection"""
        delta_x = self.residual(x)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + delta_x
