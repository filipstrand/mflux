"""Wan 3D Causal Convolution for FIBO VAE.

This is similar to QwenImageCausalConv3D but follows the WanVAE structure.
"""

import mlx.core as mx
from mlx import nn


class WanCausalConv3d(nn.Module):
    """3D Causal Convolution for WanVAE.

    Similar to QwenImageCausalConv3D - applies causal padding in the temporal dimension.
    For now, we implement a simplified version without feature caching.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal 3D convolution.

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)

        Returns:
            Convolved tensor
        """
        pad_t = pad_h = pad_w = self.padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # Causal padding: pad temporal dimension only on the left (past)
            # Spatial dimensions: pad symmetrically
            pad_spec = [
                (0, 0),  # batch
                (0, 0),  # channels
                (2 * pad_t, 0),  # temporal: only pad left (causal)
                (pad_h, pad_h),  # height: symmetric
                (pad_w, pad_w),  # width: symmetric
            ]
            x = mx.pad(x, pad_spec)

        # MLX Conv3d expects (batch, channels, depth, height, width)
        # Transpose to (batch, depth, height, width, channels) for conv, then back
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        x = self.conv3d(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x
