"""Wan 3D Causal Convolution for FIBO VAE.

This is similar to QwenImageCausalConv3D but follows the WanVAE structure.
"""

import mlx.core as mx
from mlx import nn

from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_save


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
        name: str | None = None,  # Optional name for debugging
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
        self.name = name or f"conv3d_{in_channels}to{out_channels}"

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal 3D convolution.

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)

        Returns:
            Convolved tensor
        """
        # CHECKPOINT: Before padding
        debug_checkpoint(
            f"mlx_{self.name}_before_padding",
            metadata={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(x, f"mlx_{self.name}_before_padding")

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

        # CHECKPOINT: After padding, before transpose
        debug_checkpoint(
            f"mlx_{self.name}_after_padding",
            metadata={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(x, f"mlx_{self.name}_after_padding")

        # MLX Conv3d expects (batch, channels, depth, height, width)
        # Transpose to (batch, depth, height, width, channels) for conv, then back
        x = mx.transpose(x, (0, 2, 3, 4, 1))

        # CHECKPOINT: After transpose, before conv3d
        debug_checkpoint(
            f"mlx_{self.name}_after_transpose_before_conv",
            metadata={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(x, f"mlx_{self.name}_after_transpose_before_conv")

        x = self.conv3d(x)

        # CHECKPOINT: After conv3d, before transpose back
        debug_checkpoint(
            f"mlx_{self.name}_after_conv_before_transpose_back",
            metadata={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(x, f"mlx_{self.name}_after_conv_before_transpose_back")

        x = mx.transpose(x, (0, 4, 1, 2, 3))

        # CHECKPOINT: After transpose back (final output)
        debug_checkpoint(
            f"mlx_{self.name}_after_transpose_back",
            metadata={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(x, f"mlx_{self.name}_after_transpose_back")

        return x
