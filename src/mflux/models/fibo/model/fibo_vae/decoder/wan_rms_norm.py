"""Wan RMS Normalization for FIBO VAE.

Similar to QwenImageRMSNorm but follows WanVAE structure.
"""

import mlx.core as mx
from mlx import nn


class WanRMSNorm(nn.Module):
    """RMS Normalization layer for WanVAE.

    Similar to QwenImageRMSNorm - normalizes over channels and scales by sqrt(dim).
    """

    def __init__(self, dim: int, eps: float = 1e-12, images: bool = True):
        """Initialize RMS normalization.

        Args:
            dim: Number of channels
            eps: Small epsilon for numerical stability
            images: Whether input is image data (affects weight shape)
        """
        super().__init__()
        self.eps = eps
        self.scale = float(dim) ** 0.5
        self.images = images
        if images:
            self.weight = mx.ones((dim, 1, 1))
        else:
            # For 3D (5D tensor: batch, channels, time, height, width)
            self.weight = mx.ones((dim, 1, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input tensor (4D or 5D)

        Returns:
            Normalized tensor
        """
        # Compute L2 norm over channel dimension
        sum_sq = mx.sum(x * x, axis=1, keepdims=True)
        l2_norm = mx.sqrt(sum_sq)
        denom = mx.maximum(l2_norm, mx.array(self.eps, dtype=l2_norm.dtype))
        x_normalized = x / denom

        # Reshape weight to match input dimensions
        if x.ndim == 5 and not self.images:
            weight = self.weight.reshape(1, -1, 1, 1, 1)
        elif x.ndim == 4 and self.images:
            weight = self.weight.reshape(1, -1, 1, 1)
        else:
            if x.ndim == 5:
                weight = self.weight.reshape(1, -1, 1, 1, 1)
            elif x.ndim == 4:
                weight = self.weight.reshape(1, -1, 1, 1)
            else:
                weight = self.weight

        return x_normalized * self.scale * weight
