"""Wan Attention Block for FIBO VAE decoder.

Causal self-attention with a single head.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm


class WanAttentionBlock(nn.Module):
    """Causal self-attention with a single head.

    Args:
        dim: Number of channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Layers
        self.norm = WanRMSNorm(dim, images=True)  # For 4D tensors (after reshape)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply attention block.

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)

        Returns:
            Output tensor with residual connection
        """
        identity = x
        batch_size, channels, time, height, width = x.shape

        # Reshape for 2D attention: (b, c, t, h, w) -> (b*t, c, h, w)
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, t, c, h, w)
        x = mx.reshape(x, (batch_size * time, channels, height, width))

        # Normalize (expects 4D: batch, channels, height, width)
        x = self.norm(x)

        # Transpose to channels-last for Conv2d: (b*t, c, h, w) -> (b*t, h, w, c)
        x = mx.transpose(x, (0, 2, 3, 1))

        # Compute query, key, value
        qkv = self.to_qkv(x)  # (b*t, h, w, c*3)

        # Reshape to match PyTorch: (b*t, h, w, c*3) -> (b*t, 1, c*3, h*w)
        # First transpose to channels-first: (b*t, h, w, c*3) -> (b*t, c*3, h, w)
        qkv = mx.transpose(qkv, (0, 3, 1, 2))  # (b*t, c*3, h, w)
        # Reshape: (b*t, c*3, h, w) -> (b*t, 1, c*3, h*w)
        qkv = mx.reshape(qkv, (batch_size * time, 1, channels * 3, height * width))
        # Permute to match PyTorch: (b*t, 1, c*3, h*w) -> (b*t, 1, h*w, c*3)
        qkv = mx.transpose(qkv, (0, 1, 3, 2))  # (b*t, 1, h*w, c*3)

        # Split into q, k, v: each becomes (b*t, 1, h*w, c)
        q, k, v = mx.split(qkv, 3, axis=3)

        # Apply scaled dot-product attention
        # q, k, v: (b*t, 1, h*w, c)
        scale = 1.0 / (channels**0.5)
        # scores: (b*t, 1, h*w, h*w)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        x = mx.matmul(attn_weights, v)  # (b*t, 1, h*w, c)

        # Squeeze and reshape: (b*t, 1, h*w, c) -> (b*t, h*w, c) -> (b*t, c, h, w)
        x = mx.reshape(x, (batch_size * time, height * width, channels))
        x = mx.transpose(x, (0, 2, 1))  # (b*t, c, h*w)
        x = mx.reshape(x, (batch_size * time, channels, height, width))

        # Transpose to channels-last for Conv2d: (b*t, c, h, w) -> (b*t, h, w, c)
        x = mx.transpose(x, (0, 2, 3, 1))

        # Output projection
        x = self.proj(x)  # (b*t, h, w, c)

        # Transpose back to channels-first: (b*t, h, w, c) -> (b*t, c, h, w)
        x = mx.transpose(x, (0, 3, 1, 2))

        # Reshape back: (b*t, c, h, w) -> (b, c, t, h, w)
        x = mx.reshape(x, (batch_size, time, channels, height, width))
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, c, t, h, w)

        # Residual connection
        return x + identity
