import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_rms_norm import Wan2_2_RMSNorm


class Wan2_2_AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = Wan2_2_RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch_size, channels, time, height, width = x.shape
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (batch_size * time, channels, height, width))

        x = self.norm(x)
        x = mx.transpose(x, (0, 2, 3, 1))

        qkv = self.to_qkv(x)
        qkv = mx.transpose(qkv, (0, 3, 1, 2))
        qkv = mx.reshape(qkv, (batch_size * time, 1, channels * 3, height * width))
        qkv = mx.transpose(qkv, (0, 1, 3, 2))
        q, k, v = mx.split(qkv, 3, axis=3)

        scale = 1.0 / (channels**0.5)
        x = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            mask=None,
        )
        x = mx.reshape(x, (batch_size * time, height * width, channels))
        x = mx.transpose(x, (0, 2, 1))
        x = mx.reshape(x, (batch_size * time, channels, height, width))
        x = mx.transpose(x, (0, 2, 3, 1))

        x = self.proj(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (batch_size, time, channels, height, width))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x + identity
