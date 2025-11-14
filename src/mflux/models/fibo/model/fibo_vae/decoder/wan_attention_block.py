import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch_size, channels, time, height, width = x.shape
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (batch_size * time, channels, height, width))

        # Normalize
        x = self.norm(x)
        x = mx.transpose(x, (0, 2, 3, 1))

        qkv = self.to_qkv(x)
        qkv = mx.transpose(qkv, (0, 3, 1, 2))
        qkv = mx.reshape(qkv, (batch_size * time, 1, channels * 3, height * width))
        qkv = mx.transpose(qkv, (0, 1, 3, 2))
        q, k, v = mx.split(qkv, 3, axis=3)

        # Apply scaled dot-product attention
        scale = 1.0 / (channels**0.5)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        x = mx.matmul(attn_weights, v)
        x = mx.reshape(x, (batch_size * time, height * width, channels))
        x = mx.transpose(x, (0, 2, 1))  # (b*t, c, h*w)
        x = mx.reshape(x, (batch_size * time, channels, height, width))
        x = mx.transpose(x, (0, 2, 3, 1))

        x = self.proj(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (batch_size, time, channels, height, width))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x + identity
