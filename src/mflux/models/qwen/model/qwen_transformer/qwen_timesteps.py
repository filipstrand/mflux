import mlx.core as mx
from mlx import nn


class QwenTimesteps(nn.Module):
    def __init__(self, proj_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        half_dim = self.proj_dim // 2
        max_period = 10000.0
        exponent = -mx.log(mx.array(max_period)) * mx.arange(0, half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - 0.0)
        freqs = mx.exp(exponent)
        emb = timesteps.astype(mx.float32)[:, None] * freqs[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb
