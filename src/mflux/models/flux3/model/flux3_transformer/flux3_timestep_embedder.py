import math

import mlx.core as mx
from mlx import nn


class Flux3TimestepEmbedder(nn.Module):
    # The reference carries timesteps in bfloat16, so the sinusoidal embedding is bfloat16-rounded.
    EMBEDDING_DTYPE = mx.bfloat16

    def __init__(self, in_channels: int = 256, embedding_dim: int = 3072):
        super().__init__()
        self.in_channels = in_channels
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timestep: mx.array) -> mx.array:
        emb = Flux3TimestepEmbedder._timestep_embedding(timestep, self.in_channels)
        return self.linear_2(nn.silu(self.linear_1(emb)))

    @staticmethod
    def _timestep_embedding(timestep: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        t = timestep.astype(Flux3TimestepEmbedder.EMBEDDING_DTYPE) * 1000.0
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(t.dtype)
