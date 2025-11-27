import math

import mlx.core as mx
from mlx import nn


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size: int,
        mid_size: int | None = None,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        if mid_size is None:
            mid_size = out_size

        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size, bias=True)
        self.linear2 = nn.Linear(mid_size, out_size, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.linear1(t_freq)
        t_emb = nn.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb

    @staticmethod
    def _timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
