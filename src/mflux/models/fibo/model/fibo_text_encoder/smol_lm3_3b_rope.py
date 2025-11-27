from typing import Tuple

import mlx.core as mx
from mlx import nn


class SmolLM3_3B_RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 65_536,
        base: float = 5_000_000.0,
    ):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        cos, sin = SmolLM3_3B_RotaryEmbedding._build_cos_sin(self.inv_freq, seq_len)
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=0)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=0)
        return cos, sin

    @staticmethod
    def _build_cos_sin(inv_freq, seq_len: int) -> Tuple[mx.array, mx.array]:
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin
