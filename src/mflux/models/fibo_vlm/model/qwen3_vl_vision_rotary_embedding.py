import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = mx.array(inv_freq)

    def __call__(self, seqlen: int) -> mx.array:
        seq = mx.arange(seqlen, dtype=mx.float32)
        freqs = mx.outer(seq, self.inv_freq)
        return freqs
