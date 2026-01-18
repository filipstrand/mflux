import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 1000000.0,
        scaling_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.mrope_section = mrope_section or [24, 20, 20]
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if len(position_ids.shape) == 2:
            batch_size, seq_len = position_ids.shape
            position_ids = mx.broadcast_to(mx.expand_dims(position_ids, axis=0), (3, batch_size, seq_len))

        inv_freq_expanded = mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0)
        inv_freq_expanded = mx.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = mx.broadcast_to(inv_freq_expanded, (3, position_ids.shape[1], self.inv_freq.shape[0], 1))
        inv_freq_expanded = inv_freq_expanded.astype(mx.float32)

        position_ids_expanded = mx.expand_dims(position_ids, axis=2)
        position_ids_expanded = position_ids_expanded.astype(mx.float32)
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = mx.transpose(freqs, (0, 1, 3, 2))
        freqs_interleaved = Qwen3VLRotaryEmbedding._apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs_interleaved, freqs_interleaved], axis=-1)

        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos.astype(x.dtype), sin.astype(x.dtype)

    @staticmethod
    def _apply_interleaved_mrope(freqs: mx.array, mrope_section: list[int]) -> mx.array:
        freqs_t = freqs[0]
        freqs_t_np = np.array(freqs_t)

        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            indices_np = np.arange(offset, length, 3)
            freqs_dim_np = np.array(freqs[dim])
            freqs_t_np[..., indices_np] = freqs_dim_np[..., indices_np]

        freqs_t = mx.array(freqs_t_np)
        return freqs_t
