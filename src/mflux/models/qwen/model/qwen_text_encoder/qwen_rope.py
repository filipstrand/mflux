import mlx.core as mx
from mlx import nn


class QwenRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        base: float = 1000000.0,
        device: str = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        config=None,
    ):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.attention_scaling = scaling_factor

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
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)
