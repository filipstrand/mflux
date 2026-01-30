import mlx.core as mx
from mlx import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1000000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    # Maximum safe position ID to prevent overflow in frequency computation
    # With base=1e6 and dim=128, position * inv_freq can overflow float32 around 1e10
    MAX_POSITION_ID = 1_000_000

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # Use actual position_ids values for correct embeddings with non-contiguous sequences
        # (e.g., padding/masking during training). Previously only used seq_len via mx.arange().
        positions = position_ids[0].astype(mx.float32)

        # Clamp positions to prevent numerical overflow in frequency computation
        # Large position values (>1M) would cause overflow when multiplied by inv_freq
        positions = mx.clip(positions, 0, self.MAX_POSITION_ID)

        freqs = mx.outer(positions, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, :, :]
        sin = mx.sin(emb)[None, :, :]
        return cos.astype(x.dtype), sin.astype(x.dtype)
