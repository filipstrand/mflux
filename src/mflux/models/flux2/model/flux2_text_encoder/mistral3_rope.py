"""Rotary Position Embeddings for Mistral3 text encoder."""

import mlx.core as mx
from mlx import nn


class Mistral3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Mistral3.

    Args:
        dim: Head dimension
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation (default 1_000_000_000 for Mistral3)
    """

    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 131072,
        base: float = 1_000_000_000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute rotary embeddings.

        Args:
            hidden_states: Hidden states [B, seq_len, hidden_dim]
            position_ids: Position IDs [B, seq_len]

        Returns:
            Tuple of (cos, sin) embeddings
        """
        # Compute frequencies: [seq_len, dim//2]
        seq_len = position_ids.shape[-1]
        position_ids = position_ids.astype(mx.float32)

        # [seq_len, 1] @ [1, dim//2] -> [seq_len, dim//2]
        freqs = mx.expand_dims(position_ids.flatten(), axis=-1) * mx.expand_dims(self.inv_freq, axis=0)

        # Concatenate to get full dimension
        # [seq_len, dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(emb)
        sin = mx.sin(emb)

        # Reshape for broadcasting with attention: [1, seq_len, 1, dim]
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)

        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [B, num_heads, seq_len, head_dim]
        k: Key tensor [B, num_kv_heads, seq_len, head_dim]
        cos: Cosine embeddings [1, seq_len, 1, head_dim]
        sin: Sine embeddings [1, seq_len, 1, head_dim]

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    # Transpose for proper broadcasting
    # cos, sin: [1, seq_len, 1, head_dim] -> [1, 1, seq_len, head_dim]
    cos = mx.transpose(cos, (0, 2, 1, 3))
    sin = mx.transpose(sin, (0, 2, 1, 3))

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
