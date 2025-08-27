"""
RoPE (Rotary Position Embedding) implementation for Qwen text encoder.
Ported from transformers Qwen2_5_VLRotaryEmbedding with identical math.
"""

import mlx.core as mx
from mlx import nn


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
) -> tuple[mx.array, mx.array]:
    """
    Apply simple rotary position embedding for text-only case.
    """
    # For text-only, we just use the first dimension of the 3D cos/sin
    # cos/sin shape: (3, batch_size, seq_len, head_dim) -> use [0] for text
    cos = cos[0]  # Shape: (batch_size, seq_len, head_dim)
    sin = sin[0]  # Shape: (batch_size, seq_len, head_dim)

    # Add head dimension if needed
    if unsqueeze_dim == 1:
        cos = mx.expand_dims(cos, axis=1)  # (batch_size, 1, seq_len, head_dim)
        sin = mx.expand_dims(sin, axis=1)  # (batch_size, 1, seq_len, head_dim)
    elif unsqueeze_dim == 2:
        cos = mx.expand_dims(cos, axis=2)
        sin = mx.expand_dims(sin, axis=2)

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, mrope_section: list[int], unsqueeze_dim: int = 1
) -> tuple[mx.array, mx.array]:
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    For text-only models, we use simplified RoPE. For multimodal, this would be more complex.
    """
    # For text-only models (which is our case), use simple RoPE
    # The multimodal sections are only needed for vision+text models
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)


class QwenRotaryEmbedding(nn.Module):
    """
    Qwen Rotary Position Embedding implementation in MLX.

    This is a faithful port of Qwen2_5_VLRotaryEmbedding from transformers.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        base: float = 1000000.0,  # rope_theta from config
        device: str = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        config=None,
    ):
        super().__init__()

        self.rope_kwargs = {}
        # BC: "rope_type" was originally "type"
        if config is not None and hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
            self.rope_kwargs = {key: value for key, value in config.rope_scaling.items() if key != "rope_type"}
        else:
            self.rope_type = rope_type

        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        # Initialize inverse frequencies
        # This matches the reference implementation exactly
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

        # Attention scaling (usually 1.0 for default rope)
        self.attention_scaling = scaling_factor

        # For text-only models, we don't need the complex multimodal setup
        # but we keep it for compatibility
        self.rope_init_fn = self._default_rope_init

    def _default_rope_init(self, config, device):
        """Default RoPE initialization matching transformers."""
        return self.inv_freq, 1.0

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        """
        Forward pass to compute cos and sin embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position indices [3, batch_size, seq_len] for multimodal
                         or [batch_size, seq_len] for text-only

        Returns:
            tuple: (cos, sin) embeddings
        """
        # Handle different position_ids shapes
        if len(position_ids.shape) == 2:
            # Text-only case: [batch_size, seq_len] -> [3, batch_size, seq_len]
            batch_size, seq_len = position_ids.shape
            position_ids = mx.broadcast_to(mx.expand_dims(position_ids, axis=0), (3, batch_size, seq_len))

        # Expand inv_freq to shape (3, ...)
        # position_ids shape: (3, batch_size, seq_len)
        inv_freq_dim = self.inv_freq.shape[0]  # dim//2
        inv_freq_expanded = mx.expand_dims(
            mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0), axis=-1
        )  # Shape: (1, 1, dim//2, 1)
        inv_freq_expanded = mx.broadcast_to(inv_freq_expanded, (3, position_ids.shape[1], inv_freq_dim, 1))

        position_ids_expanded = mx.expand_dims(position_ids, axis=2)  # (3, bs, 1, seq_len)

        # Compute frequencies: (inv_freq_expanded @ position_ids_expanded)
        # This is matrix multiplication along the last two dimensions
        freqs = mx.matmul(
            inv_freq_expanded.astype(mx.float32), position_ids_expanded.astype(mx.float32)
        )  # Shape: (3, bs, dim//2, seq_len)

        freqs = mx.transpose(freqs, (0, 1, 3, 2))  # Shape: (3, bs, seq_len, dim//2)

        # Create embeddings by concatenating freqs with itself
        emb = mx.concatenate([freqs, freqs], axis=-1)  # Shape: (3, bs, seq_len, dim)

        # Compute cos and sin
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        # Convert to appropriate dtype
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)

        return cos, sin
