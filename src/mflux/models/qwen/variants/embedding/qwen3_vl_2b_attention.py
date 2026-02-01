"""Attention module for Qwen3-VL-2B Embedding and Reranker models.

Key differences from Qwen2-VL-7B attention:
- No attention bias on q/k/v projections
- QK normalization (q_norm, k_norm) with RMSNorm
"""

import math

import mlx.core as mx
from mlx import nn


class QwenRMSNorm(nn.Module):
    """RMSNorm for QK normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight * x).astype(input_dtype)


class Qwen3VL2BAttention(nn.Module):
    """Multi-head attention for Qwen3-VL-2B.

    Key features:
    - No bias on q/k/v projections
    - QK normalization before RoPE
    - GQA (grouped query attention) with num_key_value_heads < num_attention_heads
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # No bias on projections (key difference from 7B)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # QK normalization (key difference from 7B)
        self.q_norm = QwenRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=1e-6)

        self.rope_scaling = rope_scaling or {"mrope_section": [16, 24, 24]}

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization before RoPE
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention: [batch, heads, seq, head_dim]
        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        # Apply multimodal rotary position embeddings
        query_states, key_states = self._apply_multimodal_rotary_pos_emb(
            q=query_states,
            k=key_states,
            position_embeddings=position_embeddings,
            mrope_section=self.rope_scaling["mrope_section"],
        )

        # Repeat KV heads for GQA
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Attention scores
        attn_weights = mx.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax and output
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(query_states.dtype)
        attn_output = mx.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        """Repeat KV heads for grouped query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _apply_multimodal_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
        mrope_section: list[int],
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Apply multimodal rotary position embeddings.

        This implementation matches the original QwenAttention approach:
        1. Position embeddings have shape (3, batch, seq, head_dim) for t/h/w
        2. Split by mrope_section_doubled and select from corresponding dimension
        3. Concatenate back together
        """
        if position_embeddings is None:
            return q, k

        mrope_section_doubled = [s * 2 for s in mrope_section]

        cos, sin = position_embeddings
        cos_chunks = []
        sin_chunks = []
        start_idx = 0
        for section_size in mrope_section_doubled:
            end_idx = start_idx + section_size
            cos_chunk = cos[..., start_idx:end_idx]
            sin_chunk = sin[..., start_idx:end_idx]
            cos_chunks.append(cos_chunk)
            sin_chunks.append(sin_chunk)
            start_idx = end_idx

        # Select from t/h/w dimensions cycling
        cos_selected = [chunk[i % 3] for i, chunk in enumerate(cos_chunks)]
        sin_selected = [chunk[i % 3] for i, chunk in enumerate(sin_chunks)]

        cos_combined = mx.concatenate(cos_selected, axis=-1)
        sin_combined = mx.concatenate(sin_selected, axis=-1)

        if unsqueeze_dim == 1:
            cos_combined = mx.expand_dims(cos_combined, axis=1)
            sin_combined = mx.expand_dims(sin_combined, axis=1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        cos_combined = cos_combined.astype(mx.float32)
        sin_combined = sin_combined.astype(mx.float32)

        q_embed = (q * cos_combined) + (Qwen3VL2BAttention._rotate_half(q) * sin_combined)
        k_embed = (k * cos_combined) + (Qwen3VL2BAttention._rotate_half(k) * sin_combined)

        q_embed = q_embed.astype(orig_q_dtype)
        k_embed = k_embed.astype(orig_k_dtype)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        """Rotate half of the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
