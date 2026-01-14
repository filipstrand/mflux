"""Grouped Query Attention for Mistral3 text encoder."""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_text_encoder.mistral3_rope import apply_rotary_pos_emb


class Mistral3Attention(nn.Module):
    """Grouped Query Attention for Mistral3.

    Uses 32 query heads and 8 key/value heads (4:1 ratio).

    Args:
        hidden_size: Hidden dimension (5120)
        num_heads: Number of query attention heads (32)
        num_kv_heads: Number of key/value heads (8)
        head_dim: Dimension per head (128)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads  # 4

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Forward pass through attention.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, 1, seq_len, seq_len]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            Output tensor [B, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [B, seq_len, num_heads * head_dim] -> [B, num_heads, seq_len, head_dim]
        query_states = mx.reshape(query_states, (batch_size, seq_len, self.num_heads, self.head_dim))
        query_states = mx.transpose(query_states, (0, 2, 1, 3))

        key_states = mx.reshape(key_states, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        key_states = mx.transpose(key_states, (0, 2, 1, 3))

        value_states = mx.reshape(value_states, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        value_states = mx.transpose(value_states, (0, 2, 1, 3))

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand K, V for GQA: [B, num_kv_heads, seq, head_dim] -> [B, num_heads, seq, head_dim]
        key_states = self._expand_kv_heads(key_states)
        value_states = self._expand_kv_heads(value_states)

        # Compute attention
        attn_weights = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2)))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        attn_output = mx.matmul(attn_weights, value_states)

        # Reshape: [B, num_heads, seq_len, head_dim] -> [B, seq_len, hidden_size]
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, self.num_heads * self.head_dim))

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _expand_kv_heads(self, states: mx.array) -> mx.array:
        """Expand KV heads for GQA.

        Repeats each KV head to match the number of query heads.

        Args:
            states: [B, num_kv_heads, seq_len, head_dim]

        Returns:
            Expanded states [B, num_heads, seq_len, head_dim]
        """
        batch_size, num_kv_heads, seq_len, head_dim = states.shape
        # [B, num_kv_heads, seq, head_dim] -> [B, num_kv_heads, 1, seq, head_dim]
        states = mx.expand_dims(states, axis=2)
        # Repeat along the new axis
        states = mx.broadcast_to(states, (batch_size, num_kv_heads, self.num_kv_groups, seq_len, head_dim))
        # Reshape to combine heads
        states = mx.reshape(states, (batch_size, self.num_heads, seq_len, head_dim))
        return states
