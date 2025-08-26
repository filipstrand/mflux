import math

import mlx.core as mx
from mlx import nn

from .qwen_rope import QwenRotaryEmbedding, apply_multimodal_rotary_pos_emb


class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int = None,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.kv_head_dim = hidden_size // num_attention_heads
        self.num_queries_per_kv = num_attention_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )
        self.rope_scaling = rope_scaling or {"mrope_section": [16, 24, 24]}

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention (GQA)
        query_states = query_states.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.kv_head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.kv_head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        query_states = mx.transpose(query_states, (0, 2, 1, 3))
        key_states = mx.transpose(key_states, (0, 2, 1, 3))
        value_states = mx.transpose(value_states, (0, 2, 1, 3))

        # Apply RoPE (Rotary Position Embedding)
        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.int32)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))

        # Get RoPE embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Apply multimodal rotary position embedding
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
            unsqueeze_dim=1,
        )

        # Expand key and value states to match query heads (GQA)
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = mx.repeat(key_states, self.num_queries_per_kv, axis=1)
            value_states = mx.repeat(value_states, self.num_queries_per_kv, axis=1)

        # Compute attention scores
        attn_weights = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2))) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = mx.expand_dims(mx.expand_dims(attention_mask, axis=1), axis=1)
            attn_weights = attn_weights + (1.0 - mask) * (-1e9)

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, value_states)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output
