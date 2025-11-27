import math
from typing import Tuple

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from .smol_lm3_3b_rope import SmolLM3_3B_RotaryEmbedding


class SmolLM3_3B_SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 65_536,
        rope_theta: float = 5_000_000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = attention_dropout

        # Projections – no bias, matching SmolLM3 config (attention_bias = False)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = SmolLM3_3B_RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cos_sin: Tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # Projections: Use nn.Linear directly (like Qwen/Flux models)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Cast back to input dtype if Linear output is float32 (for consistency)
        if q.dtype != hidden_states.dtype:
            q = q.astype(hidden_states.dtype)
            k = k.astype(hidden_states.dtype)
            v = v.astype(hidden_states.dtype)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Rotary embeddings
        if cos_sin is None:
            cos_sin = self.rotary_emb(seq_len)
        cos, sin = cos_sin
        q, k = SmolLM3_3B_SelfAttention._apply_rope(q, k, cos, sin)

        # Grouped-query attention: repeat kv heads if needed
        if self.num_key_value_heads != self.num_attention_heads:
            k = SmolLM3_3B_SelfAttention._repeat_kv(k, self.num_key_value_groups)
            v = SmolLM3_3B_SelfAttention._repeat_kv(v, self.num_key_value_groups)

        # Prepare mask for scaled_dot_product_attention
        attn_mask = None
        if attention_mask is not None:
            seq_len_k = k.shape[2]
            causal_mask = attention_mask[:, :, :, :seq_len_k]
            if causal_mask.shape[1] == 1:
                causal_mask = mx.broadcast_to(causal_mask, (batch_size, self.num_attention_heads, seq_len, seq_len_k))
            attn_mask = causal_mask.astype(q.dtype)

        # Use MLX fast SDPA
        attn_output = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=attn_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    @staticmethod
    def _apply_rope(
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        cos = cos.astype(mx.float32)
        sin = sin.astype(mx.float32)

        q_embed = (q * cos) + (SmolLM3_3B_SelfAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (SmolLM3_3B_SelfAttention._rotate_half(k) * sin)

        return q_embed.astype(q_dtype), k_embed.astype(k_dtype)

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_kv_heads, n_rep, seq_len, head_dim))
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
