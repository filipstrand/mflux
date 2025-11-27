import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.fibo.model.fibo_transformer.fibo_single_attention import FiboSingleAttention


class FiboJointAttention(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = attention_head_dim
        self.num_heads = num_attention_heads
        self.inner_dim = dim

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

        # Added KV projections for encoder_hidden_states
        self.add_q_proj = nn.Linear(dim, self.inner_dim)
        self.add_k_proj = nn.Linear(dim, self.inner_dim)
        self.add_v_proj = nn.Linear(dim, self.inner_dim)

        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=eps)

        # Output projections
        self.to_out = [nn.Linear(self.inner_dim, dim)]
        self.to_add_out = nn.Linear(self.inner_dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        batch_size, seq_img, dim = hidden_states.shape
        _, seq_ctx, _ = encoder_hidden_states.shape

        cos, sin = image_rotary_emb

        # QKV for image stream: [B, S_img, inner_dim]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # QKV for context stream (added_kv): [B, S_ctx, inner_dim]
        enc_query = self.add_q_proj(encoder_hidden_states)
        enc_key = self.add_k_proj(encoder_hidden_states)
        enc_value = self.add_v_proj(encoder_hidden_states)

        # Reshape to [B, S, H, D]
        query = mx.reshape(query, (batch_size, seq_img, self.num_heads, self.head_dim))
        key = mx.reshape(key, (batch_size, seq_img, self.num_heads, self.head_dim))
        value = mx.reshape(value, (batch_size, seq_img, self.num_heads, self.head_dim))

        enc_query = mx.reshape(enc_query, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_key = mx.reshape(enc_key, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_value = mx.reshape(enc_value, (batch_size, seq_ctx, self.num_heads, self.head_dim))

        # RMSNorm over last dim: produce normalized Q/K matching BriaFiboAttention semantics.
        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)

        enc_query = self.norm_added_q(enc_query.astype(mx.float32)).astype(enc_query.dtype)
        enc_key = self.norm_added_k(enc_key.astype(mx.float32)).astype(enc_key.dtype)

        # Concatenate encoder + image along sequence dim (matches BriaFiboAttnProcessor)
        query = mx.concatenate([enc_query, query], axis=1)
        key = mx.concatenate([enc_key, key], axis=1)
        value = mx.concatenate([enc_value, value], axis=1)

        seq_total = seq_ctx + seq_img

        # Apply RoPE to Q,K (layout [B, S_total, H, D])
        query = FiboSingleAttention.apply_rotary_emb(query, cos, sin)
        key = FiboSingleAttention.apply_rotary_emb(key, cos, sin)

        # Convert to [B, H, S, D] for fast SDPA
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        # Prepare mask for scaled_dot_product_attention
        attn_mask = None
        if attention_mask is not None:
            attn_mask = mx.broadcast_to(attention_mask, (batch_size, self.num_heads, seq_total, seq_total))
            attn_mask = attn_mask.astype(query_bhsd.dtype)

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))
        attn_output_bhsd = scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=scale,
            mask=attn_mask,
        )

        # Back to [B, S_total, H, D] then flatten to [B, S_total, inner_dim]
        attn_output = mx.transpose(attn_output_bhsd, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_total, self.inner_dim))

        # Split back into context and image streams
        context_attn_output = attn_output[:, :seq_ctx, :]
        hidden_attn_output = attn_output[:, seq_ctx:, :]

        # Output projections
        hidden_attn_output = self.to_out[0](hidden_attn_output)
        context_attn_output = self.to_add_out(context_attn_output)

        return hidden_attn_output, context_attn_output
