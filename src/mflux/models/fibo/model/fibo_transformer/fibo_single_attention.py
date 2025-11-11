import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class FiboSingleAttention(nn.Module):
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

    def __call__(
        self,
        hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = image_rotary_emb

        # [B, S, inner_dim]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = mx.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = mx.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = mx.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))

        # RMSNorm
        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)

        # RoPE
        query = FiboSingleAttention.apply_rotary_emb(query, cos, sin)
        key = FiboSingleAttention.apply_rotary_emb(key, cos, sin)

        # [B, H, S, D]
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))
        attn_output = scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=scale,
            mask=attention_mask,
        )

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, self.inner_dim))
        return attn_output

    @staticmethod
    def apply_rotary_emb(
        x: mx.array,
        freqs_cos: mx.array,
        freqs_sin: mx.array,
    ) -> mx.array:
        bsz, seq_len, num_heads, head_dim = x.shape
        cos = mx.expand_dims(mx.expand_dims(freqs_cos, axis=0), axis=2)
        sin = mx.expand_dims(mx.expand_dims(freqs_sin, axis=0), axis=2)
        x2 = x.reshape(bsz, seq_len, num_heads, -1, 2)
        x_real = x2[..., 0]
        x_imag = x2[..., 1]
        x_rotated_real = -x_imag
        x_rotated_imag = x_real
        x_rotated = mx.stack([x_rotated_real, x_rotated_imag], axis=-1).reshape(bsz, seq_len, num_heads, head_dim)
        out = (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)
        return out
