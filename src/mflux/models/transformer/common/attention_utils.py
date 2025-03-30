import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class AttentionUtils:
    @staticmethod
    def process_qkv(
        hidden_states: mx.array,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        norm_q: nn.RMSNorm,
        norm_k: nn.RMSNorm,
        num_heads: int,
        head_dim: int,
    ):
        query = to_q(hidden_states)
        key = to_k(hidden_states)
        value = to_v(hidden_states)

        # Reshape and transpose
        query = mx.transpose(mx.reshape(query, (1, -1, num_heads, head_dim)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (1, -1, num_heads, head_dim)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (1, -1, num_heads, head_dim)), (0, 2, 1, 3))

        # Apply normalization
        query = norm_q(query)
        key = norm_k(key)

        return query, key, value

    @staticmethod
    def compute_attention(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> mx.array:
        scale = 1 / mx.sqrt(query.shape[-1])
        hidden_states = scaled_dot_product_attention(query, key, value, scale=scale)

        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(
            hidden_states,
            (batch_size, -1, num_heads * head_dim),
        )

        return hidden_states

    @staticmethod
    def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
        xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).astype(mx.float32), xk_out.reshape(*xk.shape).astype(mx.float32)
