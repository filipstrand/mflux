import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Siglip2Attention(nn.Module):
    """SigLIP2-G384 self-attention: hidden_size=1536, num_heads=16, head_dim=96."""

    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 96  # 1536 / 16
        dim = self.num_heads * self.head_dim  # 1536
        self.q_proj = nn.Linear(input_dims=dim, output_dims=dim)
        self.k_proj = nn.Linear(input_dims=dim, output_dims=dim)
        self.v_proj = nn.Linear(input_dims=dim, output_dims=dim)
        self.out_proj = nn.Linear(input_dims=dim, output_dims=dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, N, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1 / mx.sqrt(mx.array(self.head_dim, dtype=query.dtype))
        hidden_states = scaled_dot_product_attention(query, key, value, scale=scale)

        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.out_proj(hidden_states)
