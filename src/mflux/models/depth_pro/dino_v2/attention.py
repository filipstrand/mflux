import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.LayerNorm(dims=self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(dims=self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]

        # Linear projections
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape and transpose for multi-head attention
        q = self._reshape_and_transpose(q, batch_size)
        k = self._reshape_and_transpose(k, batch_size)
        v = self._reshape_and_transpose(v, batch_size)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot product attention
        scale = 1 / mx.sqrt(q.shape[-1])
        attn_output = scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, -1, self.num_heads * self.head_dim))

        # Final projection and dropout
        x = self.proj(attn_output)
        return x

    def _reshape_and_transpose(self, x, batch_size):
        return mx.transpose(mx.reshape(x, (batch_size, -1, self.num_heads, self.head_dim)), (0, 2, 1, 3))
