import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        head_dim: int = 64,
        num_heads: int = 16,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # Produce queries, keys and values
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot product attention
        scale = 1 / mx.sqrt(q.shape[-1])
        attn_output = scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (B, -1, self.num_heads * self.head_dim))

        # Final projection
        return self.proj(attn_output)
