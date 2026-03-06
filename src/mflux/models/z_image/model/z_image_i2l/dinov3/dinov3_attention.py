import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_rope import apply_dinov3_rope


class DINOv3Attention(nn.Module):
    """DINOv3 attention with RoPE on patch tokens.

    hidden_size=4096, num_heads=32, head_dim=128.
    Bias config: q=False, k=False, v=False, o=True.
    """

    def __init__(self):
        super().__init__()
        self.num_heads = 32
        self.head_dim = 128  # 4096 / 32
        dim = 4096
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_prefix_tokens: int = 5,
    ) -> mx.array:
        B, N, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE only to patch tokens
        q, k = apply_dinov3_rope(q, k, cos, sin, num_prefix_tokens=num_prefix_tokens)

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=q.dtype))
        out = scaled_dot_product_attention(q, k, v, scale=scale)

        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.o_proj(out)
