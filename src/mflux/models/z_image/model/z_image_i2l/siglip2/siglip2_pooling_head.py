import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_mlp import Siglip2MLP


class Siglip2PoolingHead(nn.Module):
    """SigLIP2-G384 multi-head attention pooling head.

    Uses a learnable probe to attend over all hidden states and produce
    a single pooled output vector of dimension 1536.

    The original HF checkpoint stores Q/K/V as a fused `in_proj_weight` [4608, 1536].
    We use separate projections to match the weight mapping after split.
    """

    num_heads = 16
    head_dim = 96  # 1536 / 16

    def __init__(self):
        super().__init__()
        self.probe = mx.random.normal(shape=(1, 1, 1536))
        self.query_proj = nn.Linear(1536, 1536, bias=True)
        self.key_proj = nn.Linear(1536, 1536, bias=True)
        self.value_proj = nn.Linear(1536, 1536, bias=True)
        self.out_proj = nn.Linear(1536, 1536, bias=True)
        self.layernorm = nn.LayerNorm(1536, eps=1e-6)
        self.mlp = Siglip2MLP()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B = hidden_states.shape[0]
        query = mx.broadcast_to(self.probe, (B, 1, 1536))

        # Multi-head attention: probe attends to hidden_states
        q = self.query_proj(query).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key_proj(hidden_states).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value_proj(hidden_states).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=q.dtype))
        pooled = scaled_dot_product_attention(q, k, v, scale=scale)
        pooled = pooled.transpose(0, 2, 1, 3).reshape(B, 1, 1536)
        pooled = self.out_proj(pooled)

        # Residual + LayerNorm + MLP
        residual = pooled
        pooled = self.layernorm(pooled)
        pooled = residual + self.mlp(pooled)
        pooled = pooled[:, 0]  # (B, 1536)
        return pooled
