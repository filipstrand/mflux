import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_mlp import Siglip2MLP


class Siglip2PoolingHead(nn.Module):
    """SigLIP2-G384 multi-head attention pooling head.

    Uses a learnable probe to attend to all hidden states and produce
    a single pooled output vector of dimension 1536.
    """

    def __init__(self):
        super().__init__()
        self.probe = mx.random.normal(shape=(1, 1, 1536))
        self.attention = nn.MultiHeadAttention(dims=1536, num_heads=16, bias=True)
        self.layernorm = nn.LayerNorm(1536, eps=1e-6)
        self.mlp = Siglip2MLP()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B = hidden_states.shape[0]
        query = mx.broadcast_to(self.probe, (B, 1, 1536))
        pooled_output = self.attention(query, hidden_states, hidden_states)
        residual = pooled_output
        pooled_output = self.layernorm(pooled_output)
        pooled_output = residual + self.mlp(pooled_output)
        pooled_output = pooled_output[:, 0]  # (B, 1536)
        return pooled_output
