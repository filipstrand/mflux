import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.siglip_vision_transformer.siglip_mlp import SiglipMLP


class SiglipMultiHeadAttentionPoolingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = mx.random.normal(shape=(1, 1, 1152))
        self.attention = nn.MultiHeadAttention(dims=1152, num_heads=16, bias=True)
        self.layernorm = nn.LayerNorm(1152, eps=1e-6)
        self.mlp = SiglipMLP()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        query = mx.broadcast_to(self.probe, (1, 1, 1152))
        pooled_output = self.attention(query, hidden_states, hidden_states)
        residual = pooled_output
        pooled_output = self.layernorm(pooled_output)
        pooled_output = residual + self.mlp(pooled_output)
        pooled_output = pooled_output[:, 0]
        return pooled_output
