import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_attention import Siglip2Attention
from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_mlp import Siglip2MLP


class Siglip2EncoderLayer(nn.Module):
    """Single SigLIP2-G384 encoder layer."""

    def __init__(self):
        super().__init__()
        self.self_attn = Siglip2Attention()
        self.layer_norm1 = nn.LayerNorm(1536, eps=1e-6)
        self.mlp = Siglip2MLP()
        self.layer_norm2 = nn.LayerNorm(1536, eps=1e-6)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
