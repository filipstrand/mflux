import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.siglip_vision_transformer.siglip_mlp import SiglipMLP
from mflux.models.flux.model.siglip_vision_transformer.siglip_sdpa_attention import SiglipSdpaAttention


class SiglipEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SiglipSdpaAttention()
        self.layer_norm1 = nn.LayerNorm(1152, eps=1e-6)
        self.mlp = SiglipMLP()
        self.layer_norm2 = nn.LayerNorm(1152, eps=1e-6)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attention_output = self.self_attn(hidden_states)
        hidden_states = residual + attention_output

        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
