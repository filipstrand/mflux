from mlx import nn
import mlx.core as mx

from mflux.models.text_encoder.clip_encoder.clip_mlp import CLIPMLP
from mflux.models.text_encoder.clip_encoder.clip_sdpa_attention import CLIPSdpaAttention


class CLIPEncoderLayer(nn.Module):

    def __init__(self, layer: int):
        super().__init__()
        self.self_attn = CLIPSdpaAttention()
        self.layer_norm1 = nn.LayerNorm(dims=768)
        self.mlp = CLIPMLP()
        self.layer_norm2 = nn.LayerNorm(dims=768)

    def forward(self, hidden_states: mx.array, causal_attention_mask: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
