import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.t5_encoder.t5_layer_norm import T5LayerNorm
from mflux.models.text_encoder.t5_encoder.t5_self_attention import T5SelfAttention


class T5Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.SelfAttention = T5SelfAttention()
        self.layer_norm = T5LayerNorm()

    def forward(self, hidden_states: mx.array) -> mx.array:
        normed_hidden_states = self.layer_norm.forward(hidden_states)
        attention_output = self.SelfAttention.forward(normed_hidden_states)
        hidden_states = hidden_states + attention_output
        return hidden_states
