import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.t5_encoder.t5_attention import T5Attention
from mflux.models.text_encoder.t5_encoder.t5_feed_forward import T5FeedForward


class T5Block(nn.Module):

    def __init__(self, layer: int):
        super().__init__()
        self.attention = T5Attention()
        self.ff = T5FeedForward()

    def forward(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.attention.forward(hidden_states)
        hidden_states = self.ff.forward(hidden_states)
        outputs = hidden_states
        return outputs
