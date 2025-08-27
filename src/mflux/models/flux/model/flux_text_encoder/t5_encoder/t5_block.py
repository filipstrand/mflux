import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_attention import T5Attention
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_feed_forward import T5FeedForward


class T5Block(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.attention = T5Attention()
        self.ff = T5FeedForward()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.attention(hidden_states)
        hidden_states = self.ff(hidden_states)
        outputs = hidden_states
        return outputs
