import math

from mlx import nn
import mlx.core as mx

from mflux.models.text_encoder.t5_encoder.t5_dense_relu_dense import T5DenseReluDense
from mflux.models.text_encoder.t5_encoder.t5_layer_norm import T5LayerNorm


class T5FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_norm = T5LayerNorm()
        self.DenseReluDense = T5DenseReluDense()

    def forward(self, hidden_states: mx.array) -> mx.array:
        forwarded_states = self.layer_norm.forward(hidden_states)
        forwarded_states = self.DenseReluDense.forward(forwarded_states)
        return hidden_states + forwarded_states
