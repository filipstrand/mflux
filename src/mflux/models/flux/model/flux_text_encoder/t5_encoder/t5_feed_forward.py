import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_dense_relu_dense import T5DenseReluDense
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_layer_norm import T5LayerNorm


class T5FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = T5LayerNorm()
        self.DenseReluDense = T5DenseReluDense()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return hidden_states + forwarded_states
