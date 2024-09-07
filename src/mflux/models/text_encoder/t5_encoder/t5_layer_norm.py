import mlx.core as mx
from mlx import nn


class T5LayerNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = mx.ones((4096,))
        self.variance_epsilon = 1e-06

    def forward(self, hidden_states: mx.array) -> mx.array:
        variance = mx.mean(mx.power(hidden_states.astype(mx.float32), 2), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
