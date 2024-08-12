import math

from mlx import nn
import mlx.core as mx


class T5DenseReluDense(nn.Module):

    def __init__(self):
        super().__init__()
        self.wi_0 = nn.Linear(4096, 10240, bias=False)
        self.wi_1 = nn.Linear(4096, 10240, bias=False)
        self.wo = nn.Linear(10240, 4096, bias=False)

    def forward(self, hidden_states: mx.array) -> mx.array:
        hidden_gelu = T5DenseReluDense.new_gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states

    @staticmethod
    def new_gelu(input_array: mx.array) -> mx.array:
        return 0.5 * input_array * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (input_array + 0.044715 * mx.power(input_array, 3.0))))
