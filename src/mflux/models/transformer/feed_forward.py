import mlx.core as mx
from mlx import nn


class FeedForward(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.linear1 = nn.Linear(3072, 12288)
        self.linear2 = nn.Linear(12288, 3072)
        self.activation_function = activation_function

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation_function(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states
