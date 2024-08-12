import mlx.core as mx
from mlx import nn


class CLIPMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dims=768, output_dims=3072)
        self.fc2 = nn.Linear(input_dims=3072, output_dims=768)

    def forward(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = CLIPMLP.quick_gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    @staticmethod
    def quick_gelu(input_array: mx.array) -> mx.array:
        return input_array * nn.sigmoid(1.702 * input_array)
