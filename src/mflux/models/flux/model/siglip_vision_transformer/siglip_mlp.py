import mlx.core as mx
from mlx import nn


class SiglipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1152, 4304)
        self.fc2 = nn.Linear(4304, 1152)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)  # Check that this is equivalent to gelu_pytorch_tanh
        hidden_states = self.fc2(hidden_states)
        return hidden_states
