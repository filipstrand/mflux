import mlx.core as mx
from mlx import nn


class Siglip2MLP(nn.Module):
    """SigLIP2-G384 MLP: hidden_size=1536, intermediate_size=6144."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1536, 6144)
        self.fc2 = nn.Linear(6144, 1536)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
