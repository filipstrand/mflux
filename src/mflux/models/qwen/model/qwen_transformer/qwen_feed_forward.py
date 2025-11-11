import mlx.core as mx
from mlx import nn


class QwenFeedForward(nn.Module):
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=True)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.mlp_in(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.mlp_out(hidden_states)
        return hidden_states
