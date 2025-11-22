import mlx.core as mx
from mlx import nn


class FiboGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "tanh", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        return hidden_states
